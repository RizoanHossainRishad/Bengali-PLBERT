#!/usr/bin/env python
"""Build the Bengali PL-BERT corpus using the project's OWN G2P rules.

Runs on the Linux server (needs espeak-ng + phonemizer). Produces a
self-contained data/ folder the Windows box can train from with no Bengali
tooling installed.

Fidelity is the whole point of this script
------------------------------------------
PL-BERT must see the same phoneme strings the TTS model sees, or its
embeddings are indexed by a different pronunciation of the same word. So this
does NOT reimplement G2P. It imports `3_g2p_pipeline.py` -- the exact file that
produced the training manifests -- and calls its `phonemize_language_spans`,
which brings the curated override table, the bn/en-us span routing and
`sanitize_model_phonemes` along with it.

PL-BERT additionally needs per-word phonemes, which that pipeline does not
emit. Instead of phonemizing word by word (measured: only 95.5% of sentences
then match the manifests, because eSpeak applies cross-word effects inside a
span), this phonemizes the whole sentence and splits the result on whitespace.
Sentences whose word count and phoneme-group count disagree are dropped, so
every record that survives carries the pipeline's verbatim output.
"""

import argparse
import collections
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import signal
import pickle
import random
import re
import sys

csv.field_size_limit(10 ** 7)

DEFAULT_PIPELINE = (
    "/mnt/newworkspace/rizoan/preprocess/"
    "Bilingual-Automatic-Text-to-Speech-Recognition/bangla/"
    "Dataset_unification/3_g2p_pipeline.py"
)
DEFAULT_NORMALIZER = (
    "/mnt/newworkspace/rizoan/preprocess/"
    "Bilingual-Automatic-Text-to-Speech-Recognition/bangla/"
    "preprocessing_task/2_text_process.py"
)
BENGALI_RANGE = re.compile(r'[ঀ-৿]')
SENTENCE_SPLIT = re.compile(r'(?<=[।॥.!?])\s+')

_WORKER = {}

# eSpeak occasionally wedges on a particular input -- two full runs of this
# script stalled at the same point in the Wikipedia stream with every worker
# at ~2% CPU and no progress for over ten minutes. A hung eSpeak call cannot
# be recovered from inside the process, so each one runs under an alarm and
# the offending text is dropped. The corpus is far larger than needed, so
# losing a handful of sentences costs nothing; hanging the build costs hours.
PHONEMIZE_TIMEOUT_SECONDS = 5
# A chunk of 200 short sentences finishes in well under a second; 90 s means
# the worker holding it is wedged, not busy.
CHUNK_TIMEOUT_SECONDS = 90
MAX_ABANDONED_CHUNKS = 40


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def log(message):
    print(message, flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pipeline(path):
    return load_module("dataset_g2p", path)


def source_words(text, pipeline):
    """Whitespace words, split the same way sanitize_model_phonemes splits."""
    normalized = pipeline.normalize_punctuation_bn(text).replace('-', ' ')
    return normalized.split()


def _init_worker(pipeline_path, normalizer_path):
    from phonemizer.backend import EspeakBackend
    pipeline = load_pipeline(pipeline_path)
    _WORKER['pipeline'] = pipeline
    _WORKER['normalizer'] = load_module("dataset_normalizer", normalizer_path)
    _WORKER['bn'] = EspeakBackend(
        language='bn', preserve_punctuation=True, with_stress=True
    )
    _WORKER['en'] = EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True
    )
    signal.signal(signal.SIGALRM, _raise_timeout)


def _phonemize_chunk(items):
    """Return [(words, phoneme_groups) | None, ...] for one chunk.

    ``items`` are ``(text, needs_normalization)`` pairs. Manifest transcripts
    arrive already normalised -- they ARE the text the TTS model trained on --
    so they pass through untouched. Raw Wikipedia must go through
    ``normalize_text_bn`` first, or PL-BERT would learn to pronounce digits
    and dates literally, which never happens at inference because the serving
    stack expands them to words beforehand.
    """
    pipeline = _WORKER['pipeline']
    normalizer = _WORKER['normalizer']
    results = []
    for text, needs_normalization in items:
        if needs_normalization:
            text = normalizer.normalize_text_bn(text)
            if not text:
                results.append(None)
                continue
        try:
            signal.alarm(PHONEMIZE_TIMEOUT_SECONDS)
            try:
                phonemes = pipeline.phonemize_language_spans(
                    text, _WORKER['bn'], _WORKER['en']
                )
            finally:
                signal.alarm(0)
        except Exception:
            results.append(None)
            continue
        words = source_words(text, pipeline)
        groups = phonemes.split()
        # Alignment is only unambiguous when the counts agree.
        results.append((words, groups) if words and len(words) == len(groups)
                       else None)
    return results


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def read_transcripts(csv_path):
    out = []
    with open(csv_path, encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            text = (row.get('transcript_normalized') or '').strip()
            if text:
                out.append(text)
    return out


def bengali_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(BENGALI_RANGE.match(c)) for c in letters) / len(letters)


def read_wikipedia(parquet_dir, budget_chars, max_unit_chars):
    """Yield sentence-sized units; eSpeak is happiest below a few hundred chars."""
    import pyarrow.parquet as pq

    units, total = [], 0
    files = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.endswith('.parquet')
    )
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=['text']):
            for value in batch.column('text'):
                article = value.as_py()
                if not article:
                    continue
                for para in article.split('\n'):
                    para = para.strip()
                    if len(para) < 80 or bengali_ratio(para) < 0.80:
                        continue
                    for sentence in SENTENCE_SPLIT.split(para):
                        sentence = sentence.strip()
                        if not (20 <= len(sentence) <= max_unit_chars):
                            continue
                        units.append(sentence)
                        total += len(sentence)
                        if total >= budget_chars:
                            return units, total
    return units, total


def pack(units, target_words, shuffle_seed=None):
    """Group aligned (words, phonemes) units into longer documents."""
    if shuffle_seed is not None:
        units = list(units)
        random.Random(shuffle_seed).shuffle(units)
    documents, words, phonemes = [], [], []
    for unit_words, unit_phonemes in units:
        words.extend(unit_words)
        phonemes.extend(unit_phonemes)
        if len(words) >= target_words:
            documents.append((words, phonemes))
            words, phonemes = [], []
    if len(words) >= 8:
        documents.append((words, phonemes))
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--styletts2-root', required=True)
    parser.add_argument('--pipeline', default=DEFAULT_PIPELINE)
    parser.add_argument('--normalizer', default=DEFAULT_NORMALIZER)
    parser.add_argument('--transcripts-csv', required=True)
    parser.add_argument('--wiki-dir', default=None)
    parser.add_argument('--wiki-mb', type=float, default=60.0)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--vocab-size', type=int, default=30000)
    parser.add_argument('--doc-words', type=int, default=220)
    parser.add_argument('--max-unit-chars', type=int, default=400)
    parser.add_argument('--val-fraction', type=float, default=0.02)
    parser.add_argument('--processes', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    sys.path.insert(0, args.styletts2_root)
    from meldataset import symbols as symbol_list, dicts   # noqa: E402
    allowed = set(dicts)
    # The list has 178 entries (n_token) but one, "'", appears twice, so only
    # 177 distinct characters can ever be produced.
    log(f'[1/7] StyleTTS2 alphabet: {len(symbol_list)} slots, '
        f'{len(allowed)} distinct characters')

    pipeline = load_pipeline(args.pipeline)
    log(f'[2/7] G2P rules imported from {args.pipeline}')
    log(f'      curated overrides: {len(pipeline.TRAINING_OVERRIDES_BN)} bn + '
        f'{len(pipeline.TRAINING_OVERRIDES_EN)} en')

    transcripts = read_transcripts(args.transcripts_csv)
    log(f'[3/7] TTS transcripts : {len(transcripts):,}')
    wiki = []
    if args.wiki_dir:
        wiki, wiki_chars = read_wikipedia(
            args.wiki_dir, int(args.wiki_mb * 1e6), args.max_unit_chars
        )
        log(f'      Wikipedia units : {len(wiki):,} '
            f'({wiki_chars / 1e6:.1f} M chars)')

    # (text, needs_normalization)
    texts = ([(t, False) for t in transcripts]
             + [(t, True) for t in wiki])
    total_chars = sum(len(t) for t, _ in texts)
    log(f'      total text      : {total_chars / 1e6:.1f} M chars '
        f'(~{total_chars * 3 / 1e6:.0f} MB utf-8)')

    log(f'[4/7] Normalizing Wikipedia with {args.normalizer}')
    log(f'      then phonemizing everything with the dataset pipeline '
        f'({args.processes} processes) ...')
    # Chunks are submitted individually and collected with a deadline rather
    # than through imap, which yields strictly in order: a single wedged
    # eSpeak call would then block the whole collection even though the other
    # fifteen workers had finished everything else. That is exactly what
    # happened twice before the timeout below was added.
    chunk_size = 200
    chunks = list(chunked(texts, chunk_size))
    aligned, dropped, abandoned = [], 0, 0
    with mp.Pool(args.processes, initializer=_init_worker,
                 initargs=(args.pipeline, args.normalizer)) as pool:
        pending = [pool.apply_async(_phonemize_chunk, (chunk,))
                   for chunk in chunks]
        for index, handle in enumerate(pending, start=1):
            try:
                result = handle.get(timeout=CHUNK_TIMEOUT_SECONDS)
            except mp.TimeoutError:
                abandoned += 1
                dropped += chunk_size
                if abandoned > MAX_ABANDONED_CHUNKS:
                    log(f'      giving up on the remaining chunks after '
                        f'{abandoned} timeouts; the corpus already collected '
                        f'is more than enough')
                    break
                continue
            for item in result:
                if item is None:
                    dropped += 1
                else:
                    aligned.append(item)
            if index % 250 == 0:
                log(f'      {index * chunk_size:,}/{len(texts):,} texts '
                    f'({len(aligned):,} aligned)')
        pool.terminate()
    if abandoned:
        log(f'      abandoned chunks: {abandoned} '
            f'(a wedged eSpeak call, ~{abandoned * chunk_size:,} texts)')
    rate = 100 * len(aligned) / max(len(texts), 1)
    log(f'      aligned         : {len(aligned):,} ({rate:.2f}%)')
    log(f'      dropped         : {dropped:,} (word/phoneme counts disagreed)')

    # Hard guarantee: nothing outside StyleTTS2's alphabet may reach training.
    bad = collections.Counter()
    clean = []
    for words, groups in aligned:
        offenders = {c for g in groups for c in g if c not in allowed}
        if offenders:
            bad.update(offenders)
            continue
        clean.append((words, groups))
    log(f'      alphabet-clean  : {len(clean):,} '
        f'(rejected {len(aligned) - len(clean):,})')
    if bad:
        log(f'      out-of-alphabet characters seen: '
            f'{ {repr(c): n for c, n in bad.most_common(10)} }')

    log('[5/7] Packing documents and building the vocabulary ...')
    documents = pack(clean, args.doc_words, shuffle_seed=args.seed)
    frequency = collections.Counter()
    for words, _groups in documents:
        frequency.update(words)
    grand_total = sum(frequency.values())
    log(f'      documents       : {len(documents):,}')
    log(f'      word tokens     : {grand_total:,}')
    log(f'      unique words    : {len(frequency):,}')

    ordered = frequency.most_common()
    running = 0
    marks = {10000, 20000, 30000, 40000, 50000, 80000}
    log('      coverage by vocabulary size:')
    for index, (_word, count) in enumerate(ordered, start=1):
        running += count
        if index in marks:
            log(f'          {index:>7,} -> {running / grand_total * 100:5.2f}%')

    kept = ordered[:args.vocab_size]
    word_to_id = {word: index for index, (word, _c) in enumerate(kept)}
    unk_id = len(word_to_id)
    separator_id = unk_id + 1
    coverage = sum(c for _w, c in kept) / grand_total
    log(f'      vocabulary      : {len(word_to_id):,} '
        f'(coverage {coverage * 100:.2f}%)')

    token_maps = {i: {'token': i, 'word': w} for w, i in word_to_id.items()}
    token_maps[unk_id] = {'token': unk_id, 'word': '<unk>'}
    token_maps[separator_id] = {'token': separator_id, 'word': '<sep>'}

    log('[6/7] Assembling records ...')
    records = []
    for words, groups in documents:
        phonemes = [g for g in groups if g]
        input_ids = [word_to_id.get(w, unk_id)
                     for w, g in zip(words, groups) if g]
        if len(phonemes) >= 8 and sum(len(p) for p in phonemes) >= 64:
            records.append({'phonemes': phonemes, 'input_ids': input_ids})
    random.Random(args.seed + 1).shuffle(records)
    n_val = max(64, int(len(records) * args.val_fraction))
    val, train = records[:n_val], records[n_val:]
    phoneme_chars = sum(sum(len(p) for p in r['phonemes']) for r in records)
    log(f'      records         : {len(records):,} '
        f'(train {len(train):,} / val {len(val):,})')
    log(f'      phoneme chars   : {phoneme_chars:,}')

    log('[7/7] Writing ...')
    os.makedirs(args.out_dir, exist_ok=True)
    for name, payload in (('train', train), ('val', val)):
        with open(os.path.join(args.out_dir, f'{name}.pkl'), 'wb') as handle:
            pickle.dump(payload, handle, protocol=4)
    with open(os.path.join(args.out_dir, 'token_maps.pkl'), 'wb') as handle:
        pickle.dump(token_maps, handle, protocol=4)

    stats = {
        'g2p_source': os.path.abspath(args.pipeline),
        'normalizer_source': os.path.abspath(args.normalizer),
        'normalizer_applied_to': 'wikipedia only (transcripts arrive normalized)',
        'g2p_overrides_bn': len(pipeline.TRAINING_OVERRIDES_BN),
        'g2p_overrides_en': len(pipeline.TRAINING_OVERRIDES_EN),
        'symbol_table_slots': len(symbol_list),
        'symbol_table_distinct_characters': len(allowed),
        'source_texts': len(texts),
        'source_characters': total_chars,
        'aligned_texts': len(aligned),
        'alignment_rate': rate / 100,
        'abandoned_chunks': abandoned,
        'alphabet_clean_texts': len(clean),
        'out_of_alphabet_characters': len(bad),
        'documents': len(documents),
        'records_train': len(train),
        'records_val': len(val),
        'phoneme_characters': phoneme_chars,
        'unique_words': len(frequency),
        'vocabulary_size': len(word_to_id),
        'num_vocab_for_model': separator_id + 1,
        'unk_id': unk_id,
        'word_separator_id': separator_id,
        'token_coverage': coverage,
        'tts_transcripts': len(transcripts),
        'wiki_units': len(wiki),
    }
    with open(os.path.join(args.out_dir, 'stats.json'), 'w') as handle:
        json.dump(stats, handle, indent=2)
    log(json.dumps(stats, indent=2))
    log('Done.')


if __name__ == '__main__':
    main()
