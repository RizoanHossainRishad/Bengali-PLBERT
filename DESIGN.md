# Bengali PL-BERT — design document

Why this sub-project exists, what each decision was, and what was measured
rather than assumed. `README.md` is the operating manual; this is the record of
reasoning. Written 2026-10-22.

---

## 1. The problem

The live StyleTTS2 run (`config_bengali_dominant_40epoch.yaml`,
`Models/Bengali_dominant_schema_v2_40epoch`) trains on data that is **90.2%
Bengali** — 261,623 of 290,060 manifest rows. Its validation scores barely move:

| Stage | Epochs | Best epoch | Score A | spectral | duration MAE |
|---|---|---|---|---|---|
| stage_1 base | 1–10 | 6 | 0.35680 | 0.28284 | 0.52948 |
| stage_2 diffusion | 11–20 | 18 | 0.34656 | 0.26878 | 0.52815 |
| stage_3 joint SLM | 21–40 | 24 | 0.35403 | 0.27922 | 0.52868 |

Across 25 epochs and roughly half a million steps, Score A improved ~3% and
**duration MAE did not move at all** (0.529 → 0.528 → 0.529). Stage 3 is worse
than stage 2's best. The model reached most of its quality by epoch 6.

Measured context for that duration number: tokens average **3.45 aligner frames**
(p95 6.11, max 11.43, from a 300-file sample of the manifest). A MAE of 0.53
frames is therefore a **~15% relative error on every phoneme's length**, static
across the whole run.

## 2. Why PL-BERT is the suspect

`Utils/PLBERT/config.yml`:

```yaml
data_folder: "wikipedia_20220301.en.processed"
tokenizer: "transfo-xl-wt103"
```

The shipped PL-BERT is pretrained on **English Wikipedia only**. Upstream's own
README (line 70) says it "probably does not work very well on other languages,
so you will need to train a different PL-BERT for different languages".

Its output feeds two places, not one:

```
texts ──► bert (PL-BERT) ──► bert_dur
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
      bert_encoder                      style-diffusion conditioning
            │                           (sampler embedding=bert_dur)
            ▼
       predictor
       ├─ duration_proj  → how long each token lasts
       └─ F0Ntrain       → pitch and energy contours
```

Those are exactly the two things that are not improving. `bert` is fine-tuned
during the run, but at `bert_lr: 1e-5` from an English starting point — far too
little to relocate a 12-layer ALBERT into another language.

This is a **well-supported hypothesis, not a proven cause.** It will only be
confirmed by the warm-start run in section 8.

### What was ruled out first

| Suspect | Verdict | Evidence |
|---|---|---|
| Ready-made multilingual PL-BERT | Unusable | `papercup-ai/multilingual-pl-bert` covers 16 subsets (en, en-xl, ca, de, es, el, fa, fi, fr, it, pl, pt, ru, sv, uk, zh). **No Bengali.** |
| `max_dur: 50` clipping durations | Not the cause | Measured p95 = 6.11, max = 11.43 aligner frames per token. Nowhere near 50. |
| ASR text aligner is English-only | Minor | Pretrained on English + Japanese + Chinese; upstream says it transfers. |
| JDC pitch extractor is English-only | Not an issue | F0 is language-independent. |

## 3. Why training our own is cheap

| | |
|---|---|
| Encoder parameters | **6,292,480** |
| Released checkpoint size | 25 MB |
| Architecture | ALBERT, 12 layers sharing one weight group |
| StyleTTS2 checkpoint, for comparison | 2.26 GB |

PL-BERT is roughly **100× smaller than the TTS model**. A consumer GPU can
train it overnight. That is what makes this worth doing at all.

## 4. Data pipeline — the fidelity problem

This was the hardest decision and it is worth stating plainly.

**PL-BERT must see the phoneme strings the TTS model actually sees.** If the
pronunciation of a word differs between the two, PL-BERT's embedding for that
phoneme sequence is indexed by a pronunciation StyleTTS2 never produces, and
the transfer degrades silently — no error, just a worse model.

The real chain that produced the training manifests is:

```
raw text
  └─► bangla/preprocessing_task/2_text_process.py :: normalize_text_bn
        (dates, currency, percentages, decimals, ordinals, time,
         abbreviations, numbers → Bengali words; emoji stripped)
        └─► transcript_normalized
              └─► bangla/Dataset_unification/3_g2p_pipeline.py
                    :: phonemize_language_spans
                    (120 curated bn overrides + 3 en, bn/en-us span routing,
                     sanitize_model_phonemes)
                    └─► phonemized_transcription  +  « stop token
```

### 4.1 First attempt, and why it was wrong

The first version of `prepare_data.py` reimplemented G2P: one `bn` eSpeak
backend, word by word, no override table, no language routing, no
normalization. Measured against 400 manifest rows:

| | exact sentence match | character similarity |
|---|---|---|
| Reimplementation | **88.2%** | 99.7% |

Three distinct defects showed up in the diff:

* `খুলনা` → reimplementation `kʰˈulɔnˌa`, manifest `khulna` — the curated
  override table was never consulted.
* `…tʰˈakɔtʃʰˌen , kˈaɟ` vs `…tʰˈakɔtʃʰˌen, kˈaɟ` — joining tokens with spaces
  puts a space before punctuation that eSpeak never emits inside a span.
* Latin-script words were sent to the Bengali backend instead of `en-us`.

### 4.2 Second attempt: import the real rules

Rewriting the word-level logic on top of the *imported* `3_g2p_pipeline.py`
(its override tables, token pattern, punctuation normaliser and sanitiser)
raised fidelity to **95.5% exact / 99.97% character**. The residue was eSpeak's
own cross-word behaviour inside a span — hyphen joins (`কী-যে` → one unit), a
dropped sentence-final period, and punctuation with no following space. Those
cannot be reproduced word by word, because the information only exists when
eSpeak sees the whole span.

### 4.3 What was actually shipped

Phonemize the **whole sentence** with the real pipeline, then split the result
on whitespace and align to the whitespace words of the source.

Measured on 2,000 manifest rows: **1,999 (99.95%)** have matching word and
phoneme-group counts. The single failure was `ঢাকা-রাজশাহী` — one whitespace
word, two phoneme groups, because `sanitize_model_phonemes` turns `-` into a
space. The source text is therefore split on `-` as well, which removes that
case.

**On every record that survives, the phonemes are the dataset pipeline's
verbatim output.** Fidelity is 100% by construction rather than by measurement;
records that cannot be aligned unambiguously are dropped instead of guessed.

### 4.4 Normalization for Wikipedia

Manifest transcripts arrive already normalized — they *are* the text the TTS
model trained on — so they pass through untouched. Raw Wikipedia does not, and
skipping that stage was a real defect in the first build:

```
raw  : ১৯৭১ সালের ২৬শে মার্চ বাংলাদেশ স্বাধীনতা ঘোষণা করে।
norm : এক হাজার নয়শত একাত্তর সালের ছাব্বিশশে মার্চ বাংলাদেশ স্বাধীনতা ঘোষণা করে.

raw  : জনসংখ্যা ১৬,৫০,০০০ জন এবং বৃদ্ধির হার ১.২%।
norm : জনসংখ্যা ষোল লাখ পঞ্চাশ হাজার জন এবং বৃদ্ধির হার এক দশমিক দুই শতাংশ.
```

Without it PL-BERT would learn to pronounce digits and dates literally —
something that never happens at inference, because `Services/backend/
text_preprocessing.py` expands them to words before the engine phonemizes.

`normalize_text_bn` is **98.4% idempotent** on already-normalized transcripts,
so applying it to both would have been nearly harmless; it is applied only to
Wikipedia so the manifest side stays byte-exact.

### 4.5 Alphabet guarantee

`text_utils.py` is generated by copying the symbol block out of StyleTTS2's
`meldataset.py`, and the copy is asserted equal to the original (`symbols` list
and `dicts` mapping both compared). The table has **178 slots but 177 distinct
characters** — `'` appears twice in `_letters_ipa`, so one embedding row is
unreachable. `n_token: 178` refers to the slot count and must not be changed.

Every phoneme group is then checked against that alphabet and the record is
rejected if anything falls outside. In the shipped build this rejected **25
texts**, all containing `(` or `)` picked up from Wikipedia — characters the
TTS model has no embedding for.

### 4.6 eSpeak wedges, and the watchdog

Two full builds stalled at the same point in the Wikipedia stream: every worker
at ~2% CPU, no progress for over ten minutes. Some particular input wedges
eSpeak, and a hung call cannot be recovered from inside the process.

Two defences, both in `server_side/prepare_data.py`:

1. a 5-second `SIGALRM` around each phonemize call;
2. chunks submitted with `apply_async` and collected with a 90-second deadline,
   instead of `imap` — `imap` yields strictly in order, so one wedged chunk
   blocked collection even after the other fifteen workers had finished
   everything else. That is precisely what happened.

The shipped build abandoned **4 chunks (~800 texts)** this way. The corpus is
several times larger than it needs to be, so that costs nothing.

### 4.7 The resulting corpus

| | |
|---|---|
| Source: TTS transcripts | 234,751 |
| Source: Bengali Wikipedia sentences | 623,769 |
| Total source text | 65.3 M characters ≈ **196 MB UTF-8** |
| Aligned (word count == phoneme-group count) | 763,343 (**88.91%**) |
| Rejected for out-of-alphabet characters | 25 |
| Packed documents | 37,518 |
| Training / validation records | 36,768 / 750 |
| Phoneme characters | 65,919,381 |
| Unique words | 481,002 |
| Word vocabulary | 30,000 (**87.62%** token coverage) |

Upstream guidance is "a text dataset of more than 30 MB is sufficient"; this is
roughly 6×.

The alignment rate is lower than the 99.95% measured on manifest rows alone,
because Wikipedia prose carries far more hyphens, brackets and quotes. Dropping
11% of a corpus that is already 6× oversized is the right trade against
guessing an alignment.

### 4.8 Why the word vocabulary is 30,000

PL-BERT's second head predicts which word a phoneme span belongs to. That head
is `nn.Linear(768, num_vocab)`, so the vocabulary directly sets both parameter
count and the size of the logits tensor — on an 8 GB card that is the dominant
activation.

Measured coverage:

| Vocabulary | Token coverage |
|---|---|
| 10,000 | 78.65% |
| 20,000 | 84.68% |
| **30,000** | **87.62%** |
| 40,000 | 89.45% |
| 50,000 | 90.73% |
| 80,000 | 93.06% |

At sequence 512 in fp16 the logits tensor is ~29 MB per sample at 30k and
~49 MB at 50k, and autograd also keeps an fp32 copy and its gradient. 30,000
buys 87.6% coverage for the smaller footprint. Raise it if the card turns out
to have room; nothing else changes.

**Correction, 2026-10-22.** The first version of this package shipped
`batch_size: 16` on the strength of an estimate. It does not fit an 8 GB card:
measured breakdown at sequence 512 is 2.29 GB of word logits (fp16 tensor plus
the fp32 copy plus its gradient), 2.25 GB of attention scores across the twelve
layer applications, 2.06 GB of hidden states and 0.46 GB of weights and
optimizer state — about 7.06 GB before the CUDA context and the display driver.
The default is now 8, and `compute_losses` upcasts each utterance slice
individually instead of calling `.float()` on the whole batch, which alone
saved roughly a gigabyte. `check_setup.py` now walks a descending ladder and
reports the largest batch size that actually fits, so the number no longer
depends on anyone's estimate.

### 4.9 Word-level, not sub-word

Upstream uses `transfo-xl-wt103`, which is word-level. A sub-word tokenizer
(mBERT, XLM-R) splits Bengali into pieces like `##ের`, which have no sensible
pronunciation in isolation — and the phoneme↔token alignment PL-BERT needs is
per word. StyleTTS2 discards the word-prediction head entirely, so this choice
affects training only and is free to differ from upstream.

## 5. Model and training

Unchanged from upstream in every respect that touches the checkpoint:

```python
MultiTaskModel(
    encoder = AlbertModel(AlbertConfig(**model_params)),   # 6.29 M params
    mask_predictor = nn.Linear(768, 178),                  # masked phonemes
    word_predictor = nn.Linear(768, 30002),                # phoneme → word
)
loss = loss_vocab + loss_token      # both plain CrossEntropyLoss
optimizer = AdamW(lr=1e-4)
```

The attribute name `encoder` is load-bearing: `load_plbert()` selects
checkpoint keys by the `encoder.` prefix and discards everything else.

What was changed, and why:

| Upstream | Here | Reason |
|---|---|---|
| `train.ipynb` + `notebook_launcher`, 3 processes | plain script, 1 GPU | runs on Windows with no accelerate/DDP setup |
| HuggingFace `datasets` on disk | two pickles | one less dependency on the training box |
| no resume | resumes from newest `step_*.t7` | a home PC gets switched off |
| no validation | held-out set every `val_interval` | otherwise overfitting is invisible |
| no metrics | W&B + `metrics.csv` | see section 7 |
| losses in autocast dtype | losses forced to fp32 | fp16 softmax over 30k classes loses too much precision |

Loss baselines, for reading the curves: random guessing gives
**ln(30002) ≈ 10.31** for `loss_vocab` and **ln(178) ≈ 5.18** for `loss_token`.

## 6. Checkpoint format and two silent traps

`load_plbert()` is seven lines of key surgery, and both of its assumptions can
fail quietly.

**Trap 1 — `strict=False`.** The function ends with
`bert.load_state_dict(new_state_dict, strict=False)`. If key names disagree,
the affected tensors stay at their random initialisation and **nothing is
printed**. Training would run to completion and produce a worse model with no
error anywhere. `server_side/verify_checkpoint.py` exists for this: it must
report `MATCHED: 25`, `MISSING: 0`, `SHAPE MISMATCHES: 0`.

**Trap 2 — `position_ids`.** `Utils/PLBERT/util.py` line 39 does an
unconditional `del new_state_dict["embeddings.position_ids"]`. transformers
≥ 4.31 no longer puts that key in ALBERT's `state_dict`, so a freshly trained
checkpoint makes the stock loader raise `KeyError`. Rather than patching a file
inside a live training tree, `train_plbert.py` re-inserts the tensor when
saving. The stock loader then works unchanged — verified end to end.

Every saved key is also prefixed with `module.`, because the loader strips
exactly seven characters off each key (`k[7:]`) on the assumption it was saved
from a DataParallel wrapper.

## 7. Metrics

W&B plus an unconditional local CSV. The CSV is not redundancy for its own
sake: this runs on a home machine that may be offline or not logged in, and
losing a night of loss history to a network error would be worse than having no
dashboard. The W&B run id is persisted to `log_dir/wandb_run_id.txt` so a
restart continues the same run instead of fragmenting it — the same pattern the
StyleTTS2 trainer uses with `wandb_run_id` in its config.

The CSV uses a fixed column set. An earlier version built the `DictWriter`
fieldnames from each row, which wrote validation numbers underneath the
training headers — a silently corrupted history. Metric names outside
`MetricsLogger.FIELDS` now raise instead.

## 8. Integration path

```
checkpoints/step_200000.t7   (333 MB: model + AdamW moments + scaler)
      │  export_for_styletts2.py
      ▼
export/step_200000.t7        (24 MB: the 25 encoder tensors + position_ids)
      │  upload to the server as Utils/PLBERT_bn/
      ▼
server_side/verify_checkpoint.py        → must print MATCHED: 25
      ▼
server_side/warm_start_from_checkpoint.py
      │  replaces net['bert'] (25 tensors)
      │  preserves the other 1,435 tensors across 12 modules
      │  drops the optimizer state (its moments belong to the old embeddings)
      ▼
Models/<run>/warmstart_bn_plbert.pth
      │  pretrained_model: <that file>
      │  load_only_params: true
      │  PLBERT_dir: Utils/PLBERT_bn/
      ▼
resume StyleTTS2 training
```

The decoder's 678 tensors, both style encoders, the text aligner and the
diffusion stack all carry over untouched. None of their knowledge is
invalidated by changing the text encoder's prosody context, so there is no
reason to start from scratch.

## 9. What was actually verified

Everything below was executed, not reasoned about. **No step ran on a GPU** —
both server cards were busy with the 40-epoch run — so all of it is CPU.

| Check | Result |
|---|---|
| Reference pipeline reproduces the manifest CSV | 400/400 = 100% (harness validity) |
| Shipped alignment approach | 1,999/2,000 = 99.95% of rows align |
| `text_utils.symbols` vs `meldataset.symbols` | identical list and dict |
| Bengali phonemes outside the alphabet | 0 over 269,897 characters |
| Training loop | loss falls 9.85 → 8.50 (vocab) and 4.60 → 3.78 (token) |
| Encoder parameter count | 6,292,480 — matches the English PL-BERT exactly |
| Resume | picked up at step 12, continued to 15 |
| Checkpoint pruning | kept the newest 2, removed the rest |
| `stats.json` consistency check | separator 30001 / vocab 30,002 accepted |
| `metrics.csv` | columns align across train and validation rows |
| Export | 333 MB → 24.0 MB, 26 tensors, all identical to the source |
| `verify_checkpoint.py` | MATCHED 25, MISSING 0, SHAPE MISMATCHES 0 |
| **Stock, unpatched `load_plbert()`** | loads; weights equal the file, not random |
| `build_model()` with the new encoder | 13 modules, `bert_encoder.in_features` 768 |
| `warm_start_from_checkpoint.py` | 25 replaced, 1,435 preserved byte-identical |

## 10. Open risks

1. **No GPU run.** `batch_size: 16` is an estimate. If it does not fit 8 GB,
   `README.md` section 3 has the knobs in the order to turn them. Nothing about
   the checkpoint changes.
2. **The hypothesis is unproven.** A Bengali PL-BERT is the best-supported
   explanation for the flat duration MAE, not a demonstrated one. The warm-start
   run is the experiment. Watch `eval/duration_alignment_mae`: if it does not
   move within a few epochs, the bottleneck is elsewhere.
3. **87.62% vocabulary coverage** means 12% of word positions train against
   `<unk>`. Acceptable for an auxiliary task; raise the vocabulary if VRAM
   allows.
4. **11% of source texts were dropped** at alignment. The loss is biased toward
   hyphen- and bracket-heavy Wikipedia prose rather than being random. Given
   the corpus is 6× the recommended size this is judged acceptable, not ideal.
5. **`transformers` is pinned to 4.36.2** to match the server. A newer version
   that renames ALBERT keys would break the handoff; `verify_checkpoint.py`
   would catch it before any damage.
