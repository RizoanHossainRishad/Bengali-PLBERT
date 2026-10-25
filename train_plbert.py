# coding: utf-8
"""Train a Bengali PL-BERT on one GPU.

Single-process, single-GPU rewrite of yl4579/PL-BERT's train.ipynb. The losses
and the checkpoint layout match upstream; the distributed launcher, the
notebook wrapper and the HuggingFace ``datasets`` dependency are gone so this
runs on a plain Windows box with nothing but PyTorch and transformers.

    python train_plbert.py --config config.yml

Training resumes automatically from the newest ``step_*.t7`` in log_dir, so you
can close the machine overnight and start it again with the same command.
"""

import argparse
import csv
import json
import math
import os
import os.path as osp
import pickle
import re
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AlbertConfig, AlbertModel

from dataset import build_dataloader
from env_file import load_env_file
from model import MultiTaskModel

CHECKPOINT_RE = re.compile(r'^step_(\d+)\.t7$')
# Deliberately not step_*.t7, so resume and pruning never pick it up.
BEST_CHECKPOINT = 'best.t7'
BEST_RECORD = 'best.json'
VAL_SEED = 1234


class MetricsLogger:
    """Weights & Biases when available, always a local CSV as well.

    The CSV is not a nicety: this runs on a home PC that may be offline or
    behind a login prompt, and losing 14 hours of loss history because wandb
    could not reach the network would be worse than having no dashboard.
    The W&B run id is persisted so a restart continues the same run instead
    of littering the project with fragments -- the same pattern the StyleTTS2
    trainer uses.
    """

    # Fixed column set. Train and validation rows carry different keys, and a
    # DictWriter built per row would place validation numbers under the train
    # headers -- a silently corrupted loss history.
    FIELDS = (
        'step',
        'train/loss_vocab', 'train/loss_token', 'train/loss_total',
        'train/steps_per_sec', 'train/lr', 'train/grad_norm_max',
        'val/loss_vocab', 'val/loss_token', 'val/loss_total',
    )

    def __init__(self, config, log_dir, resume_step):
        self.csv_path = osp.join(log_dir, 'metrics.csv')
        self._csv_header_written = osp.exists(self.csv_path)
        if self._csv_header_written:
            # Appending rows under a header with different columns would put
            # numbers in the wrong place, so an old-format file is set aside.
            with open(self.csv_path, encoding='utf-8') as handle:
                header = handle.readline().strip().split(',')
            if tuple(header) != self.FIELDS:
                kept = osp.join(
                    log_dir, f'metrics_old_{time.strftime("%Y%m%d_%H%M%S")}.csv'
                )
                os.replace(self.csv_path, kept)
                self._csv_header_written = False
                log(f'metrics    : column set changed, old file kept as {kept}')
        self.run = None

        settings = config.get('wandb') or {}
        if not settings.get('enabled', False):
            log('metrics    : CSV only (set wandb.enabled to use W&B)')
            return
        if 'WANDB_API_KEY' in load_env_file():
            log('metrics    : WANDB_API_KEY loaded from .env')
        try:
            import wandb
        except ImportError:
            log('metrics    : wandb not installed -- CSV only. '
                'Run: pip install wandb')
            return

        id_path = osp.join(log_dir, 'wandb_run_id.txt')
        run_id = None
        if osp.exists(id_path):
            run_id = open(id_path, encoding='utf-8').read().strip() or None
        try:
            self.run = wandb.init(
                project=settings.get('project', 'PLBERT-Bengali'),
                name=settings.get('run_name'),
                id=run_id,
                resume='allow',
                mode=settings.get('mode', 'online'),
                config=config,
            )
            with open(id_path, 'w', encoding='utf-8') as handle:
                handle.write(self.run.id)
            self._wandb = wandb
            wandb.define_metric('train/*', step_metric='step')
            wandb.define_metric('val/*', step_metric='step')
            log(f'metrics    : W&B run {self.run.name} ({self.run.id}) + CSV')
        except Exception as error:
            # An unreachable dashboard must never stop a training run.
            log(f'metrics    : W&B init failed ({error}) -- CSV only')
            self.run = None

    def log(self, step, values):
        row = {'step': step, **values}
        unknown = set(row) - set(self.FIELDS)
        if unknown:
            raise KeyError(f'metric names not in MetricsLogger.FIELDS: {unknown}')
        try:
            write_header = not self._csv_header_written
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=self.FIELDS, restval=''
                )
                if write_header:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(row)
        except OSError as error:
            log(f'  [metrics] could not append to CSV: {error}')
        if self.run is not None:
            try:
                # No step= here: after a resume the trainer replays steps
                # W&B has already seen, and an explicit step lower than the
                # run's internal counter is silently dropped. The 'step' key in
                # row is the x-axis instead, via define_metric in __init__.
                self._wandb.log(row)
            except Exception as error:
                log(f'  [metrics] wandb.log failed: {error}')

    def finish(self):
        if self.run is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def length_to_mask(lengths):
    """True where the position is PADDING."""
    mask = torch.arange(lengths.max()).unsqueeze(0)
    mask = mask.expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def latest_checkpoint(log_dir):
    best_step, best_path = 0, None
    if osp.isdir(log_dir):
        for name in os.listdir(log_dir):
            match = CHECKPOINT_RE.match(name)
            if match and int(match.group(1)) > best_step:
                best_step = int(match.group(1))
                best_path = osp.join(log_dir, name)
    return best_step, best_path


def prune_checkpoints(log_dir, keep):
    found = []
    for name in os.listdir(log_dir):
        match = CHECKPOINT_RE.match(name)
        if match:
            found.append((int(match.group(1)), osp.join(log_dir, name)))
    for _step, path in sorted(found)[:-keep] if keep > 0 else []:
        try:
            os.remove(path)
        except OSError:
            pass


def save_checkpoint(path, model, optimizer, scaler, step, max_positions):
    """Write a checkpoint StyleTTS2's ``load_plbert()`` can read unchanged.

    Two details are load-bearing:
      * every key carries a ``module.`` prefix, because load_plbert strips the
        first seven characters off each key unconditionally;
      * ``embeddings.position_ids`` is re-inserted. transformers >= 4.31 drops
        it from ALBERT's state_dict, but load_plbert still does a bare
        ``del new_state_dict["embeddings.position_ids"]`` and would raise
        KeyError on a checkpoint that lacks it.
    """
    net = {f'module.{key}': value
           for key, value in model.state_dict().items()}
    net['module.encoder.embeddings.position_ids'] = (
        torch.arange(max_positions).unsqueeze(0)
    )
    state = {
        'net': net,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
    }
    temp = f'{path}.tmp'
    torch.save(state, temp)
    os.replace(temp, path)


def compute_losses(tokens_pred, words_pred, words, labels,
                   input_lengths, masked_indices, criterion, device):
    # Word prediction over every real position, averaged per utterance first
    # so that long and short samples count equally (upstream behaviour).
    #
    # Each slice is upcast to fp32 individually rather than calling .float()
    # on the whole tensor. The vocabulary head emits batch x 512 x 30,002
    # values; a full fp32 copy of that is ~940 MB at batch 16, and autograd
    # keeps its gradient too. Upcasting only the non-padding rows avoids
    # roughly a gigabyte on an 8 GB card.
    loss_vocab = 0.0
    for pred, target, length in zip(words_pred, words, input_lengths):
        loss_vocab = loss_vocab + criterion(
            pred[:length].float(), target[:length]
        )
    loss_vocab = loss_vocab / words.size(0)

    # Masked-phoneme prediction, restricted to the masked positions.
    loss_token = 0.0
    counted = 0
    for pred, target, length, masked in zip(
            tokens_pred, labels, input_lengths, masked_indices):
        if not len(masked):
            continue
        index = torch.as_tensor(masked, dtype=torch.long, device=device)
        loss_token = loss_token + criterion(
            pred[:length][index].float(), target[:length][index]
        )
        counted += 1
    if counted:
        loss_token = loss_token / counted
    else:
        loss_token = torch.zeros((), device=device)

    return loss_vocab, loss_token


def load_best(log_dir):
    """Return the best-validation record saved so far, or None."""
    try:
        with open(osp.join(log_dir, BEST_RECORD), encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def save_best_record(log_dir, record):
    path = osp.join(log_dir, BEST_RECORD)
    with open(f'{path}.tmp', 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    os.replace(f'{path}.tmp', path)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, max_batches):
    # Masking and cropping draw from numpy's global RNG. Without a fixed seed
    # every validation sees different masks, so a lower val loss could be luck
    # rather than a better model -- and the best checkpoint is chosen on it.
    # The training RNG state is restored afterwards.
    rng_state = np.random.get_state()
    np.random.seed(VAL_SEED)
    model.eval()
    totals = [0.0, 0.0]
    seen = 0
    try:
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            words, labels, phonemes, input_lengths, masked_indices = batch
            words = words.to(device)
            labels = labels.to(device)
            phonemes = phonemes.to(device)
            text_mask = length_to_mask(
                torch.LongTensor(input_lengths)
            ).to(device)
            with torch.autocast(device, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                tokens_pred, words_pred = model(
                    phonemes, attention_mask=(~text_mask).int()
                )
                loss_vocab, loss_token = compute_losses(
                    tokens_pred, words_pred, words, labels,
                    input_lengths, masked_indices, criterion, device,
                )
            totals[0] += float(loss_vocab)
            totals[1] += float(loss_token)
            seen += 1
    finally:
        np.random.set_state(rng_state)
        model.train()
    if not seen:
        return float('nan'), float('nan')
    return totals[0] / seen, totals[1] / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='override num_steps from the config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='override batch_size (lower this first on OOM)')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'),
                        help='cpu runs a slow sanity check without a GPU')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    if args.num_steps:
        config['num_steps'] = args.num_steps
    if args.batch_size:
        config['batch_size'] = args.batch_size

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit(
            'No CUDA GPU found. Install the CUDA build of PyTorch (see '
            'README), or pass --device cpu for a slow sanity check.'
        )

    precision = str(config.get('precision', 'fp16')).lower()
    if device == 'cpu':
        precision = 'fp32'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if precision == 'fp16':
        amp_dtype, scaler = torch.float16, torch.amp.GradScaler('cuda')
    elif precision == 'bf16':
        amp_dtype, scaler = torch.bfloat16, None
    else:
        amp_dtype, scaler = None, None

    if device == 'cuda':
        log(f'GPU        : {torch.cuda.get_device_name(0)}')
        log(f'VRAM       : {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB')
    else:
        log('device     : CPU (sanity check only -- far too slow to train)')
    log(f'precision  : {precision}')

    data_dir = config['data_folder']
    with open(osp.join(data_dir, 'train.pkl'), 'rb') as handle:
        train_records = pickle.load(handle)
    with open(osp.join(data_dir, 'val.pkl'), 'rb') as handle:
        val_records = pickle.load(handle)
    with open(osp.join(data_dir, 'token_maps.pkl'), 'rb') as handle:
        token_maps = pickle.load(handle)

    num_vocab = 1 + max(entry['token'] for entry in token_maps.values())
    word_separator = config['dataset_params']['word_separator']

    # The separator id is produced by prepare_data.py and merely copied into
    # the config. If the two disagree the dataloader maps the separator onto a
    # real word and every sample is quietly mislabelled, so refuse to start.
    stats_path = osp.join(data_dir, 'stats.json')
    if osp.exists(stats_path):
        import json
        with open(stats_path, encoding='utf-8') as handle:
            stats = json.load(handle)
        expected_separator = stats.get('word_separator_id')
        expected_vocab = stats.get('num_vocab_for_model')
        if expected_separator is not None and expected_separator != word_separator:
            raise SystemExit(
                f'config dataset_params.word_separator is {word_separator} but '
                f'data/stats.json says {expected_separator}. Fix the config.'
            )
        if expected_vocab is not None and expected_vocab != num_vocab:
            raise SystemExit(
                f'token_maps implies num_vocab {num_vocab} but data/stats.json '
                f'says {expected_vocab}. The data folder is inconsistent.'
            )
        log(f'data check : separator id {word_separator}, '
            f'vocab {num_vocab:,} -- consistent with stats.json')
    log(f'train / val: {len(train_records):,} / {len(val_records):,} records')
    log(f'num_vocab  : {num_vocab:,}')

    dataset_config = {
        key: config['dataset_params'][key]
        for key in ('token_separator', 'token_mask', 'max_mel_length',
                    'word_mask_prob', 'phoneme_mask_prob', 'replace_prob')
        if key in config['dataset_params']
    }
    batch_size = config['batch_size']
    num_workers = int(config.get('num_workers', 0))
    train_loader = build_dataloader(
        train_records, token_maps, word_separator, batch_size=batch_size,
        num_workers=num_workers, shuffle=True, pin_memory=(device == 'cuda'),
        dataset_config=dataset_config,
    )
    val_loader = build_dataloader(
        val_records, token_maps, word_separator, batch_size=batch_size,
        num_workers=0, shuffle=False, pin_memory=(device == 'cuda'),
        dataset_config=dataset_config,
    )

    albert_config = AlbertConfig(**config['model_params'])
    encoder = AlbertModel(albert_config)
    model = MultiTaskModel(
        encoder,
        num_tokens=config['model_params']['vocab_size'],
        num_vocab=num_vocab,
        hidden_size=config['model_params']['hidden_size'],
    ).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    encoder_parameters = sum(p.numel() for p in encoder.parameters())
    log(f'parameters : {total_parameters:,} total, '
        f'{encoder_parameters:,} in the encoder StyleTTS2 will keep')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get('learning_rate', 1e-4))
    )
    criterion = nn.CrossEntropyLoss()

    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    step, resume_path = latest_checkpoint(log_dir)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu',
                                weights_only=False)
        state = {key[7:]: value                     # drop the module. prefix
                 for key, value in checkpoint['net'].items()}
        state.pop('encoder.embeddings.position_ids', None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise SystemExit(f'checkpoint is missing tensors: {missing}')
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler is not None and checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        log(f'resumed from {resume_path} at step {step:,}')
    else:
        log('starting from scratch')

    metrics = MetricsLogger(config, log_dir, step)

    num_steps = int(config['num_steps'])
    log_interval = int(config['log_interval'])
    save_interval = int(config['save_interval'])
    val_interval = int(config.get('val_interval', save_interval))
    keep_last = int(config.get('keep_last_checkpoints', 1))
    best = load_best(log_dir)
    if best is not None:
        log(f'best so far: val loss {best["val_loss_total"]:.4f} '
            f'at step {best["step"]:,} ({BEST_CHECKPOINT})')
    max_positions = int(config['model_params']['max_position_embeddings'])

    grad_clip = float(config.get('grad_clip', 0) or 0)
    collapse_margin = float(config.get('collapse_margin', 0) or 0)
    log(f'grad clip  : {grad_clip if grad_clip > 0 else "off"} | '
        f'collapse guard: {collapse_margin if collapse_margin > 0 else "off"}')
    grad_norm_max = 0.0

    accumulation = int(config.get('gradient_accumulation', 1))
    if accumulation < 1:
        raise SystemExit('gradient_accumulation must be 1 or more')
    if accumulation > 1:
        log(f'accumulation: {accumulation} micro-batches of '
            f'{batch_size} -> effective batch {batch_size * accumulation}')

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_batches = 0
    running = [0.0, 0.0]
    counted = 0
    window_start = time.time()
    log(f'training to step {num_steps:,} ...')

    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break
            words, labels, phonemes, input_lengths, masked_indices = batch
            words = words.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            phonemes = phonemes.to(device, non_blocking=True)
            text_mask = length_to_mask(
                torch.LongTensor(input_lengths)
            ).to(device)

            with torch.autocast(device, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                tokens_pred, words_pred = model(
                    phonemes, attention_mask=(~text_mask).int()
                )
            # Losses in fp32: the vocabulary head is wide and fp16 softmax
            # over 30k classes loses too much precision.
            loss_vocab, loss_token = compute_losses(
                tokens_pred, words_pred, words, labels,
                input_lengths, masked_indices, criterion, device,
            )
            loss = loss_vocab + loss_token

            # Gradient accumulation. A bigger batch_size costs VRAM linearly;
            # accumulating N micro-batches reaches the same effective batch at
            # the same peak memory, trading wall-clock per optimizer step
            # instead. Dividing by N keeps the gradient magnitude identical to
            # one real batch of batch_size * N.
            backward_loss = loss / accumulation
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            running[0] += float(loss_vocab)
            running[1] += float(loss_token)
            micro_batches += 1
            if micro_batches % accumulation != 0:
                continue

            # Unscale first so the clip threshold applies to the real
            # gradient, not the fp16 loss-scaled one. A non-finite norm means
            # an overflowed step, which scaler.step() skips on its own.
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip if grad_clip > 0 else float('inf')
            ))
            if math.isfinite(grad_norm):
                grad_norm_max = max(grad_norm_max, grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            counted += 1
            step += 1

            if step % log_interval == 0:
                elapsed = time.time() - window_start
                mean_vocab = running[0] / (counted * accumulation)
                mean_token = running[1] / (counted * accumulation)
                steps_per_second = counted / max(elapsed, 1e-6)
                log(f'step {step:>8,}/{num_steps:,} | '
                    f'vocab {mean_vocab:7.4f} | '
                    f'token {mean_token:7.4f} | '
                    f'grad {grad_norm_max:7.2f} | '
                    f'{steps_per_second:5.2f} steps/s')
                metrics.log(step, {
                    'train/loss_vocab': mean_vocab,
                    'train/loss_token': mean_token,
                    'train/loss_total': mean_vocab + mean_token,
                    'train/steps_per_sec': steps_per_second,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/grad_norm_max': grad_norm_max,
                })
                running = [0.0, 0.0]
                counted = 0
                grad_norm_max = 0.0
                window_start = time.time()

            if step % val_interval == 0:
                v_vocab, v_token = evaluate(
                    model, val_loader, criterion, device, amp_dtype,
                    max_batches=int(config.get('val_batches', 40)),
                )
                log(f'  validation @ {step:,}: '
                    f'vocab {v_vocab:7.4f} | token {v_token:7.4f}')
                v_total = v_vocab + v_token
                metrics.log(step, {
                    'val/loss_vocab': v_vocab,
                    'val/loss_token': v_token,
                    'val/loss_total': v_total,
                })
                # Checked before any save, so a collapsed model never replaces
                # the newest good step_N.t7 at this step.
                if (best is not None and collapse_margin > 0
                        and v_total > best['val_loss_total'] + collapse_margin):
                    metrics.finish()
                    raise SystemExit(
                        f'\nSTOPPED: val loss {v_total:.4f} is more than '
                        f'{collapse_margin} above the best '
                        f'({best["val_loss_total"]:.4f} at step '
                        f'{best["step"]:,}). The model has most likely '
                        f'collapsed and will not recover on its own.\n'
                        f'To restart from the good weights: move every '
                        f'step_*.t7 in {log_dir} into a subfolder, copy '
                        f'{BEST_CHECKPOINT} to step_{best["step"]}.t7, and '
                        f'run training again.'
                    )
                if best is None or v_total < best['val_loss_total']:
                    save_checkpoint(osp.join(log_dir, BEST_CHECKPOINT), model,
                                    optimizer, scaler, step, max_positions)
                    best = {'step': step, 'val_loss_total': v_total,
                            'val_loss_vocab': v_vocab,
                            'val_loss_token': v_token}
                    save_best_record(log_dir, best)
                    log(f'  new best val loss {v_total:.4f} at step '
                        f'{step:,} -> {BEST_CHECKPOINT}')
                window_start = time.time()

            if step % save_interval == 0:
                path = osp.join(log_dir, f'step_{step}.t7')
                save_checkpoint(path, model, optimizer, scaler, step,
                                max_positions)
                prune_checkpoints(log_dir, keep_last)
                log(f'  saved {path}')
                window_start = time.time()

    path = osp.join(log_dir, f'step_{step}.t7')
    save_checkpoint(path, model, optimizer, scaler, step, max_positions)
    prune_checkpoints(log_dir, keep_last)
    metrics.finish()
    log(f'done. final checkpoint: {path}')
    if best is not None:
        log(f'best checkpoint: {osp.join(log_dir, BEST_CHECKPOINT)} '
            f'(step {best["step"]:,}, val loss {best["val_loss_total"]:.4f})')
    log(f'loss history: {osp.join(log_dir, "metrics.csv")}')
    log('Next: python export_for_styletts2.py --checkpoint ' + path)


if __name__ == '__main__':
    main()
