# coding: utf-8
"""Preflight check. Run this BEFORE starting a 14-hour training run.

    python check_setup.py

It verifies the interpreter, the packages, the GPU, the data files and the
config, then runs two real training steps at the configured batch size and
reports the peak VRAM actually used. That last number is the one that decides
whether batch_size needs lowering -- on this machine, not in theory.
"""

import os
import os.path as osp
import platform
import sys
import traceback

PASS, FAIL, WARN = 'PASS', 'FAIL', 'WARN'
results = []


def record(status, name, detail=''):
    results.append((status, name, detail))
    mark = {'PASS': '  ok  ', 'FAIL': ' FAIL ', 'WARN': ' warn '}[status]
    print(f'[{mark}] {name}' + (f'\n           {detail}' if detail else ''),
          flush=True)


def section(title):
    print(f'\n--- {title} ' + '-' * max(0, 58 - len(title)), flush=True)


def main():
    # --device cpu exercises the same two steps without a GPU, which is useful
    # before CUDA is working and on machines where the card is busy.
    force_cpu = '--device' in sys.argv and 'cpu' in sys.argv
    print('=' * 68)
    print(' Bengali PL-BERT -- setup check')
    print('=' * 68)

    # ---------------------------------------------------------- interpreter
    section('interpreter')
    version = sys.version_info
    text = f'{version.major}.{version.minor}.{version.micro} ({platform.machine()})'
    if version[:2] < (3, 9):
        record(FAIL, 'Python version', f'{text} -- need 3.9 or newer')
    elif version[:2] > (3, 12):
        record(FAIL, 'Python version',
               f'{text} -- 3.13+ has no Windows wheels for tokenizers/numpy. '
               'Install Python 3.12 and rebuild the venv.')
    else:
        record(PASS, 'Python version', text)
    if sys.maxsize < 2 ** 32:
        record(FAIL, 'Interpreter is 32-bit', 'A 64-bit Python is required.')
    inside_venv = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    record(PASS if inside_venv else WARN, 'Running inside a venv',
           sys.prefix if inside_venv else 'not in a venv -- that is allowed '
           'but the README assumes one')

    # ---------------------------------------------------------- packages
    section('packages')
    try:
        import torch
    except ImportError:
        record(FAIL, 'torch installed',
               'pip install -r requirements.txt  (torch is in there; it comes '
               'from the PyTorch index, not PyPI)')
        summarise()
        return
    record(PASS, 'torch installed', torch.__version__)

    if '+cu' not in torch.__version__ and not torch.cuda.is_available():
        record(FAIL, 'torch is the CUDA build',
               f'{torch.__version__} looks like the CPU-only wheel. '
               'pip uninstall torch, then pip install -r requirements.txt')
    else:
        record(PASS, 'torch is the CUDA build', torch.version.cuda or 'n/a')

    for module, label in (('transformers', 'transformers'), ('yaml', 'PyYAML'),
                          ('numpy', 'numpy')):
        try:
            loaded = __import__(module)
            record(PASS, f'{label} installed',
                   getattr(loaded, '__version__', 'ok'))
        except ImportError:
            record(FAIL, f'{label} installed', 'pip install -r requirements.txt')
    try:
        import wandb
        record(PASS, 'wandb installed (optional)', wandb.__version__)
        from env_file import load_env_file
        if 'WANDB_API_KEY' in load_env_file():
            record(PASS, 'W&B API key', 'loaded from .env')
        elif os.environ.get('WANDB_API_KEY'):
            record(PASS, 'W&B API key', 'set in the environment')
        else:
            record(WARN, 'W&B API key',
                   'not in .env -- paste it after WANDB_API_KEY= in .env, '
                   'or run: wandb login')
    except ImportError:
        record(WARN, 'wandb installed (optional)',
               'training works without it; losses still go to metrics.csv')

    # ---------------------------------------------------------- gpu
    section('gpu')
    gpu_ok, total_vram = False, 0.0
    if force_cpu:
        record(WARN, 'GPU checks skipped', '--device cpu was passed')
    elif not torch.cuda.is_available():
        record(FAIL, 'CUDA GPU visible',
               'torch.cuda.is_available() is False. Either the CPU-only wheel '
               'is installed or the NVIDIA driver is missing.')
    else:
        try:
            name = torch.cuda.get_device_name(0)
            total_vram = (torch.cuda.get_device_properties(0).total_memory
                          / 2 ** 30)
            # Creating a context can itself fail when the card is already full,
            # so this must never be allowed to abort the whole check.
            try:
                free_vram = torch.cuda.mem_get_info()[0] / 2 ** 30
            except Exception:
                free_vram = None
            detail = f'{name} -- {total_vram:.1f} GiB total'
            if free_vram is not None:
                detail += f', {free_vram:.1f} GiB free'
            record(PASS, 'CUDA GPU visible', detail)
            gpu_ok = True
            if free_vram is None:
                record(WARN, 'Could not read free VRAM',
                       'The card appears to be fully occupied by another '
                       'process. Close it before training.')
            elif free_vram < total_vram * 0.6:
                record(WARN, 'Another process is using this GPU',
                       f'only {free_vram:.1f} of {total_vram:.1f} GiB is free. '
                       'Close it first, or the measurement below will be '
                       'pessimistic.')
            if total_vram < 7.0:
                record(WARN, 'VRAM is small',
                       f'{total_vram:.1f} GiB. Expect to lower batch_size; '
                       'see README section 4.6.')
        except Exception as error:
            record(FAIL, 'CUDA GPU usable',
                   f'{type(error).__name__}: {error}'.splitlines()[0])

    # ---------------------------------------------------------- files
    section('package files')
    here = osp.dirname(osp.abspath(__file__))
    for relative in ('config.yml', 'train_plbert.py', 'model.py', 'dataset.py',
                     'text_utils.py', 'env_file.py', 'data/train.pkl', 'data/val.pkl',
                     'data/token_maps.pkl', 'data/stats.json'):
        path = osp.join(here, relative.replace('/', os.sep))
        if osp.exists(path):
            size = osp.getsize(path)
            record(PASS, relative,
                   f'{size / 2 ** 20:.1f} MB' if size > 2 ** 20 else f'{size} bytes')
        else:
            record(FAIL, relative, 'missing -- the copy is incomplete')

    # ---------------------------------------------------------- alphabet
    section('phoneme alphabet')
    sys.path.insert(0, here)
    try:
        import text_utils
        count = len(text_utils.symbols)
        record(PASS if count == 178 else FAIL, 'symbol table is 178 slots',
               f'{count} slots, {len(text_utils.dicts)} distinct characters')
        record(PASS if 'M' in text_utils.dicts else FAIL,
               'mask token "M" is in the table')
    except Exception as error:
        record(FAIL, 'text_utils imports', repr(error))

    # ---------------------------------------------------------- config/data
    section('config and data agree')
    import json
    import pickle
    import yaml
    try:
        with open(osp.join(here, 'config.yml'), encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        with open(osp.join(here, 'data', 'stats.json'), encoding='utf-8') as handle:
            stats = json.load(handle)
        separator = config['dataset_params']['word_separator']
        if separator == stats['word_separator_id']:
            record(PASS, 'word_separator matches stats.json', str(separator))
        else:
            record(FAIL, 'word_separator matches stats.json',
                   f'config says {separator}, data says '
                   f'{stats["word_separator_id"]}')
        for key, want in (('vocab_size', 178), ('hidden_size', 768),
                          ('num_hidden_layers', 12),
                          ('max_position_embeddings', 512)):
            got = config['model_params'].get(key)
            record(PASS if got == want else FAIL, f'model_params.{key}',
                   f'{got} (must be {want})')

        with open(osp.join(here, 'data', 'token_maps.pkl'), 'rb') as handle:
            token_maps = pickle.load(handle)
        num_vocab = 1 + max(e['token'] for e in token_maps.values())
        record(PASS if num_vocab == stats['num_vocab_for_model'] else FAIL,
               'token_maps size matches stats.json',
               f'{num_vocab:,} entries')
        with open(osp.join(here, 'data', 'train.pkl'), 'rb') as handle:
            train = pickle.load(handle)
        record(PASS if len(train) == stats['records_train'] else FAIL,
               'train.pkl record count', f'{len(train):,} records')
        sample = train[0]
        aligned = len(sample['phonemes']) == len(sample['input_ids'])
        record(PASS if aligned else FAIL, 'records are word-aligned',
               f'first record: {len(sample["phonemes"])} words')
    except Exception:
        record(FAIL, 'config/data cross-check', traceback.format_exc(limit=1))

    # ---------------------------------------------------------- real steps
    section('two real training steps')
    device = 'cpu' if force_cpu else ('cuda' if gpu_ok else None)
    if device is None:
        record(WARN, 'training step check skipped', 'no GPU available')
    else:
        try:
            from transformers import AlbertConfig, AlbertModel
            from dataset import build_dataloader
            from model import MultiTaskModel

            dataset_config = {
                k: config['dataset_params'][k]
                for k in ('token_separator', 'token_mask', 'max_mel_length',
                          'word_mask_prob', 'phoneme_mask_prob', 'replace_prob')
                if k in config['dataset_params']
            }
            precision = str(config.get('precision', 'fp16')).lower()
            if device == 'cpu':
                precision = 'fp32'
            amp_dtype = {'fp16': torch.float16,
                         'bf16': torch.bfloat16}.get(precision)
            criterion = torch.nn.CrossEntropyLoss()

            def attempt(batch_size):
                """Run two real steps; return peak GiB, or raise."""
                loader = build_dataloader(
                    train[:batch_size * 4], token_maps, separator,
                    batch_size=batch_size, num_workers=0, shuffle=False,
                    # pin_memory allocates page-locked host memory through CUDA
                    # and fails outright when no context can be created.
                    pin_memory=(device == 'cuda'),
                    dataset_config=dataset_config,
                )
                model = MultiTaskModel(
                    AlbertModel(AlbertConfig(**config['model_params'])),
                    num_tokens=config['model_params']['vocab_size'],
                    num_vocab=num_vocab,
                    hidden_size=config['model_params']['hidden_size'],
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = (torch.amp.GradScaler('cuda')
                          if precision == 'fp16' else None)
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                steps, loss = 0, None
                for words, labels, phonemes, lengths, masked in loader:
                    words, labels = words.to(device), labels.to(device)
                    phonemes = phonemes.to(device)
                    mask = torch.arange(max(lengths)).unsqueeze(0)
                    mask = mask.expand(len(lengths), -1)
                    mask = torch.gt(mask + 1,
                                    torch.LongTensor(lengths).unsqueeze(1))
                    with torch.autocast(device, dtype=amp_dtype,
                                        enabled=amp_dtype is not None):
                        _tokens_pred, words_pred = model(
                            phonemes, attention_mask=(~mask).int().to(device)
                        )
                    loss = 0.0
                    # Matches train_plbert.py: upcast each slice, never the
                    # whole batch x 512 x 30,002 tensor.
                    for pred, target, length in zip(words_pred, words, lengths):
                        loss = loss + criterion(pred[:length].float(),
                                                target[:length])
                    loss = loss / words.size(0)
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    steps += 1
                    if steps >= 2:
                        break
                peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                        if device == 'cuda' else 0.0)
                value = float(loss)
                del model, optimizer, loader, scaler, words_pred
                if device == 'cuda':
                    torch.cuda.empty_cache()
                return peak, value

            def is_oom(error):
                return (isinstance(error,
                                   getattr(torch.cuda, 'OutOfMemoryError', ()))
                        or 'out of memory' in str(error).lower())

            configured = int(config['batch_size'])
            # Descending ladder starting at whatever the config asks for.
            ladder = [configured] + [b for b in (16, 12, 8, 6, 4, 2)
                                     if b < configured]
            peak = loss_value = None
            working = None
            for candidate in ladder:
                try:
                    peak, loss_value = attempt(candidate)
                    working = candidate
                    break
                except Exception as error:
                    if not is_oom(error):
                        raise
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    record(WARN, f'batch_size={candidate} does not fit',
                           'retrying with a smaller batch')

            if working is None:
                record(FAIL, 'two real training steps',
                       'even batch_size=2 ran out of memory. Close every other '
                       'program using the GPU and try again; if it still '
                       'fails, lower max_mel_length to 256 in config.yml.')
            else:
                record(PASS, f'ran 2 steps at batch_size={working}',
                       f'loss {loss_value:.4f} on {device}')
                if device != 'cuda':
                    record(WARN, 'peak VRAM not measured',
                           'ran on CPU; re-run without --device cpu on the '
                           'real GPU to get this number')
                elif working == configured:
                    headroom = total_vram - peak
                    record(PASS if headroom > 1.0 else WARN, 'peak VRAM',
                           f'{peak:.2f} GiB of {total_vram:.1f} GiB '
                           f'({headroom:.2f} GiB headroom). '
                           + ('Comfortable.' if headroom > 1.5 else
                              'Tight -- drop one step on the ladder if '
                              'training OOMs later.'))
                else:
                    record(FAIL, 'config.yml needs one change',
                           f'batch_size: {configured} does not fit this card. '
                           f'batch_size: {working} does, using {peak:.2f} GiB '
                           f'of {total_vram:.1f} GiB. Edit config.yml:\n'
                           f'               batch_size: {working}')
        except Exception:
            record(FAIL, 'two real training steps',
                   traceback.format_exc(limit=3))

    summarise()


def summarise():
    print('\n' + '=' * 68)
    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
    if failures:
        print(f' {len(failures)} CHECK(S) FAILED -- do not start training yet')
        for _s, name, detail in failures:
            print(f'   - {name}: {detail.splitlines()[0] if detail else ""}')
        print('=' * 68)
        sys.exit(1)
    print(f' All checks passed'
          + (f' ({len(warnings)} warning(s))' if warnings else '')
          + '. Start with:  python train_plbert.py --config config.yml')
    print('=' * 68)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # A setup checker that crashes tells you nothing. Report and exit 1.
        print('\nThe check itself crashed:\n')
        traceback.print_exc()
        print('\nThat is usually a broken or partial install. Recreate the '
              'venv and run pip install -r requirements.txt again.')
        sys.exit(1)
