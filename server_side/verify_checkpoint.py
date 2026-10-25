# coding: utf-8
"""Check a trained Bengali PL-BERT before wiring it into StyleTTS2.

Run this on the SERVER, inside the tts environment, after uploading:

    python verify_checkpoint.py --plbert-dir Utils/PLBERT_bn

It fails loudly on the two problems that otherwise stay silent:
  * load_plbert() uses strict=False, so a key-name mismatch leaves tensors at
    their random initialisation without printing anything;
  * a checkpoint without embeddings.position_ids makes the stock loader raise
    KeyError.
"""

import argparse
import os.path as osp
import sys

import torch
import yaml
from transformers import AlbertConfig, AlbertModel

EXPECTED = {'vocab_size': 178, 'hidden_size': 768,
            'num_hidden_layers': 12, 'max_position_embeddings': 512}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plbert-dir', required=True)
    args = parser.parse_args()

    config = yaml.safe_load(
        open(osp.join(args.plbert_dir, 'config.yml'), encoding='utf-8')
    )
    params = config['model_params']
    print('model_params:')
    failures = []
    for key, want in EXPECTED.items():
        got = params.get(key)
        ok = got == want
        print(f'   {key:<28} {got!s:<8} expected {want}  {"OK" if ok else "MISMATCH"}')
        if not ok:
            failures.append(f'{key}={got} (expected {want})')

    steps = sorted(
        int(name[5:].split('.')[0])
        for name in __import__('os').listdir(args.plbert_dir)
        if name.startswith('step_') and name.endswith('.t7')
    )
    if not steps:
        failures.append('no step_*.t7 checkpoint found')
        print('\n'.join(failures))
        sys.exit(1)
    path = osp.join(args.plbert_dir, f'step_{steps[-1]}.t7')
    print(f'\ncheckpoint: {path}')

    model = AlbertModel(AlbertConfig(**params))
    expected_keys = set(model.state_dict())

    raw = torch.load(path, map_location='cpu', weights_only=False)['net']
    has_position_ids = any('position_ids' in k for k in raw)
    print(f'position_ids present in checkpoint : {has_position_ids} '
          f'{"(stock load_plbert is happy)" if has_position_ids else "(stock load_plbert will raise KeyError)"}')
    if not has_position_ids:
        failures.append('checkpoint lacks embeddings.position_ids')

    encoder = {}
    for key, value in raw.items():
        name = key[7:]                               # drop "module."
        if name.startswith('encoder.'):
            encoder[name[8:]] = value
    encoder.pop('embeddings.position_ids', None)

    matched = expected_keys & set(encoder)
    missing = expected_keys - set(encoder)
    unexpected = set(encoder) - expected_keys

    print(f'\nAlbertModel expects : {len(expected_keys)} tensors')
    print(f'checkpoint provides : {len(encoder)} tensors')
    print(f'MATCHED             : {len(matched)}')
    print(f'MISSING             : {len(missing)}')
    for key in sorted(missing):
        print(f'   - {key}')
    print(f'UNEXPECTED          : {len(unexpected)}')
    for key in sorted(unexpected):
        print(f'   + {key}')

    bad_shapes = [
        k for k in matched
        if tuple(encoder[k].shape) != tuple(model.state_dict()[k].shape)
    ]
    print(f'SHAPE MISMATCHES    : {len(bad_shapes)}')
    for key in bad_shapes:
        print(f'   ! {key}: {tuple(encoder[key].shape)} vs '
              f'{tuple(model.state_dict()[key].shape)}')

    if missing or bad_shapes:
        failures.append(f'{len(missing)} missing, {len(bad_shapes)} wrong shape')

    print()
    if failures:
        print('FAILED: ' + '; '.join(failures))
        sys.exit(1)
    print('PASS - all 25 encoder tensors load cleanly. Safe to point '
          'PLBERT_dir at this folder.')


if __name__ == '__main__':
    main()
