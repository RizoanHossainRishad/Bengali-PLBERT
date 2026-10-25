# coding: utf-8
"""Shrink a training checkpoint to the part StyleTTS2 actually uses.

A training checkpoint is ~350 MB: model + AdamW moments + GradScaler state.
StyleTTS2 keeps only the 25 encoder tensors (~6.3 M parameters, ~25 MB) and
throws the rest away at load time. Run this before uploading so you move 25 MB
instead of 350 MB, and keep the big file locally in case you want to resume.

    python export_for_styletts2.py --checkpoint checkpoints/step_200000.t7

Writes export/step_200000.t7 and export/config.yml -- copy that whole folder to
the server as Utils/PLBERT_bn/.
"""

import argparse
import os
import os.path as osp
import shutil

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--out-dir', default='export')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    source = checkpoint['net']

    kept = {
        key: value.clone() for key, value in source.items()
        if key.startswith('module.encoder.')
    }
    if 'module.encoder.embeddings.position_ids' not in kept:
        raise SystemExit(
            'position_ids missing -- this checkpoint predates the fix and '
            "StyleTTS2's load_plbert() would raise KeyError."
        )
    # 25 encoder tensors + position_ids.
    if len(kept) != 26:
        raise SystemExit(f'expected 26 encoder tensors, found {len(kept)}')

    os.makedirs(args.out_dir, exist_ok=True)
    # Named by step, never by source filename: load_plbert() only considers
    # files called step_*.t7, so exporting best.t7 under its own name would
    # leave the server with nothing it can load.
    step = int(checkpoint.get('step', 0))
    name = f'step_{step}.t7'
    out_path = osp.join(args.out_dir, name)
    # load_plbert() silently takes the HIGHEST step in the folder, so a stale
    # export from an earlier, later-step checkpoint would win over this one.
    stale = [f for f in os.listdir(args.out_dir)
             if f.startswith('step_') and f != name]
    if stale:
        raise SystemExit(
            f'{args.out_dir} already holds {stale}. load_plbert() would load '
            f'the highest step, not this export. Remove them or pass a new '
            f'--out-dir.'
        )
    torch.save({'net': kept, 'step': step}, out_path)

    # load_plbert() reads config.yml from the same folder.
    import yaml
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    with open(osp.join(args.out_dir, 'config.yml'), 'w',
              encoding='utf-8') as handle:
        handle.write(
            '# Bengali PL-BERT for StyleTTS2. Place beside step_*.t7.\n'
        )
        yaml.safe_dump({'model_params': config['model_params']}, handle,
                       default_flow_style=False, sort_keys=False)

    if osp.exists('server_side/util.py'):
        shutil.copy('server_side/util.py',
                    osp.join(args.out_dir, 'util.py'))

    before = osp.getsize(args.checkpoint) / 2 ** 20
    after = osp.getsize(out_path) / 2 ** 20
    print(f'{args.checkpoint}  {before:7.1f} MB')
    print(f'{out_path}  {after:7.1f} MB   ({len(kept)} tensors)')
    print(f'\nUpload the whole "{args.out_dir}" folder to the server as '
          f'StyleTTS2-Fork/Utils/PLBERT_bn/')


if __name__ == '__main__':
    main()
