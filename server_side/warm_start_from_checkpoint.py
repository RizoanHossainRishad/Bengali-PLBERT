# coding: utf-8
"""Graft the new Bengali PL-BERT into an existing StyleTTS2 checkpoint.

Run on the SERVER, after verify_checkpoint.py passes.

The current 40-epoch run has learned a decoder, two style encoders, an aligner
and a style-diffusion stack. None of that knowledge is language-specific in a
way the new PL-BERT invalidates, so there is no reason to start from scratch:
replace the 25 `bert` tensors and keep the other ~1,500.

    python warm_start_from_checkpoint.py \
        --checkpoint Models/<run>/epoch_000NN.pth \
        --plbert-dir Utils/PLBERT_bn \
        --out        Models/<run>/warmstart_bn_plbert.pth

Then point the training config at --out with load_only_params: true. The
optimizer state is deliberately dropped: its AdamW moments belong to the old
English embeddings and would fight the new ones.
"""

import argparse
import os.path as osp
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--plbert-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--styletts2-root', default='.')
    args = parser.parse_args()

    sys.path.insert(0, args.styletts2_root)
    from Utils.PLBERT.util import load_plbert

    print(f'loading {args.checkpoint} ...')
    state = torch.load(args.checkpoint, map_location='cpu',
                       weights_only=False)
    net = state['net']
    if 'bert' not in net:
        raise SystemExit("checkpoint has no 'bert' module")

    old_bert = net['bert']
    print(f"existing net['bert'] : {len(old_bert)} tensors")

    new_bert_model = load_plbert(args.plbert_dir)
    new_bert = {
        f'module.{key}': value
        for key, value in new_bert_model.state_dict().items()
    }
    print(f'new PL-BERT         : {len(new_bert)} tensors')

    missing = set(old_bert) - set(new_bert) - {
        'module.embeddings.position_ids'
    }
    extra = set(new_bert) - set(old_bert)
    if missing or extra:
        raise SystemExit(
            f'key mismatch -- missing {sorted(missing)}, extra {sorted(extra)}'
        )
    for key in new_bert:
        if tuple(new_bert[key].shape) != tuple(old_bert[key].shape):
            raise SystemExit(
                f'shape mismatch on {key}: {tuple(new_bert[key].shape)} '
                f'vs {tuple(old_bert[key].shape)}'
            )

    net['bert'] = new_bert
    # The optimizer moments were accumulated against the English embeddings.
    state.pop('optimizer', None)
    state.pop('scaler', None)
    state['plbert_source'] = osp.abspath(args.plbert_dir)

    untouched = sum(len(net[k]) for k in net if k != 'bert')
    print(f'replaced  : bert ({len(new_bert)} tensors)')
    print(f'preserved : {untouched} tensors across '
          f'{len(net) - 1} other modules')

    torch.save(state, args.out + '.tmp')
    import os
    os.replace(args.out + '.tmp', args.out)
    print(f'\nwrote {args.out}')
    print('Set in the training config:')
    print(f'    pretrained_model: {osp.abspath(args.out)}')
    print('    second_stage_load_pretrained: true')
    print('    load_only_params: true        # optimizer state was dropped')

if __name__ == '__main__':
    main()
