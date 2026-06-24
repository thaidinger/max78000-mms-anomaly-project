#!/usr/bin/env python3
"""
Export a deployable MMS checkpoint from an auxiliary-head training checkpoint.

The auxiliary image-classification head is useful during training, but the MAX78000
deployment graph remains the map-only ai85mmsanomalynet. This script removes aux_fc.*
weights and rewrites the checkpoint architecture name accordingly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def is_aux_key(key: str) -> bool:
    """Return true for auxiliary-head parameters, with or without DDP's module. prefix."""
    normalized = key.removeprefix('module.')
    return normalized.startswith('aux_fc.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Aux-head checkpoint to export')
    parser.add_argument('--output', required=True, help='Deploy checkpoint path to write')
    parser.add_argument('--arch', default='ai85mmsanomalynet',
                        help='Deploy architecture name to write into the checkpoint')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' not in checkpoint:
        raise KeyError(f'{checkpoint_path} does not contain a state_dict')

    state_dict = checkpoint['state_dict']
    deploy_state = {key: value for key, value in state_dict.items() if not is_aux_key(key)}
    stripped = sorted(key for key in state_dict if is_aux_key(key))

    output = {
        'epoch': checkpoint.get('epoch', 0),
        'arch': args.arch,
        'state_dict': deploy_state,
        'extras': checkpoint.get('extras', {}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f'Read: {checkpoint_path}')
    print(f'Wrote: {output_path}')
    print(f'Architecture: {checkpoint.get("arch", "unknown")} -> {args.arch}')
    print(f'State entries: {len(state_dict)} -> {len(deploy_state)}')
    print(f'Stripped aux entries: {len(stripped)}')
    for key in stripped:
        print(f'  - {key}')


if __name__ == '__main__':
    main()
