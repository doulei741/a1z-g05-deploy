# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Validate local inputs without loading CUDA or choosing a dataset."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from require_v21 import validate

ROOT = Path(__file__).resolve().parents[1]


def check(root, dataset):
    if not dataset:
        raise ValueError('A1Z_DATASET is empty; set your real dataset directory')
    data = validate(dataset)
    if (data/'INCOMPLETE').exists():
        raise ValueError('Dataset conversion is INCOMPLETE')
    info = json.loads((data/'meta/info.json').read_text())
    version = str(info.get('codebase_version', '')).removeprefix('v')
    if version != '2.1':
        raise ValueError(f'Unsupported codebase_version: {version!r}; expected v2.1')
    if int(info.get('total_episodes', 0)) < 1 or not (data/'data').is_dir():
        raise ValueError('Dataset has no episodes or data directory')
    for rel in ('g05-base/checkpoints/model_state_dict.pt', 'g05-base/.hydra/config.yaml',
                'g05-base/dataset_stats.json', 'action_tokenizer.pt', 'qwen3_5_2b_base_processor'):
        p = root/'checkpoints'/rel
        if not p.exists() or (p.is_file() and p.stat().st_size == 0):
            raise ValueError(f'Missing or empty model asset: {p}')
    return version


if __name__ == '__main__':
    try:
        print(check(ROOT, os.environ.get('A1Z_DATASET', '')))
    except (ValueError, OSError, TypeError) as exc:
        sys.exit(str(exc))
