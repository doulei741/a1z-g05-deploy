# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Validate the supported G0.5 dataset contract without loading training code."""
import json
import os
from pathlib import Path
import sys


def validate(dataset):
    if not dataset:
        raise ValueError('A1Z_DATASET is empty. Set a real LeRobot v2.1 dataset.')
    root = Path(dataset).expanduser().resolve()
    if (root / 'INCOMPLETE').exists():
        raise ValueError('Dataset conversion is INCOMPLETE')
    info = json.loads((root / 'meta/info.json').read_text())
    if info.get('codebase_version') != 'v2.1':
        raise ValueError('G0.5 training requires LeRobot v2.1; v3.0 is not supported by this workflow. Do not relabel info.json.')
    for name in ('episodes.jsonl', 'tasks.jsonl', 'episodes_stats.jsonl'):
        if not (root / 'meta' / name).is_file():
            raise ValueError(f'Missing v2.1 metadata: meta/{name}; do not relabel v3 data.')
    if not (root / 'data').is_dir():
        raise ValueError('Missing data directory')
    return root


if __name__ == '__main__':
    try:
        validate(os.environ.get('A1Z_DATASET', ''))
    except (ValueError, OSError, TypeError) as exc:
        sys.exit(str(exc))
