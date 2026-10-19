#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

# Full fine-tuning for the A1Z single-arm/two-camera task.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "${A1Z_DATASET:-}" ]] || {
  echo "A1Z_DATASET is empty. Set a real single-arm LeRobot v2.1 dataset." >&2
  exit 2
}
export G05_PYTHON="${G05_PYTHON:-$ROOT/GalaxeaVLA/.venv/bin/python}"
if [[ " $* " != *" --dry-run "* ]]; then
  "$G05_PYTHON" "$ROOT/scripts/full_preflight.py" >/dev/null
fi
export G05_TRAIN_TASK=a1z_single_left_full
exec bash "$ROOT/scripts/train.sh" "$@"
