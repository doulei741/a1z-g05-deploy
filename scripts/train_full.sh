#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${A1Z_DATASET:-}" ]]; then
  echo 'A1Z_DATASET is empty. Set the real LeRobot dataset directory before full fine-tuning.' >&2
  exit 2
fi
PYTHON_BIN="${G05_PYTHON:-python3}"
VERSION=2.1
if [[ " $* " != *" --dry-run "* ]]; then
  VERSION="$("$PYTHON_BIN" "$ROOT/scripts/full_preflight.py")"
fi
export A1Z_DATASET="$("$PYTHON_BIN" -c 'import os,pathlib; print(pathlib.Path(os.environ["A1Z_DATASET"]).expanduser().resolve())')"
export G05_TRAIN_TASK=a1z
exec bash "$ROOT/scripts/train.sh" "$@" \
  ++model.model_arch.lora.enabled=false \
  model.model_arch.vla_training_strategy=vla-full-train \
  "data.embodiment_datasets.galaxea_a1z.lerobot_ds_version='$VERSION'"
