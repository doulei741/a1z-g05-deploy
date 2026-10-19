#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
A1Z_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${G05_PYTHON:-python}"
if [[ -z "${G05_PYTHON:-}" && -x "$A1Z_ROOT/GalaxeaVLA/.venv/bin/python" ]]; then
  PYTHON_BIN="$A1Z_ROOT/GalaxeaVLA/.venv/bin/python"
fi
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export A1Z_WORKSPACE="$A1Z_ROOT"
export A1Z_DATASET="${A1Z_DATASET:-}"
"$PYTHON_BIN" "$A1Z_ROOT/scripts/require_v21.py"
export A1Z_DATASET="$("$PYTHON_BIN" -c 'import os,pathlib; print(pathlib.Path(os.environ["A1Z_DATASET"]).expanduser().resolve())')"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-$A1Z_ROOT/outputs}"
export PYTHONPATH="$A1Z_ROOT/GalaxeaVLA/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$A1Z_ROOT/GalaxeaVLA"
if [[ " $* " != *" --dry-run "* ]]; then
  test -f "$A1Z_DATASET/meta/info.json" || { echo "Missing converted dataset: $A1Z_DATASET"; exit 1; }
  test ! -e "$A1Z_DATASET/INCOMPLETE" || { echo "Dataset conversion incomplete"; exit 1; }
  test -f "$A1Z_ROOT/checkpoints/g05-base/checkpoints/model_state_dict.pt" || { echo "Download G05 base checkpoint first (see README)"; exit 1; }
fi
# The upstream --dry-run path still loads the model. Use Hydra's config-only
# mode so this preflight works before downloading weights or reserving a GPU.
DRY_RUN=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then
    DRY_RUN=1
  else
    ARGS+=("$arg")
  fi
done
if [[ "$DRY_RUN" == 1 ]]; then
  exec "$PYTHON_BIN" scripts/finetune.py --cfg job --resolve \
    "task=${G05_TRAIN_TASK:-a1z}" model.max_epochs=null "${ARGS[@]}" \
    "data.embodiment_datasets.galaxea_a1z.lerobot_ds_version='2.1'"
fi
exec bash scripts/run/finetune.sh "${NUM_GPUS:-1}" "${G05_TRAIN_TASK:-a1z}" model.max_epochs=null "${ARGS[@]}" "data.embodiment_datasets.galaxea_a1z.lerobot_ds_version='2.1'"
