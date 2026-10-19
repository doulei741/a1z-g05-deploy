#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail

A1Z_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CKPT="$A1Z_ROOT/checkpoints/g05-base/checkpoints/model_state_dict.pt"
if [[ -n "${G05_PYTHON:-}" ]]; then
  PYTHON_BIN="$G05_PYTHON"
elif [[ -x "$A1Z_ROOT/GalaxeaVLA/.venv/bin/python" ]]; then
  PYTHON_BIN="$A1Z_ROOT/GalaxeaVLA/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

BASE_CKPT="$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$BASE_CKPT")"

# No positional checkpoint means the public G0.5 base checkpoint. A non-base
# checkpoint remains supported as the first argument, followed by Hydra overrides.
A1Z_CKPT="$BASE_CKPT"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  A1Z_CKPT="$1"
  shift
fi
A1Z_CKPT="$($PYTHON_BIN -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$A1Z_CKPT")"
test -f "$A1Z_CKPT" || { echo "Checkpoint missing: $A1Z_CKPT"; exit 1; }

HF_PROCESSOR="$A1Z_ROOT/checkpoints/qwen3_5_2b_base_processor"
ACTION_TOKENIZER="$A1Z_ROOT/checkpoints/action_tokenizer.pt"
test -d "$HF_PROCESSOR" || { echo "HF processor missing: $HF_PROCESSOR"; exit 1; }
test -f "$ACTION_TOKENIZER" || { echo "Action tokenizer missing: $ACTION_TOKENIZER"; exit 1; }

EMBODIMENT="${G05_EMBODIMENT:-galaxea_a1z}"

export A1Z_WORKSPACE="$A1Z_ROOT"
# The saved training config resolves this path even when serving from sidecar stats.
export A1Z_DATASET="${A1Z_DATASET:-$A1Z_ROOT/data}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-$A1Z_ROOT/outputs}"
export PYTHONPATH="$A1Z_ROOT/GalaxeaVLA/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$A1Z_ROOT/GalaxeaVLA"
exec "$PYTHON_BIN" scripts/serve_policy.py \
  --ckpt_path "$A1Z_CKPT" \
  --host "${POLICY_HOST:-127.0.0.1}" \
  --port "${POLICY_PORT:-8765}" \
  --device "${POLICY_DEVICE:-cuda}" \
  --action_steps "${ACTION_STEPS:-1}" \
  "eval_embodiment=$EMBODIMENT" \
  "model.model_arch.hf_processor_path=$HF_PROCESSOR" \
  "model.processor.tokenizer_params.pretrained_model_name_or_path=$HF_PROCESSOR" \
  "tokenizer.vq_config.ckpt_dir=$ACTION_TOKENIZER" \
  "model.tokenizer.vq_config.ckpt_dir=$ACTION_TOKENIZER" \
  "model.model_arch.AT_CONFIG.ckpt_dir=$ACTION_TOKENIZER" \
  "$@"
