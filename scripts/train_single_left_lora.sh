#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

# LoRA fine-tuning for a single arm and head/wrist cameras.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export G05_TRAIN_TASK=a1z_single_left_lora
exec bash "$ROOT/scripts/train.sh" "$@"
