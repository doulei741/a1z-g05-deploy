#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-status}"
G05_ARM_MODE="${G05_ARM_MODE:-dual}"
case "$G05_ARM_MODE" in
  dual) TRAIN_ENTRY=train_full.sh ;;
  single|single-left) TRAIN_ENTRY=train_single_full.sh ;;
  *) echo "G05_ARM_MODE must be single or dual" >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || shift
EXP_NAME="${EXP_NAME:-full_$(date +%Y%m%d_%H%M%S)}"
[[ "$EXP_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'EXP_NAME: use letters, digits, _ or -'; exit 2; }
SESSION="g05-full-$EXP_NAME"
RUN_DIR="$ROOT/logs/$EXP_NAME"
case "$ACTION" in
  start)
    "${G05_PYTHON:-python3}" "$ROOT/scripts/full_preflight.py" >/dev/null
    command -v tmux >/dev/null
    [[ ! -e "$RUN_DIR" ]] || { echo "Run directory exists: $RUN_DIR; use a new EXP_NAME"; exit 2; }
    mkdir -p "$RUN_DIR"
    # Persist a quoted command, not machine configuration in tracked source.
    {
      echo '#!/usr/bin/env bash'
      echo 'set -uo pipefail'
      printf 'cd %q\n' "$ROOT"
      for key in A1Z_DATASET G05_PYTHON G05_OUTPUT_DIR CUDA_VISIBLE_DEVICES NUM_GPUS; do
        if [[ -n "${!key:-}" ]]; then printf 'export %s=%q\n' "$key" "${!key}"; fi
      done
      printf 'export EXP_NAME=%q WANDB_MODE=offline PYTHONUNBUFFERED=1\n' "$EXP_NAME"
      printf 'bash %q ' "$ROOT/scripts/$TRAIN_ENTRY"
      printf '%q ' "logger.type=wandb" "logger.mode=offline" "logger.workspace=null" "$@" "exp_name=$EXP_NAME"
      printf '2>&1 | tee %q\n' "$RUN_DIR/train.log"
      echo 'code=${PIPESTATUS[0]}'
      printf 'printf "%%s\\n" "$code" > %q\n' "$RUN_DIR/exit_code"
      echo 'exit "$code"'
    } > "$RUN_DIR/run.sh"
    printf -v launch 'bash %q' "$RUN_DIR/run.sh"
    tmux new-session -d -s "$SESSION" "$launch"
    echo "Session launched: $SESSION (training readiness is in the log)"
    echo "Log: $RUN_DIR/train.log"
    ;;
  status)
    if [[ -f "$RUN_DIR/exit_code" ]]; then
      echo "Exited with code $(cat "$RUN_DIR/exit_code")"
    elif tmux has-session -t "=$SESSION" 2>/dev/null; then
      echo "Session active: $SESSION; inspect train.log for actual progress"
    else
      echo 'No session or exit record; EXP_NAME may be wrong or the process was interrupted'
    fi
    ;;
  tail) tail -n 60 -f "$RUN_DIR/train.log" ;;
  *) echo 'Usage: EXP_NAME=name bash scripts/train_full_tmux.sh start|status|tail [training overrides]'; exit 2 ;;
esac
