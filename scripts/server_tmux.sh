#!/usr/bin/env bash
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail

A1Z_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${G05_TMUX_SESSION:-g05-server}"
COMMAND="${1:-status}"
shift || true

case "$COMMAND" in
  start)
    command -v tmux >/dev/null || { echo "tmux is required"; exit 1; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "G0.5 server is already running in tmux session: $SESSION"
      exit 0
    fi
    printf -v root_q '%q' "$A1Z_ROOT"
    args_q=""
    for arg in "$@"; do
      printf -v arg_q '%q' "$arg"
      args_q+=" $arg_q"
    done
    # A running tmux server keeps its old environment. Forward this invocation's
    # explicitly chosen deployment values instead of silently reusing old ones.
    env_q=""
    for name in G05_PYTHON G05_EMBODIMENT POLICY_HOST POLICY_PORT POLICY_DEVICE ACTION_STEPS G05_OUTPUT_DIR; do
      if [[ -n "${!name:-}" ]]; then
        printf -v value_q '%q' "${!name}"
        env_q+=" $name=$value_q"
      fi
    done
    tmux new-session -d -s "$SESSION" \
      "cd $root_q && exec env$env_q bash scripts/server.sh$args_q"
    echo "Started G0.5 server in tmux session: $SESSION"
    echo "Logs: bash scripts/server_tmux.sh logs"
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "running: $SESSION"
      tmux list-panes -t "$SESSION" -F 'pid=#{pane_pid} command=#{pane_current_command}'
    else
      echo "stopped: $SESSION"
      exit 1
    fi
    ;;
  logs)
    exec tmux attach-session -t "$SESSION"
    ;;
  tail)
    tmux capture-pane -pt "$SESSION" -S -120
    ;;
  stop)
    tmux kill-session -t "$SESSION"
    echo "Stopped G0.5 server: $SESSION"
    ;;
  *)
    echo "Usage: bash scripts/server_tmux.sh {start|status|logs|tail|stop} [checkpoint] [overrides...]"
    exit 2
    ;;
esac
