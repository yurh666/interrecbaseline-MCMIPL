#!/usr/bin/env bash
# Serial RL supplement: BOOK 0,1 → LAST_FM 0,1 → MOVIE 0,1 (skip if archived).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="${REPO}/main_table_experiments"
SCRIPTS="${MAIN}/scripts"
SCREEN_NAME="${MCMIPL_SUPPLEMENT_SCREEN:-mcmipl_rl_supplement_3ds}"

export MCMIPL_FORCE_CPU="${MCMIPL_FORCE_CPU:-1}"
export MCMIPL_CPU_PYTHON="${MCMIPL_CPU_PYTHON:-/home/yrh666/venvs/mcmipl-cpu/bin/python}"
export MCMIPL_SAVE_NUM="${MCMIPL_SAVE_NUM:-10}"

_inner() {
  bash "${MAIN}/rl_supplement_three_datasets/restore_phase_a_to_tmp.sh"
  bash "${SCRIPTS}/mcmipl_lane_three_datasets_serial.sh"
}

if [[ -n "${TMUX:-}" ]] || [[ "${RUN_FOREGROUND:-}" == "1" ]]; then
  _inner
else
  if tmux has-session -t "$SCREEN_NAME" 2>/dev/null; then
    echo "tmux session $SCREEN_NAME exists — attach: tmux attach -t $SCREEN_NAME"
    exit 0
  fi
  LOG="${MAIN}/logs/screen_${SCREEN_NAME}_$(date +%Y%m%d_%H%M%S).log"
  tmux new-session -d -s "$SCREEN_NAME" "cd '${REPO}' && RUN_FOREGROUND=1 bash '${MAIN}/rl_supplement_three_datasets/run_supplement_s01_serial.sh' 2>&1 | tee -a '${LOG}'"
  echo "started tmux $SCREEN_NAME — attach: tmux attach -t $SCREEN_NAME"
  echo "log: $LOG"
fi
