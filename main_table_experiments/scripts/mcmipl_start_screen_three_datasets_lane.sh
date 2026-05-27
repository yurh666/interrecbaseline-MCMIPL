#!/usr/bin/env bash
# Start (or skip if running) screen for BOOK/LAST_FM/MOVIE serial lane (seeds 0,1 only).
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCREEN_NAME="${MCMIPL_3DS_SCREEN:-mcmipl_lane_3ds}"
LOG="${MAIN}/logs/screen_${SCREEN_NAME}_$(date +%Y%m%d_%H%M%S).log"

if screen -ls | grep -qE "[0-9]+\.${SCREEN_NAME}[[:space:]]"; then
  echo "[start] screen ${SCREEN_NAME} already running — attach: screen -r ${SCREEN_NAME}"
  exit 0
fi

screen -dmS "${SCREEN_NAME}" bash -lc \
  "bash ${MAIN}/scripts/mcmipl_lane_three_datasets_serial.sh 2>&1 | tee -a ${LOG}"

echo "[start] launched screen ${SCREEN_NAME}"
echo "        log: ${LOG}"
echo "        attach: screen -r ${SCREEN_NAME}"
