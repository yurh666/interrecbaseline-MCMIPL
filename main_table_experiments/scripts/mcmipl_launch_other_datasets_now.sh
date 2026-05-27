#!/usr/bin/env bash
# Start BOOK / LAST_FM / MOVIE recovery lanes now (YELP s0 keeps running separately).
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

launch_lane() {
  local dataset="$1"
  shift
  local seeds=("$@")
  local sname="mcmipl_lane_${dataset}"
  if screen -ls | grep -q "[0-9]*\.${sname}[[:space:]]"; then
    echo "[skip] screen ${sname} already exists"
    return 0
  fi
  screen -dmS "$sname" bash -lc \
    "bash ${MAIN}/scripts/mcmipl_lane_sequential.sh ${dataset} ${seeds[*]} 2>&1 | tee -a ${MAIN}/logs/screen_${sname}_$(date +%Y%m%d_%H%M%S).log"
  echo "[launch] screen ${sname} seeds=${seeds[*]}"
}

echo "=== launch OTHER datasets (not YELP) | $(date) ==="
echo "Checkpoints save to tmp/<slug>/RL-agent/seed_<N>/ (MCMIPL_RL_SEED)"

launch_lane BOOK 0 1
sleep 3
launch_lane LAST_FM_STAR 0 1
sleep 3
launch_lane MOVIE 0 1

echo ""
screen -ls || true
