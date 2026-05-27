#!/usr/bin/env bash
# Launch parallel Phase-B recovery lanes (one screen per dataset).
# Same-dataset seeds stay sequential inside each lane; different datasets run in parallel.
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

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

echo "=== mcmipl_phase_b_parallel_recovery | MAX_PARALLEL=${MAX_PARALLEL} | $(date) ==="
echo "Lanes (checkpoints -> tmp/<slug>/RL-agent/seed_<N>/):"
echo "  YELP_STAR: 1 2  (s0 via watcher handoff)"
echo "  BOOK / LAST_FM / MOVIE: use mcmipl_launch_other_datasets_now.sh"

launch_lane YELP_STAR 1 2
sleep 5
launch_lane BOOK 0 1
sleep 5
launch_lane LAST_FM_STAR 0 1
sleep 5
launch_lane MOVIE 0 1

echo ""
echo "=== Active screens ==="
screen -ls || true
echo ""
echo "Monitor: tail -f ${MAIN}/logs/lane_*.log"
echo "Checkpoints archive: ${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives/checkpoints/"
