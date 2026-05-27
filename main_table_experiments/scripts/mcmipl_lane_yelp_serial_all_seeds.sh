#!/usr/bin/env bash
# Yelp-only lane: run seeds sequentially with per-seed archive (for s1/s2 after s0 handoff).
# s0 is normally handled by mcmipl_phase_b_cpu + watcher; this script is for explicit Yelp-only reruns.
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MCMIPL_FORCE_CPU="${MCMIPL_FORCE_CPU:-1}"
export MCMIPL_CPU_PYTHON="${MCMIPL_CPU_PYTHON:-/home/yrh666/venvs/mcmipl-cpu/bin/python}"
export MCMIPL_SAVE_NUM="${MCMIPL_SAVE_NUM:-10}"

SEEDS=("${@:-1 2}")
LANE_ID="yelp_$(date +%Y%m%d_%H%M%S)"
LANE_LOG="${MAIN}/logs/lane_YELP_STAR_${LANE_ID}.log"
exec > >(tee -a "$LANE_LOG") 2>&1

echo "=== LANE YELP_STAR seeds=${SEEDS[*]} | lane_id=${LANE_ID} | $(date) ==="

for SEED in "${SEEDS[@]}"; do
  if bash "${MAIN}/scripts/mcmipl_seed_archive_status.sh" YELP_STAR "$SEED" >/dev/null 2>&1; then
    echo "[skip] YELP_STAR seed=${SEED} already archived"
    continue
  fi
  export MCMIPL_RUN_ID="yelp_star_s${SEED}_${LANE_ID}"
  bash "${MAIN}/scripts/mcmipl_run_seed_with_archive.sh" YELP_STAR "$SEED"
done

echo "=== LANE YELP_STAR COMPLETE | $(date) ==="
