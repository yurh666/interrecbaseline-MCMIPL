#!/usr/bin/env bash
# One dataset lane: run seeds sequentially with auto-archive after each.
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?DATASET}"
shift
SEEDS=("$@")
[[ ${#SEEDS[@]} -gt 0 ]] || { echo "usage: $0 DATASET seed [seed...]" >&2; exit 1; }

LANE_ID="${LANE_ID:-${DATASET}_$(date +%Y%m%d_%H%M%S)}"
LANE_LOG="${MAIN}/logs/lane_${DATASET}_${LANE_ID}.log"
exec > >(tee -a "$LANE_LOG") 2>&1
SLUG_FOR_ID="$(bash -c "source ${MAIN}/scripts/mcmipl_data_slug.sh && data_slug ${DATASET}")"

echo "=== LANE ${DATASET} seeds=${SEEDS[*]} lane_id=${LANE_ID} | $(date) ==="

for s in "${SEEDS[@]}"; do
  if bash "${MAIN}/scripts/mcmipl_seed_archive_status.sh" "$DATASET" "$s" >/dev/null 2>&1; then
    echo "[skip] ${DATASET} seed=${s} already archived"
    continue
  fi
  export MCMIPL_RUN_ID="${SLUG_FOR_ID}_s${s}_${LANE_ID}"
  bash "${MAIN}/scripts/mcmipl_run_seed_with_archive.sh" "$DATASET" "$s"
done

echo "=== LANE ${DATASET} COMPLETE | $(date) ==="
