#!/usr/bin/env bash
# One-shot: archive Phase-A, seed-2 checkpoints (on disk), authoritative logs, CRS from logs.
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH_ROOT="${MCMIPL_ARCHIVE_ROOT:-${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives}"
LOG_DIR="${MCMIPL_LOG_DIR:-${MAIN}/logs}"
SCRIPTS="${MAIN}/scripts"

mkdir -p "${ARCH_ROOT}/logs_authoritative"

echo "=== mcmipl_archive_all_existing | $(date) ==="

for d in LAST_FM_STAR BOOK MOVIE YELP_STAR; do
  bash "${SCRIPTS}/mcmipl_archive_phase_a.sh" "$d"
done

# Disk currently holds last-completed seed per dataset (mostly seed 2; yelp partial s0)
bash "${SCRIPTS}/mcmipl_archive_seed.sh" BOOK 2
bash "${SCRIPTS}/mcmipl_archive_seed.sh" LAST_FM_STAR 2
bash "${SCRIPTS}/mcmipl_archive_seed.sh" MOVIE 2
bash "${SCRIPTS}/mcmipl_archive_seed.sh" YELP_STAR 0 || true

for log in "${LOG_DIR}"/train_{LAST_FM_STAR,BOOK,MOVIE,YELP_STAR}_s{0,1,2}.log; do
  [[ -f "$log" ]] && cp -a "$log" "${ARCH_ROOT}/logs_authoritative/$(basename "$log")"
done

python3 "${SCRIPTS}/mcmipl_extract_crs_from_logs.py" \
  --log-dir "${ARCH_ROOT}/logs_authoritative" \
  --out "${ARCH_ROOT}/crs_metrics_from_logs.json"

echo "=== done -> ${ARCH_ROOT} ==="
