#!/usr/bin/env bash
# Record start/end times for one dataset/seed run (append-only timeline + per-seed TIMING.txt).
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${MAIN}/scripts/mcmipl_data_slug.sh"

ACTION="${1:?start|end}"
DATASET="${2:?DATASET}"
SEED="${3:?SEED}"
RUN_ID="${4:-}"
NOTE="${5:-}"

ARCH_ROOT="${MCMIPL_ARCHIVE_ROOT:-${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives}"
SLUG="$(data_slug "$DATASET")"
TS="$(date -Iseconds)"
TIMELINE="${ARCH_ROOT}/phase_b_run_timeline.csv"
DEST="${ARCH_ROOT}/checkpoints/${SLUG}/seed_${SEED}"

mkdir -p "$DEST" "$ARCH_ROOT"

if [[ ! -f "$TIMELINE" ]]; then
  echo "ts,action,dataset,slug,seed,run_id,note" > "$TIMELINE"
fi
printf '%s,%s,%s,%s,%s,%s,%s\n' "$TS" "$ACTION" "$DATASET" "$SLUG" "$SEED" "$RUN_ID" "$NOTE" >> "$TIMELINE"

case "$ACTION" in
  start)
    echo "started_at=${TS}" > "${DEST}/TIMING.txt"
    echo "dataset=${DATASET}" >> "${DEST}/TIMING.txt"
    echo "seed=${SEED}" >> "${DEST}/TIMING.txt"
    echo "run_id=${RUN_ID}" >> "${DEST}/TIMING.txt"
    echo "status=running" >> "${DEST}/TIMING.txt"
    ;;
  end)
    if [[ -f "${DEST}/TIMING.txt" ]]; then
      echo "finished_at=${TS}" >> "${DEST}/TIMING.txt"
      echo "status=done" >> "${DEST}/TIMING.txt"
    else
      {
        echo "dataset=${DATASET}"
        echo "seed=${SEED}"
        echo "finished_at=${TS}"
        echo "run_id=${RUN_ID}"
        echo "status=done"
      } > "${DEST}/TIMING.txt"
    fi
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 1
    ;;
esac

echo "[timing] ${ACTION} ${DATASET} seed=${SEED} @ ${TS}"
