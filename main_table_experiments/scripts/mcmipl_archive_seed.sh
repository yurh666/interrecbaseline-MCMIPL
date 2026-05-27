#!/usr/bin/env bash
# Archive RL checkpoint + RL-log-merge + train log for ONE dataset/seed (copy-only).
# Layout: archives/checkpoints/<slug>/seed_<N>/  (canonical per-seed store)
#         archives/runs/<slug>/seed_<N>/<run_id>/  (immutable snapshot for this archive event)
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${MAIN}/scripts/mcmipl_data_slug.sh"

DATASET="${1:?DATASET required}"
SEED="${2:?SEED required}"
ARCH_ROOT="${MCMIPL_ARCHIVE_ROOT:-${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives}"
MCMIPL="${MCMIPL_DIR:-${MAIN}/baselines/mcmipl_official/MCMIPL}"
LOG_DIR="${MCMIPL_LOG_DIR:-${MAIN}/logs}"

SLUG="$(data_slug "$DATASET")"
RUN_ID="${MCMIPL_RUN_ID:-${SLUG}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
DEST="${ARCH_ROOT}/checkpoints/${SLUG}/seed_${SEED}"
RUN_SNAP="${ARCH_ROOT}/runs/${SLUG}/seed_${SEED}/${RUN_ID}"
STAMP="$(date -Iseconds)"

mkdir -p "$DEST" "$RUN_SNAP/RL-agent" "$RUN_SNAP/RL-log-merge" "$RUN_SNAP/logs"

SEED_DIR="${MCMIPL}/tmp/${SLUG}/RL-agent/seed_${SEED}"
LEGACY_DIR="${MCMIPL}/tmp/${SLUG}/RL-agent"
if [[ -d "$SEED_DIR" ]]; then
  mkdir -p "${DEST}/RL-agent"
  cp -a "${SEED_DIR}/." "${DEST}/RL-agent/"
  cp -a "${SEED_DIR}/." "${RUN_SNAP}/RL-agent/"
elif [[ -d "$LEGACY_DIR" ]]; then
  mkdir -p "${DEST}/RL-agent"
  shopt -s nullglob
  for f in "${LEGACY_DIR}"/*-epoch-*.pkl; do
    cp -a "$f" "${DEST}/RL-agent/"
    cp -a "$f" "${RUN_SNAP}/RL-agent/"
  done
  shopt -u nullglob
fi

if [[ -d "${MCMIPL}/tmp/${SLUG}/RL-log-merge" ]]; then
  mkdir -p "${DEST}/RL-log-merge"
  cp -a "${MCMIPL}/tmp/${SLUG}/RL-log-merge/." "${DEST}/RL-log-merge/"
  cp -a "${MCMIPL}/tmp/${SLUG}/RL-log-merge/." "${RUN_SNAP}/RL-log-merge/"
fi

LOG_SRC="${LOG_DIR}/train_${DATASET}_s${SEED}.log"
if [[ -f "$LOG_SRC" ]]; then
  cp -a "$LOG_SRC" "${DEST}/train_${DATASET}_s${SEED}.log"
  cp -a "$LOG_SRC" "${RUN_SNAP}/logs/train_${DATASET}_s${SEED}.log"
  grep -E '^(SR5:|=== DONE:|best!!!!!!!!)' "$LOG_SRC" | tail -30 > "${RUN_SNAP}/logs/crs_tail.txt" 2>/dev/null || true
fi

{
  echo "dataset=${DATASET}"
  echo "slug=${SLUG}"
  echo "seed=${SEED}"
  echo "run_id=${RUN_ID}"
  echo "archived_at=${STAMP}"
  echo "source_mcmipl=${MCMIPL}"
  echo "canonical_dir=${DEST}"
  echo "run_snapshot_dir=${RUN_SNAP}"
  if [[ -f "${DEST}/TIMING.txt" ]]; then
    echo "--- TIMING ---"
    cat "${DEST}/TIMING.txt"
  fi
  echo "rl_agent_files:"
  ls -la "${DEST}/RL-agent/" 2>/dev/null || echo "  (none)"
  echo "rl_log_merge_files:"
  ls -la "${DEST}/RL-log-merge/" 2>/dev/null || echo "  (none)"
} > "${DEST}/MANIFEST.txt"

cp -a "${DEST}/MANIFEST.txt" "${RUN_SNAP}/MANIFEST.txt"
ln -sfn "${RUN_SNAP}" "${ARCH_ROOT}/runs/${SLUG}/seed_${SEED}/LATEST" 2>/dev/null || true

bash "${MAIN}/scripts/mcmipl_append_run_registry.sh" "$DATASET" "$SEED" "$RUN_ID" "$DEST" "archived"

echo "[archive] ${DATASET} seed=${SEED} run_id=${RUN_ID}"
echo "          canonical -> ${DEST}"
echo "          snapshot  -> ${RUN_SNAP}"
