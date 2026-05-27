#!/usr/bin/env bash
# Archive Phase-A artifacts (dataset/kg/transe + review dicts) for InterRec eval prep.
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${MAIN}/scripts/mcmipl_data_slug.sh"

DATASET="${1:?DATASET required}"
ARCH_ROOT="${MCMIPL_ARCHIVE_ROOT:-${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives}"
MCMIPL="${MCMIPL_DIR:-${MAIN}/baselines/mcmipl_official/MCMIPL}"

SLUG="$(data_slug "$DATASET")"
DEST="${ARCH_ROOT}/phase_a/${SLUG}"
STAMP="$(date -Iseconds)"

mkdir -p "${DEST}/embeds" "${DEST}/UI_Interaction_data"

for f in dataset.pkl kg.pkl; do
  [[ -f "${MCMIPL}/tmp/${SLUG}/${f}" ]] && cp -a "${MCMIPL}/tmp/${SLUG}/${f}" "${DEST}/${f}"
done
[[ -f "${MCMIPL}/tmp/${SLUG}/embeds/transe.pkl" ]] && \
  cp -a "${MCMIPL}/tmp/${SLUG}/embeds/transe.pkl" "${DEST}/embeds/transe.pkl"

DATA_DIR="${MCMIPL}/data/${SLUG}/UI_Interaction_data"
for f in review_dict_train.json review_dict_valid.json review_dict_test.json; do
  [[ -f "${DATA_DIR}/${f}" ]] && cp -a "${DATA_DIR}/${f}" "${DEST}/UI_Interaction_data/${f}"
done

{
  echo "dataset=${DATASET}"
  echo "slug=${SLUG}"
  echo "archived_at=${STAMP}"
  echo "files:"
  find "$DEST" -type f | sort
} > "${DEST}/MANIFEST.txt"

echo "[archive phase-a] ${DATASET} -> ${DEST}"
