#!/usr/bin/env bash
# Check whether a dataset/seed already has a complete archived Phase-B run (copy-only audit).
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${MAIN}/scripts/mcmipl_data_slug.sh"

DATASET="${1:?DATASET}"
SEED="${2:?SEED}"
ARCH_ROOT="${MCMIPL_ARCHIVE_ROOT:-${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives}"

SLUG="$(data_slug "$DATASET")"
DEST="${ARCH_ROOT}/checkpoints/${SLUG}/seed_${SEED}"
RL="${DEST}/RL-agent"

_min_epoch_pkl() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  shopt -s nullglob
  local f50 f100
  f50=("$d"/*-epoch-50.pkl)
  f100=("$d"/*-epoch-100.pkl)
  shopt -u nullglob
  [[ ${#f50[@]} -gt 0 || ${#f100[@]} -gt 0 ]]
}

if [[ ! -f "${DEST}/MANIFEST.txt" ]]; then
  echo "missing_manifest"
  exit 1
fi
if ! _min_epoch_pkl "$RL"; then
  echo "missing_final_checkpoint"
  exit 1
fi
echo "archived_ok"
exit 0
