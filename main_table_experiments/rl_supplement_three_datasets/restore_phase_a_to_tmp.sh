#!/usr/bin/env bash
# Restore Phase-A archives into MCMIPL/tmp/<slug>/ for RL-only servers.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="${REPO}/experiments/mcmipl_interrec_protocol_eval/archives/phase_a"
MCMIPL="${MCMIPL_DIR:-${REPO}/main_table_experiments/baselines/mcmipl_official/MCMIPL}"

for slug in book last_fm_star movie; do
  src="${ARCH}/${slug}"
  dest="${MCMIPL}/tmp/${slug}"
  [[ -d "$src" ]] || { echo "[skip] no archive $slug"; continue; }
  mkdir -p "${dest}/embeds"
  for f in dataset.pkl kg.pkl; do
    [[ -f "${src}/${f}" ]] && cp -a "${src}/${f}" "${dest}/${f}" && echo "[ok] ${slug}/${f}"
  done
  [[ -f "${src}/embeds/transe.pkl" ]] && cp -a "${src}/embeds/transe.pkl" "${dest}/embeds/transe.pkl" && echo "[ok] ${slug}/embeds/transe.pkl"
done

echo "=== restore_phase_a done ==="
