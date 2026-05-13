#!/usr/bin/env bash
# 生成 Phase A 之后关键产物的清单与 sha256，便于 git 提交与 GPU 机核对。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MCMIPL="${REPO_ROOT}/main_table_experiments/baselines/mcmipl_official/MCMIPL"
OUT="${REPO_ROOT}/main_table_experiments/artifacts"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
MAN="${OUT}/phase_a_manifest_${STAMP}.txt"

{
  echo "# Phase A artifact manifest"
  echo "# generated: $(date -Iseconds 2>/dev/null || date)"
  echo "# host: $(hostname 2>/dev/null || echo unknown)"
  echo ""
  echo "## git HEAD"
  (cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null) || echo "(not a git checkout)"
  echo ""
  echo "## tmp/* sizes (graph + TransE)"
  if [[ -d "$MCMIPL/tmp" ]]; then
    ls -laR "$MCMIPL/tmp" 2>/dev/null | head -500
    echo ""
    echo "## sha256: tmp/**/transe.pkl dataset.pkl kg.pkl"
    find "$MCMIPL/tmp" \( -name 'transe.pkl' -o -name 'dataset.pkl' -o -name 'kg.pkl' \) -type f 2>/dev/null | sort | while read -r f; do
      sha256sum "$f"
    done
  else
    echo "(no $MCMIPL/tmp)"
  fi
  echo ""
  echo "## MCMIPL data book/movie UI_data (if present)"
  for d in book movie; do
    ud="$MCMIPL/data/$d/UI_data"
    if [[ -d "$ud" ]]; then
      ls -la "$ud"
      find "$ud" -maxdepth 1 -type f -name '*.pkl' -print -exec sha256sum {} \;
    fi
  done
} | tee "$MAN"

echo "Wrote $MAN"
ln -sf "$(basename "$MAN")" "${OUT}/phase_a_manifest_latest.txt" 2>/dev/null || true
echo "Also symlink: ${OUT}/phase_a_manifest_latest.txt"
