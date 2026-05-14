#!/usr/bin/env bash
# 将四数据集 MCMIPL 离线产物（TransE + 构图 pkl）快照到 artifacts，供 CPU 机或交接核对。
#
# 说明（与官方 MCMIPL 一致）：
# - TransE / OpenKE：**每个数据集一份** `tmp/<slug>/embeds/transe.pkl`，与 RL 的 random seed **无关**。
# - 主表 RL：**每个数据集 3 个 seed（0/1/2）** 在 **同一套** TransE 与同一套 `dataset.pkl`/`kg.pkl` 上训练。
#
# 用法：
#   cd main_table_experiments
#   bash scripts/snapshot_offline_embeddings_four_datasets.sh
#
# 可选：CREATE_TAR=1 额外生成 .tar.gz（默认 0，避免仓库根下大文件）
#
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MAIN_EXP="$(cd "${_SCRIPT_DIR}/.." && pwd)"
MCMIPL="${MCMIPL:-${_MAIN_EXP}/baselines/mcmipl_official/MCMIPL}"
OUT_ROOT="${_MAIN_EXP}/artifacts"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE="${OUT_ROOT}/offline_mcmipl_graph_transe_${STAMP}"
CREATE_TAR="${CREATE_TAR:-0}"

_missing=0

mkdir -p "$BUNDLE"

{
  echo "# offline_mcmipl_graph_transe | ${STAMP}"
  echo "# host: $(hostname 2>/dev/null || echo unknown)"
  echo "# MCMIPL source: ${MCMIPL}"
  echo "#"
  echo "# RL：四数据集各跑 seed 0/1/2；**不**为每个 seed 单独训练 TransE。"
  echo ""
} > "${BUNDLE}/README.txt"

while read -r DATA_NAME slug; do
  [[ -n "$DATA_NAME" ]] || continue
  _src="${MCMIPL}/tmp/${slug}"
  _emb="${_src}/embeds/transe.pkl"
  _ds="${_src}/dataset.pkl"
  _kg="${_src}/kg.pkl"
  _dst="${BUNDLE}/${slug}"
  mkdir -p "${_dst}/embeds"
  {
    echo "=== ${DATA_NAME} (slug=${slug}) ==="
    for f in "$_emb" "$_ds" "$_kg"; do
      if [[ ! -f "$f" ]]; then
        echo "[MISSING] $f"
        _missing=1
      else
        echo "[OK] $f ($(stat -c%s "$f" 2>/dev/null || echo ?) bytes)"
      fi
    done
    echo ""
  } >> "${BUNDLE}/README.txt"

  if [[ -f "$_emb" ]]; then cp -a "$_emb" "${_dst}/embeds/transe.pkl"; fi
  if [[ -f "$_ds" ]]; then cp -a "$_ds" "${_dst}/dataset.pkl"; fi
  if [[ -f "$_kg" ]]; then cp -a "$_kg" "${_dst}/kg.pkl"; fi
done <<'ROWS'
BOOK book
MOVIE movie
LAST_FM_STAR last_fm_star
YELP_STAR yelp_star
ROWS

(
  cd "$BUNDLE"
  find . -type f \( -name '*.pkl' -o -name 'README.txt' \) | sort | xargs -r sha256sum
) | tee "${BUNDLE}/SHA256SUMS.txt"

ln -sfn "$(basename "$BUNDLE")" "${OUT_ROOT}/offline_mcmipl_graph_transe_LATEST"

if [[ "$_missing" -ne 0 ]]; then
  echo "[ERROR] 有缺失 pkl，请按官方 README 完成 graph_init / OpenKE TransE 后再快照。" >&2
  exit 1
fi

if [[ "$CREATE_TAR" == "1" ]]; then
  _tar="${OUT_ROOT}/$(basename "$BUNDLE").tar.gz"
  tar -czf "$_tar" -C "$OUT_ROOT" "$(basename "$BUNDLE")"
  echo "Wrote archive: $_tar"
fi

chmod -R a+rX "$BUNDLE" 2>/dev/null || true
echo "OK: $BUNDLE"
echo "Latest -> ${OUT_ROOT}/offline_mcmipl_graph_transe_LATEST"
