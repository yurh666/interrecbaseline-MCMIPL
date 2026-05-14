#!/usr/bin/env bash
# 将 LAST_FM_STAR / YELP_STAR 在「重跑 graph_init 之前」的 tmp、部分 data、以及训练日志打快照，标记为预处理/构图与当前方法不一致时期的版本。
# 说明：/root/methodnew.md 描述的是交互式 CRS 方法，不是数据 CSV 切分；本仓库数据协议见 interrec/configs 与 baselines 文档。
set -euo pipefail
_MTE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_BASE="$_MTE/baselines/mcmipl_official"
_MC="${_BASE}/MCMIPL"
_LOG="${_MTE}/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
ARCH="${_MC}/_archive_wrong_preprocess_${STAMP}"
mkdir -p "${ARCH}"

cat > "${ARCH}/README_WHY_THIS_ARCHIVE.txt" <<EOF
归档时间（UTC 近似本地）: ${STAMP}
原因: 主表前两个数据集 LAST_FM_STAR、YELP_STAR 的 MCMIPL 侧预处理/构图需与当前 InterRec+MCMIPL 流程重新对齐后重跑。
本快照包含重跑 graph_init 之前的:
  - tmp/last_fm_star, tmp/yelp_star
  - data/lastfm_star 与 data/yelp_star 下 UI_Interaction_data、UI_data（若存在）
  - logs 中 train_LAST_FM_STAR_s*.log / train_YELP_STAR_s*.log 副本

注意: 若 dataset.pkl/kg.pkl 相对 transe.pkl 已变更，OpenKE/TransE 需按官方 README 对同一数据集重训后再跑 RL；否则嵌入与图不一致。

methodnew.md 为方法叙事稿，数据切分以本仓库 interrec 与 MCMIPL 官方 data 为准。
EOF

echo "=== Archiving to ${ARCH} ==="
for slug in last_fm_star yelp_star; do
  if [[ -d "${_MC}/tmp/${slug}" ]]; then
    cp -a "${_MC}/tmp/${slug}" "${ARCH}/tmp_${slug}"
    echo "OK tmp ${slug}"
  else
    echo "WARN: missing ${_MC}/tmp/${slug}"
  fi
done

for ddir in lastfm_star yelp_star; do
  if [[ -d "${_MC}/data/${ddir}" ]]; then
    mkdir -p "${ARCH}/data_${ddir}"
    for sub in UI_data UI_Interaction_data; do
      if [[ -d "${_MC}/data/${ddir}/${sub}" ]]; then
        cp -a "${_MC}/data/${ddir}/${sub}" "${ARCH}/data_${ddir}/${sub}"
      fi
    done
    echo "OK data ${ddir}"
  fi
done

mkdir -p "${ARCH}/train_logs_prev"
for f in "${_LOG}"/train_LAST_FM_STAR_s*.log "${_LOG}"/train_YELP_STAR_s*.log; do
  [[ -f "$f" ]] && cp -a "$f" "${ARCH}/train_logs_prev/" && echo "OK log $(basename "$f")"
done

echo "=== Archive complete: ${ARCH} ==="
printf '%s\n' "$ARCH"
