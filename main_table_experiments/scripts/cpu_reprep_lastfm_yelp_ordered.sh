#!/usr/bin/env bash
# CPU：按顺序 LAST_FM_STAR -> YELP_STAR 执行 prepare_data.sh + run_graph_init.sh。
# 可与 BOOK 的 GPU RL 并行（读写不同 data/*/ 与 tmp/*/ 路径）。
# 结束前 touch logs/.reprep_lastfm_yelp_done 供 GPU 队列等待。
set -euo pipefail
_MTE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_BASE="${_MTE}/baselines/mcmipl_official"
export CUDA_VISIBLE_DEVICES=""
unset NVIDIA_VISIBLE_DEVICES || true

mkdir -p "${_MTE}/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
_LOGF="${_MTE}/logs/cpu_reprep_lastfm_yelp_${STAMP}.log"
exec > >(tee -a "${_LOGF}") 2>&1

echo "====== cpu_reprep_lastfm_yelp ${STAMP} start $(date) | tee log=${_LOGF} ======"

echo "=== [0] archive previous tmp/data/logs snapshot ==="
bash "${_MTE}/scripts/archive_wrong_preprocess_lastfm_yelp.sh"

for D in LAST_FM_STAR YELP_STAR; do
  echo ""
  echo "=== ${D}: prepare_data ==="
  bash "${_BASE}/scripts/prepare_data.sh" "${D}"
  echo "=== ${D}: graph_init ==="
  bash "${_BASE}/scripts/run_graph_init.sh" "${D}"
done

echo ""
echo "=== Reprep done. TransE: 若 kg/dataset 相对原 transe 训练图有变，请在 GPU 侧按 MCMIPL README 重训 transe.pkl 后再 RL。==="

touch "${_MTE}/logs/.reprep_lastfm_yelp_done_${STAMP}"
ln -sf ".reprep_lastfm_yelp_done_${STAMP}" "${_MTE}/logs/.reprep_lastfm_yelp_done"
echo "=== touched ${_MTE}/logs/.reprep_lastfm_yelp_done → .reprep_lastfm_yelp_done_${STAMP} ==="
echo "====== cpu_reprep_lastfm_yelp finished $(date) ======"
