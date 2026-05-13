#!/usr/bin/env bash
# Phase A — 纯 CPU 阶段（建议在无 GPU 机器或 CPU conda 环境执行）
#
# 做什么（顺序固定）：
#   1) 可选：InterRec 时序重建 BOOK+MOVIE（覆盖 pkl 后必须 graph_init）
#   2) 对每个数据集：prepare_data.sh → graph_init.py
#
# 不做什么：
#   - TransE / OpenKE（可在 GPU 机做，见 Phase B 说明）
#   - RL_model.py
#
# 环境：export CUDA_VISIBLE_DEVICES=""，避免误占 GPU；请保证 `python` 指向 CPU 侧环境（如 mcmipl-reproduce）。
#
# 用法：
#   bash run_pipeline_phase_cpu.sh BOOK MOVIE
#   RUN_REBUILD_BOOK_MOVIE=1 bash run_pipeline_phase_cpu.sh    # 先跑 InterRec 全链路再 graph_init
#
set -euo pipefail

MAIN_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${MAIN_EXP}/baselines/mcmipl_official"
REBUILD_SCRIPT="${BASELINE}/scripts/rebuild_book_movie_interrec_temporal.sh"

export CUDA_VISIBLE_DEVICES=""
unset NVIDIA_VISIBLE_DEVICES || true

if [[ $# -gt 0 ]]; then
  DATASETS=( "$@" )
else
  DATASETS=( LAST_FM_STAR YELP_STAR BOOK MOVIE )
fi

echo "=== Phase A (CPU) | datasets: ${DATASETS[*]} | $(date) ==="
echo "=== python: $(command -v python) ==="

if [[ "${RUN_REBUILD_BOOK_MOVIE:-0}" == "1" ]]; then
  echo "=== [Phase A] RUN_REBUILD_BOOK_MOVIE=1 → ${REBUILD_SCRIPT} ==="
  bash "${REBUILD_SCRIPT}"
fi

for d in "${DATASETS[@]}"; do
  echo ""
  echo "=== [Phase A] prepare_data: ${d} ==="
  bash "${BASELINE}/scripts/prepare_data.sh" "${d}"
  echo "=== [Phase A] graph_init: ${d} ==="
  bash "${BASELINE}/scripts/run_graph_init.sh" "${d}"
done

echo ""
echo "=== Phase A (CPU) 完成 | $(date) ==="
echo "下一台（GPU 机）请："
echo "  1) 同步本仓库与 MCMIPL/data/、MCMIPL/tmp/（至少含各数据集的 transe.pkl）"
echo "  2) 若尚无 transe.pkl：按官方 MCMIPL README 用 OpenKE 训练 TransE 并放到 MCMIPL/tmp/<dataset>/embeds/"
echo "  3) bash ${MAIN_EXP}/run_pipeline_phase_gpu.sh <数据集...>"
