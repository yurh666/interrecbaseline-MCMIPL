#!/usr/bin/env bash
# Phase B — GPU 本机阶段（TransE 已就绪后的 RL 训练）
#
# 前置条件：
#   - 已在 CPU 机（或本机）跑完 Phase A：data + graph 一致
#   - 每个要训的数据集存在：MCMIPL/tmp/<slug>/embeds/transe.pkl
#     （slug: last_fm_star | yelp_star | book | movie）
#
# TransE：
#   本仓库未封装 OpenKE 一键命令；若在 GPU 机上生成 embedding，请遵官方 README，
#   产物路径与 MCMIPL 一致即可。已有 pkl 则可直接跑本脚本。
#
# RL：
#   调用 run_mcmipl.sh，默认使用 MCMIPL_GPU_PYTHON（若存在且未设置 MCMIPL_FORCE_CPU）。
#
# 用法：
#   bash run_pipeline_phase_gpu.sh BOOK MOVIE
#   MAX_STEPS=50 SAMPLE_TIMES=100 EVAL_NUM=10 bash run_pipeline_phase_gpu.sh BOOK
#   REQUIRE_TRANSE=1 bash run_pipeline_phase_gpu.sh BOOK   # 缺 transe.pkl 则失败（默认仅 WARN）
#
set -euo pipefail

MAIN_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCMIPL="${MAIN_EXP}/baselines/mcmipl_official/MCMIPL"

if [[ "${MCMIPL_FORCE_CPU:-}" == "1" ]]; then
  export MCMIPL_FORCE_CPU=1
  echo "=== Phase B 模式: 强制 CPU RL（MCMIPL_FORCE_CPU=1）==="
else
  unset MCMIPL_FORCE_CPU
  echo "=== Phase B 模式: 优先 GPU Python（未设置 MCMIPL_FORCE_CPU）==="
fi

MAX_STEPS="${MAX_STEPS:-50}"
SAMPLE_TIMES="${SAMPLE_TIMES:-100}"
EVAL_NUM="${EVAL_NUM:-10}"
SEEDS=( ${SEEDS:-0 1 2} )
REQUIRE_TRANSE="${REQUIRE_TRANSE:-0}"

if [[ $# -gt 0 ]]; then
  DATASETS=( "$@" )
else
  DATASETS=( BOOK MOVIE )
fi

_transe_path() {
  local data_name="$1"
  local slug=""
  case "$data_name" in
    LAST_FM_STAR) slug=last_fm_star ;;
    YELP_STAR) slug=yelp_star ;;
    BOOK) slug=book ;;
    MOVIE) slug=movie ;;
    *)
      echo "Unknown DATASET: $data_name" >&2
      return 2
      ;;
  esac
  printf '%s' "${MCMIPL}/tmp/${slug}/embeds/transe.pkl"
}

echo "=== Phase B (GPU RL) | datasets: ${DATASETS[*]} | seeds: ${SEEDS[*]} | $(date) ==="
echo "=== MCMIPL_GPU_PYTHON=\${MCMIPL_GPU_PYTHON:-默认} MCMIPL_CPU_PYTHON=\${MCMIPL_CPU_PYTHON:-默认} ==="

for d in "${DATASETS[@]}"; do
  p="$(_transe_path "$d")" || exit 2
  if [[ ! -f "$p" ]]; then
    echo "[Phase B] WARN: 缺少 TransE: $p" >&2
    if [[ "$REQUIRE_TRANSE" == "1" ]]; then
      echo "[Phase B] REQUIRE_TRANSE=1，退出。" >&2
      exit 1
    fi
  else
    echo "[Phase B] OK TransE: $p"
  fi
done

for d in "${DATASETS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo ""
    echo "=== [Phase B] run_mcmipl: $d seed=$s ==="
    bash "${MAIN_EXP}/run_mcmipl.sh" "$d" "$s" "$MAX_STEPS" "$SAMPLE_TIMES" "$EVAL_NUM"
  done
done

echo ""
echo "=== Phase B (GPU RL) 完成 | $(date) ==="
