#!/usr/bin/env bash
# BOOK seed 0/1/2 全部结束后，串行跑 Phase B（仅 GPU RL，不跑 Phase A）：
#   LAST_FM_STAR → YELP_STAR → MOVIE（主表四数据集中编号 1、2、4，跳过 BOOK=3）。
#
# 默认与 run_book_gpu_seeds_012.sh 一致：MAX_STEPS=50 SAMPLE_TIMES=100 EVAL_NUM=10，SEEDS=0 1 2。
# 实现上等价于： WAIT → bash run_pipeline_phase_gpu.sh LAST_FM_STAR YELP_STAR MOVIE
#
# 环境：
#   SKIP_WAIT=1              跳过「等 BOOK」直接开跑（仅调试用）
#   POLL_SEC=120             轮询间隔（秒）
#   REQUIRE_TRANSE=0|1       传给 run_pipeline_phase_gpu.sh（缺 transe.pkl 是否直接失败）
#   MCMIPL_GPU_PYTHON / MAX_STEPS / SAMPLE_TIMES / EVAL_NUM / SEEDS 等同 Phase B 主脚本
#
# 建议：LAST_FM 图已更新而 transe 未重训时，见 artifacts/PHASE_A_LASTFM_YELP_INTERREC_TEMPORAL_20260513.md，
# 按一致性要求决定是否先 OpenKE 再设 REQUIRE_TRANSE=1。
#
set -euo pipefail

_MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${_MAIN}/main_table_experiments"

LOG_DIR="${MCMIPL_LOG_DIR:-${_MAIN}/main_table_experiments/logs}"
POLL_SEC="${POLL_SEC:-120}"
SKIP_WAIT="${SKIP_WAIT:-0}"

_book_done() {
  local s="$1"
  local f="${LOG_DIR}/train_BOOK_s${s}.log"
  [[ -f "${f}" ]] || return 1
  grep -q "=== DONE: BOOK seed=${s} at" "${f}"
}

_wait_book_three_seeds() {
  echo "=== 等待 BOOK seed 0/1/2 在日志中出现 DONE：${LOG_DIR}/train_BOOK_s*.log ==="
  while true; do
    if _book_done 0 && _book_done 1 && _book_done 2; then
      echo "=== BOOK 三 seed 均已 DONE | $(date) ==="
      return 0
    fi
    for s in 0 1 2; do
      if _book_done "${s}"; then
        echo "  [$(date +%H:%M:%S)] BOOK seed=${s} OK"
      else
        echo "  [$(date +%H:%M:%S)] BOOK seed=${s} 尚未 DONE → ${LOG_DIR}/train_BOOK_s${s}.log"
      fi
    done
    echo "  ${POLL_SEC}s 后重试…"
    sleep "${POLL_SEC}"
  done
}

if [[ "${SKIP_WAIT}" != "1" ]]; then
  _wait_book_three_seeds
else
  echo "=== SKIP_WAIT=1：不等待 BOOK，直接 Phase B ==="
fi

echo ""
echo "=== Phase B：LAST_FM_STAR YELP_STAR MOVIE | $(date) ==="
bash "${_MAIN}/main_table_experiments/run_pipeline_phase_gpu.sh" LAST_FM_STAR YELP_STAR MOVIE

echo ""
echo "=== 队列完成 | $(date) ==="
