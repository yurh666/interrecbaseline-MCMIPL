#!/usr/bin/env bash
# 与 run_book_gpu_seeds_012.sh 同一套 GPU 环境与调用方式；唯一区别：
#   1) 先等 BOOK seed 0/1/2 在日志里全部 DONE（也可用 SKIP_WAIT=1 跳过）；
#   2) 再按主表顺序跑「除 BOOK 外」三个数据集：LAST_FM_STAR → YELP_STAR → MOVIE，
#      每个集同样：for s in 0 1 2; do run_mcmipl.sh <dataset> $s 50 100 10。
#
# TransE / OpenKE：与 BOOK 一样，不在本脚本里做；需事先保证各集已有
#   MCMIPL/tmp/<slug>/embeds/transe.pkl（或接受 RL 侧行为），逻辑顺序与 BOOK 一致。
#
# 环境（与 BOOK runner 对齐）：
#   SKIP_WAIT=1     不等待 BOOK
#   POLL_SEC=120    等 BOOK 时轮询间隔
#   QUEUE_RUN_LOG=path  可选；不设则用 logs/tmux_after_book_three_datasets_gpu_runner.log
#
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_ROOT"

unset MCMIPL_FORCE_CPU
if [[ -d /root/autodl-tmp ]]; then
  export TMPDIR=/root/autodl-tmp/pip-tmp
  export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
else
  mkdir -p "$_ROOT/.tmp/pip-tmp" "$_ROOT/.tmp/pip-cache"
  export TMPDIR="${TMPDIR:-$_ROOT/.tmp/pip-tmp}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$_ROOT/.tmp/pip-cache}"
fi
mkdir -p logs

RUN_LOG="${QUEUE_RUN_LOG:-${_ROOT}/logs/tmux_after_book_three_datasets_gpu_runner.log}"
exec >> "${RUN_LOG}" 2>&1

echo "====== runner start $(date) | log=${RUN_LOG} ======"

LOG_DIR="${MCMIPL_LOG_DIR:-${_ROOT}/logs}"
POLL_SEC="${POLL_SEC:-120}"
SKIP_WAIT="${SKIP_WAIT:-0}"

_book_done() {
  local s="$1"
  local f="${LOG_DIR}/train_BOOK_s${s}.log"
  [[ -f "${f}" ]] || return 1
  grep -q "=== DONE: BOOK seed=${s} at" "${f}"
}

_wait_book_three_seeds() {
  echo "=== 等待 BOOK seed 0/1/2 DONE（同 BOOK 日志路径）：${LOG_DIR}/train_BOOK_s*.log ==="
  while true; do
    if _book_done 0 && _book_done 1 && _book_done 2; then
      echo "=== BOOK 三 seed 均已 DONE | $(date) ==="
      return 0
    fi
    for s in 0 1 2; do
      if _book_done "${s}"; then
        echo "  [$(date +%H:%M:%S)] BOOK seed=${s} OK"
      else
        echo "  [$(date +%H:%M:%S)] BOOK seed=${s} 未 DONE → ${LOG_DIR}/train_BOOK_s${s}.log"
      fi
    done
    echo "  ${POLL_SEC}s 后重试…"
    sleep "${POLL_SEC}"
  done
}

if [[ "${SKIP_WAIT}" != "1" ]]; then
  _wait_book_three_seeds
else
  echo "=== SKIP_WAIT=1：不等待 BOOK ==="
fi

bash "${_ROOT}/scripts/mcmipl_gpu_env_autofix.sh" || echo "[runner] WARN: autofix 退出非零，继续" >&2

for _dataset in LAST_FM_STAR YELP_STAR MOVIE; do
  echo "====== ${_dataset} seeds 0–2 start $(date) ======"
  for s in 0 1 2; do
    echo "====== ${_dataset} seed=${s} $(date) ======"
    bash "${_ROOT}/run_mcmipl.sh" "${_dataset}" "${s}" 50 100 10
  done
  echo "====== ${_dataset} seeds 0–2 finished $(date) ======"
done

echo "====== LAST_FM + YELP + MOVIE 全部完成 $(date) ======"
