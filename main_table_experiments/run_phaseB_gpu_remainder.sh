#!/usr/bin/env bash
# Phase B 队列尾部：在 BOOK 的 GPU 训练结束前不启动第二进程；随后依次 GPU 跑 YELP_STAR seed2（若未 DONE）、MOVIE 三 seed。
# 与 handoff 一致：max_steps=50, sample_times=100, eval_num=10；强制 GPU conda Python（勿设 MCMIPL_FORCE_CPU）。
set -euo pipefail
_MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_MAIN"

unset MCMIPL_FORCE_CPU
export MCMIPL_GPU_PYTHON="${MCMIPL_GPU_PYTHON:-/root/miniconda3/envs/mcmipl-baseline-gpu/bin/python}"

mkdir -p logs
_RUN_LOG="logs/tmux_phaseB_remainder_$(date +%Y%m%d_%H%M%S).log"
exec >> "$_RUN_LOG" 2>&1

echo "====== phase B remainder start $(date) | log=$_RUN_LOG ======"
echo "=== MCMIPL_GPU_PYTHON=$MCMIPL_GPU_PYTHON ==="

echo "=== 等待 BOOK 的 RL 进程结束（单卡串行，避免双进程抢 GPU）==="
while pgrep -f 'RL_model.py --data_name BOOK' >/dev/null 2>&1; do
  echo "  [$(date +%H:%M:%S)] 仍有 BOOK 训练，120s 后再检查"
  sleep 120
done
echo "=== BOOK 已无 RL 进程，继续队列 ==="

if grep -q '=== DONE: YELP_STAR seed=2' logs/train_YELP_STAR_s2.log 2>/dev/null; then
  echo "=== SKIP YELP_STAR seed=2（已有 DONE）==="
else
  echo "=== RUN YELP_STAR seed=2 ==="
  REQUIRE_TRANSE=1 bash "$_MAIN/run_mcmipl.sh" YELP_STAR 2 50 100 10
fi

for s in 0 1 2; do
  if grep -q "=== DONE: MOVIE seed=${s} " logs/train_MOVIE_s"${s}".log 2>/dev/null; then
    echo "=== SKIP MOVIE seed=${s}（已有 DONE）==="
  else
    echo "=== RUN MOVIE seed=${s} ==="
    REQUIRE_TRANSE=1 bash "$_MAIN/run_mcmipl.sh" MOVIE "${s}" 50 100 10
  fi
done

echo "====== phase B remainder finished $(date) ======"
