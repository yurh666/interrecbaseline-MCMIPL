#!/usr/bin/env bash
# 从 LAST_FM_STAR seed=0 的 epoch-20 断点续跑，然后顺序跑 LAST_FM s1/s2，
# 再跑 YELP_STAR（3 个 seed）。需在 screen/tmux 下执行长跑。
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_APPEND="${MASTER_APPEND:-${MAIN}/logs/phase_b_cpu_master_20260516.log}"

export MCMIPL_FORCE_CPU="${MCMIPL_FORCE_CPU:-1}"
export MCMIPL_CPU_PYTHON="${MCMIPL_CPU_PYTHON:-/home/yrh666/venvs/mcmipl-cpu/bin/python}"
export MCMIPL_SAVE_NUM="${MCMIPL_SAVE_NUM:-10}"
export REQUIRE_TRANSE="${REQUIRE_TRANSE:-1}"

{
  echo ""
  echo "=== Phase B RESUME | $(date) ==="
  echo "=== LAST_FM_STAR s0 load_rl_epoch=20 → 然后 s1 s2 → YELP_STAR SEEDS=${SEEDS:-0 1 2} ==="
} | tee -a "${MASTER_APPEND}"

cd "${MAIN}"

# seed=0：从已成功落盘的 RL checkpoint 续训（等价于接上中断前的权重）
export MCMIPL_LOAD_RL_EPOCH="${MCMIPL_RESUME_FROM_EPOCH:-20}"
bash run_mcmipl.sh LAST_FM_STAR 0
unset MCMIPL_LOAD_RL_EPOCH

bash run_mcmipl.sh LAST_FM_STAR 1
bash run_mcmipl.sh LAST_FM_STAR 2

SEEDS="${SEEDS:-0 1 2}" bash run_pipeline_phase_gpu.sh YELP_STAR

{
  echo ""
  echo "=== Phase B RESUME 流水线结束 | $(date) ==="
} | tee -a "${MASTER_APPEND}"
