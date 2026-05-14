#!/usr/bin/env bash
# RL 阶段 GPU vs CPU 墙钟对比 + 阶段占比（不含 TransE；TransE 在 OpenKE 里独立跑）
#
# 原理：RL_model 在 MCMIPL_RL_PHASE_TIMINGS=1 时打印
#   [MCMIPL_RL_PHASE_TIMINGS] eval_s / train_sampling_s / eval_pct /
#   train_sampling_pct / eval_calls
# 其中 eval = 全部 dqn_evaluate；train_sampling = 每步采样 100×episode 循环（含环境 + RL 更新）
#
# 默认缩短评测用户数以快速对比（仅剖析用，主实验请勿设置）：
#   MCMIPL_RL_PROFILE_TEST_USERS=200
#
# 用法（需已 graph_init + transe.pkl）：
#   cd main_table_experiments
#   bash scripts/profile_rl_gpu_cpu_compare.sh
#
# 可选环境：MCMIPL_GPU_PYTHON、PROF_DATA_NAME、PROF_MAX_STEPS、PROF_SAMPLE_TIMES、
#          MCMIPL_RL_PROFILE_TEST_USERS
#
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MAIN_EXP="$(cd "${_SCRIPT_DIR}/.." && pwd)"
MCMIPL="${MCMIPL:-${_MAIN_EXP}/baselines/mcmipl_official/MCMIPL}"
PYTHON="${MCMIPL_GPU_PYTHON:-/root/miniconda3/envs/mcmipl-baseline-gpu/bin/python}"

# pip 安装的 nvidia-* wheel 在 site-packages/nvidia/*/lib；不设 LD_LIBRARY_PATH 时
# 常报 libnvrtc.so.12 找不到，DGL GraphBolt 无法加载（与是否有物理 GPU 无关）。
_env_root="$(cd "$(dirname "${PYTHON}")/.." && pwd)"
for _d in "${_env_root}"/lib/python*/site-packages/nvidia/*/lib; do
  if [[ -d "${_d}" ]]; then
    export LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done
unset _env_root _d

PROF_DATA="${PROF_DATA_NAME:-BOOK}"
PROF_STEPS="${PROF_MAX_STEPS:-10}"
PROF_SAMPLES="${PROF_SAMPLE_TIMES:-30}"
PROF_TEST_USERS="${MCMIPL_RL_PROFILE_TEST_USERS:-200}"

mkdir -p "${_MAIN_EXP}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="${_MAIN_EXP}/logs/rl_gpu_cpu_profile_${STAMP}.log"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] PYTHON 不可用: ${PYTHON}，请设置 MCMIPL_GPU_PYTHON" | tee "$REPORT"
  exit 1
fi

_common_py_args=(
  -u RL_model.py
  --data_name "$PROF_DATA"
  --embed transe
  --seed 0
  --gpu 0
  --max_steps "$PROF_STEPS"
  --sample_times "$PROF_SAMPLES"
  --attr_num 20
  --choice_num 4
  --max_turn 15
  --eval_num 10
  --save_num "$PROF_STEPS"
)

{
  echo "=== profile_rl_gpu_cpu_compare.sh | STAMP=${STAMP} ==="
  echo "MCMIPL=${MCMIPL} PYTHON=${PYTHON}"
  echo "PROF_DATA=${PROF_DATA} PROF_STEPS=${PROF_STEPS} PROF_SAMPLES=${PROF_SAMPLES}"
  echo "MCMIPL_RL_PROFILE_TEST_USERS=${PROF_TEST_USERS} （剖析专用；论文结果请 unset）"
  echo
  echo "========== GPU run (unset MCMIPL_FORCE_CPU) =========="
} | tee "$REPORT"

cd "$MCMIPL"
export MCMIPL_RL_PHASE_TIMINGS=1
export MCMIPL_RL_PROFILE_TEST_USERS="$PROF_TEST_USERS"
unset MCMIPL_FORCE_CPU || true

T0="$(date +%s)"
"${PYTHON}" "${_common_py_args[@]}" 2>&1 | tee -a "$REPORT"
T1="$(date +%s)"
echo "GPU wall_clock_s=$((T1 - T0))" | tee -a "$REPORT"

{
  echo
  echo "========== CPU run (MCMIPL_FORCE_CPU=1) =========="
} | tee -a "$REPORT"

export MCMIPL_FORCE_CPU=1
T0="$(date +%s)"
"${PYTHON}" "${_common_py_args[@]}" 2>&1 | tee -a "$REPORT"
T1="$(date +%s)"
echo "CPU wall_clock_s=$((T1 - T0))" | tee -a "$REPORT"

{
  echo
  echo "=== 如何解读见 docs/RL_PHASE_TIMING_METHODOLOGY.md ==="
  echo "完整日志: ${REPORT}"
} | tee -a "$REPORT"
