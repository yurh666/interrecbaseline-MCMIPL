#!/usr/bin/env bash
# 顺序训练四个数据集的 TransE，并周期性记录 nvidia-smi（CSV）到单独日志。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCM="${ROOT}/baselines/mcmipl_official/MCMIPL"
LOGDIR="${ROOT}/logs"
mkdir -p "${LOGDIR}"

_env="${MCMIPL_TRAIN_PYTHON:-/root/miniconda3/envs/mcmipl-baseline-gpu/bin/python}"
if [[ ! -x "${_env}" ]]; then
  echo "未找到 Python: ${_env}，请设置 MCMIPL_TRAIN_PYTHON" >&2
  exit 1
fi
for _d in "$(dirname "$_env")/../lib/python3.10/site-packages/nvidia"/*/lib; do
  [[ -d "$_d" ]] && export LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
done

STAMP="$(date +%Y%m%d_%H%M%S)"
GPU_LOG="${LOGDIR}/transe_gpu_monitor_${STAMP}.log"
RUN_LOG="${LOGDIR}/transe_four_datasets_${STAMP}.log"
EPOCHS="${TRANSE_EPOCHS:-200}"
INTERVAL_SEC="${GPU_SAMPLE_INTERVAL_SEC:-15}"

{
  while true; do
    echo "==== $(date -Is) ====" >> "${GPU_LOG}"
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader >> "${GPU_LOG}" 2>&1 || echo "nvidia-smi failed $?" >> "${GPU_LOG}"
    sleep "${INTERVAL_SEC}"
  done
} &
MON_PID=$!

cleanup() { kill "${MON_PID}" 2>/dev/null || true; }
trap cleanup EXIT

cd "${MCM}"
{
  echo "GPU 采样间隔 ${INTERVAL_SEC}s -> ${GPU_LOG}"
  echo "训练日志追加 -> ${RUN_LOG}"
  for DN in BOOK MOVIE LAST_FM_STAR YELP_STAR; do
    echo "========== $(date -Is) START ${DN} epochs=${EPOCHS} =========="
    "${_env}" train_transe_from_kg.py --data_name "${DN}" --epochs "${EPOCHS}" --device cuda
    "${_env}" verify_transe_phase_b.py --data_name "${DN}"
  done
  echo "========== $(date -Is) ALL DONE =========="
} 2>&1 | tee -a "${RUN_LOG}"
