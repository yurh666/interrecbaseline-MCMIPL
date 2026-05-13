#!/bin/bash
set -o pipefail
# 默认 max_steps=50：与 run_all_datasets.sh 中「LAST_FM 之后数据集」一致（eval_num=10 仍能在 step10 评测）。
_RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=${1:-LAST_FM_STAR}
SEED=${2:-0}
MAX_STEPS=${3:-50}
SAMPLE_TIMES=${4:-100}
EVAL_NUM=${5:-10}

MCMIPL_DIR="${MCMIPL_DIR:-$_RUN_ROOT/baselines/mcmipl_official/MCMIPL}"
LOG_DIR="${MCMIPL_LOG_DIR:-$_RUN_ROOT/logs}"

_mcmipl_scratch_pick() {
  local c tdir
  for c in "${MCMIPL_LARGE_TMP:-}" /root/autodl-tmp "${MCMIPL_LARGE_TMP_FALLBACK:-}"; do
    [[ -z "$c" ]] && continue
    tdir="$c/.mcmipl_wtest_$$"
    if mkdir -p "$tdir" 2>/dev/null && rmdir "$tdir" 2>/dev/null; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}
if _SCR="$(_mcmipl_scratch_pick)"; then
  export TMPDIR="$_SCR/pip-tmp"
  export PIP_CACHE_DIR="$_SCR/pip-cache"
  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
  if [[ -z "${CONDA_PKGS_DIRS:-}" ]]; then
    export CONDA_PKGS_DIRS="$_SCR/conda-pkgs"
    mkdir -p "$CONDA_PKGS_DIRS"
  fi
fi

GPU_PY="${MCMIPL_GPU_PYTHON:-/root/miniconda3/envs/mcmipl-baseline-gpu/bin/python}"
CPU_PY="${MCMIPL_CPU_PYTHON:-/root/miniconda3/envs/mcmipl-reproduce/bin/python}"
# Ada GPU：优先用 scripts/setup_gpu_baseline_background.sh 安装的 PyTorch2+cu121；老环境仅 CPU 或旧卡。
# 若希望在每次 GPU 训练前自动体检/修复磁盘与 PT↔DGL↔numpy：export MCMIPL_PREFLIGHT_AUTOFIX=1
if [[ -n "${MCMIPL_FORCE_CPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  PYTHON="$CPU_PY"
elif [[ -x "$GPU_PY" ]]; then
  PYTHON="$GPU_PY"
else
  PYTHON="$CPU_PY"
fi

if [[ "${MCMIPL_PREFLIGHT_AUTOFIX:-}" == "1" ]] && [[ "$PYTHON" == "$GPU_PY" ]] && [[ -x "$PYTHON" ]]; then
  _AF="$_RUN_ROOT/scripts/mcmipl_gpu_env_autofix.sh"
  if [[ -f "$_AF" ]]; then
    bash "$_AF" || echo "[run_mcmipl] WARN: autofix 失败或非零退出，继续训练" >&2
  fi
fi

# pip 安装的 NVIDIA/CUDA 轮子位于 site-packages/nvidia/*/lib；不设 LD_LIBRARY_PATH 时 DGL GraphBolt 可能找不到 libnvrtc.so.12。
if [[ "$PYTHON" == "$GPU_PY" ]] && [[ -z "${MCMIPL_FORCE_CPU:-}" ]]; then
  _SITE=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null) || true
  if [[ -n "$_SITE" ]]; then
    _NV=""
    for _d in "$_SITE"/nvidia/*/lib; do
      [[ -d "$_d" ]] || continue
      _NV="${_NV:+${_NV}:}${_d}"
    done
    if [[ -n "$_NV" ]]; then
      export LD_LIBRARY_PATH="${_NV}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  fi
fi
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${DATASET}_s${SEED}.log"

{
  echo "=== MCMIPL | $DATASET | seed=$SEED | steps=$MAX_STEPS | $(date) ==="
  echo "=== PYTHON: $PYTHON ==="
  if [[ -n "${MCMIPL_FORCE_CPU:-}" ]]; then
    echo "=== MCMIPL_FORCE_CPU=1 (CUDA hidden, CPU training — slow) ==="
  fi
} | tee "$LOG_FILE"

cd "$MCMIPL_DIR" || exit 1
$PYTHON -u RL_model.py \
  --data_name "$DATASET" \
  --embed transe \
  --seed "$SEED" \
  --gpu 0 \
  --max_steps "$MAX_STEPS" \
  --sample_times "$SAMPLE_TIMES" \
  --attr_num 20 \
  --choice_num 4 \
  --max_turn 15 \
  --eval_num "$EVAL_NUM" \
  --save_num "$MAX_STEPS" \
  2>&1 | tee -a "$LOG_FILE"

echo "=== DONE: $DATASET seed=$SEED at $(date) ===" | tee -a "$LOG_FILE"
