#!/usr/bin/env bash
# MCMIPL GPU 环境「一键体检 + 自动修」：磁盘、TMPDIR、numpy、PyTorch↔DGL GraphBolt、日志备份。
#
# 用法：
#   bash scripts/mcmipl_gpu_env_autofix.sh
#   bash scripts/mcmipl_gpu_env_autofix.sh --dry-run
#   bash scripts/mcmipl_gpu_env_autofix.sh --backup-logs BOOK
#   bash scripts/mcmipl_gpu_env_autofix.sh --print-exports    # 只打印 export TMPDIR / PIP_CACHE_DIR
#   eval "$(bash scripts/mcmipl_gpu_env_autofix.sh --print-ld)"   # 打印 export LD_LIBRARY_PATH（GPU pip nvidia 库）
#
# 建议在 tmux / nohup 训练前先跑一次。
# 可写「大数据盘」：\$MCMIPL_LARGE_TMP → /root/autodl-tmp → \$MCMIPL_LARGE_TMP_FALLBACK（自动探测可写）。
# /autodl-pub/data 常为只读共享数据；pip/conda 缓存请用 autodl-tmp 或可写挂载，不要用 pub 根目录写缓存。
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
LOG_ROOT="$EXP_ROOT/logs"

ROOT_CONDA="${MCMIPL_CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${MCMIPL_GPU_ENV:-mcmipl-baseline-gpu}"
PY="$ROOT_CONDA/envs/$ENV_NAME/bin/python"
MAMBA="$ROOT_CONDA/bin/micromamba"

DRY_RUN=0
BACKUP_DATASET=""
PRINT_EXPORTS=0
PRINT_LD=0

WARN_MB=5120
SOFT_CLEAN_MB=3072
HARD_CLEAN_MB=1024

TORCH_PIN_VERSION="${MCMIPL_TORCH_PIN:-2.2.1}"
TV_PIN="${MCMIPL_TORCHVISION_PIN:-0.17.1}"
TA_PIN="${MCMIPL_TORCHAUDIO_PIN:-2.2.1}"

pick_large_tmp() {
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

usage() {
  sed -n '1,15p' "$0" | tail -n +2
}

log() { echo "[mcmipl-autofix] $*"; }
die() { echo "[mcmipl-autofix] ERROR: $*" >&2; exit 1; }

pip_i() { "$PY" -m pip "$@"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) DRY_RUN=1 ;;
    --backup-logs)
      BACKUP_DATASET="${2:?需要数据集名，如 BOOK}"
      shift
      ;;
    --print-exports) PRINT_EXPORTS=1 ;;
    --print-ld) PRINT_LD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) log "未知参数: $1"; usage; exit 2 ;;
  esac
  shift
done

if LARGE_TMP="$(pick_large_tmp)"; then
  PIP_TMP="$LARGE_TMP/pip-tmp"
  PIP_CACHE="$LARGE_TMP/pip-cache"
else
  LARGE_TMP=""
  PIP_TMP=""
  PIP_CACHE=""
fi

nv_ld_export_line() {
  [[ -x "$PY" ]] || return 1
  local site _NV d
  site="$("$PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)" || return 1
  _NV=""
  for d in "$site"/nvidia/*/lib; do
    [[ -d "$d" ]] || continue
    _NV="${_NV:+${_NV}:}${d}"
  done
  [[ -n "$_NV" ]] || return 1
  # shellcheck disable=SC2016 # 故意保留字面量 ${LD_LIBRARY_PATH...} 供 eval
  echo "export LD_LIBRARY_PATH=\"${_NV}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
}

if [[ "$PRINT_EXPORTS" -eq 1 ]]; then
  if [[ -z "$LARGE_TMP" ]]; then
    echo "[mcmipl-autofix] 无可写大数据盘。请设置：export MCMIPL_LARGE_TMP=/path/to/scratch" >&2
    exit 1
  fi
  mkdir -p "$PIP_TMP" "$PIP_CACHE"
  _CP="$LARGE_TMP/conda-pkgs"
  echo "export TMPDIR=$PIP_TMP"
  echo "export PIP_CACHE_DIR=$PIP_CACHE"
  echo "export CONDA_PKGS_DIRS=$_CP"
  exit 0
fi

if [[ "$PRINT_LD" -eq 1 ]]; then
  nv_ld_export_line || exit 1
  exit 0
fi

ensure_large_tmp() {
  if [[ -z "$LARGE_TMP" ]]; then
    log "未检测到可写大数据盘（\$MCMIPL_LARGE_TMP、/root/autodl-tmp、\$MCMIPL_LARGE_TMP_FALLBACK）。pip/解压易占满根分区。"
    return 0
  fi
  mkdir -p "$PIP_TMP" "$PIP_CACHE"
  log "使用大数据盘 scratch: $LARGE_TMP（TMPDIR / pip cache / conda pkgs）"
  echo "export TMPDIR=$PIP_TMP"
  echo "export PIP_CACHE_DIR=$PIP_CACHE"
  export TMPDIR="$PIP_TMP"
  export PIP_CACHE_DIR="$PIP_CACHE"
  if [[ -z "${CONDA_PKGS_DIRS:-}" ]]; then
    export CONDA_PKGS_DIRS="$LARGE_TMP/conda-pkgs"
    mkdir -p "$CONDA_PKGS_DIRS"
    echo "export CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
  fi
}

avail_mb_root() {
  df -Pm / 2>/dev/null | awk 'NR==2 {print $4}'
}

disk_fix() {
  local avail
  avail="$(avail_mb_root)"
  log "根分区剩余约 ${avail}MiB"
  if [[ "${avail:-0}" -ge "$WARN_MB" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "[dry-run] 将清理 pip cache / micromamba tarball，必要时 clean -a"
    return 0
  fi
  log "磁盘偏紧，清理缓存…"
  pip_i cache purge 2>/dev/null || true
  if [[ -x "$MAMBA" ]]; then
    "$MAMBA" clean -t -y 2>/dev/null || true
  fi
  avail="$(avail_mb_root)"
  if [[ "${avail:-0}" -lt "$SOFT_CLEAN_MB" ]] && [[ -x "$MAMBA" ]]; then
    log "仍低于 ${SOFT_CLEAN_MB}MiB → micromamba clean -a"
    "$MAMBA" clean -a -y 2>/dev/null || true
  fi
  avail="$(avail_mb_root)"
  log "清理后根分区剩余约 ${avail}MiB"
  if [[ "${avail:-0}" -lt "$HARD_CLEAN_MB" ]]; then
    echo "[mcmipl-autofix] WARN: 根分区仍 < ${HARD_CLEAN_MB}MiB，请手动删文件或扩容。" >&2
  fi
}

apply_ld_path_from_py() {
  if [[ ! -x "$PY" ]]; then
    log "跳过 LD_LIBRARY_PATH：未找到 $PY"
    return 0
  fi
  local site _NV d
  site="$("$PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)" || return 0
  _NV=""
  for d in "$site"/nvidia/*/lib; do
    [[ -d "$d" ]] || continue
    _NV="${_NV:+${_NV}:}${d}"
  done
  if [[ -n "$_NV" ]]; then
    export LD_LIBRARY_PATH="${_NV}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

numpy_fix() {
  [[ -x "$PY" ]] || return 0
  apply_ld_path_from_py
  if "$PY" -c "import numpy; numpy.__version__" 2>/dev/null; then
    log "numpy 可导入"
    return 0
  fi
  log "numpy 异常（含 METADATA 损坏），强制重装 numpy<2 …"
  if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
  pip_i install --no-cache-dir --force-reinstall "numpy>=1.23,<2" || log "numpy 重装失败"
}

torchdata_compat_fix() {
  [[ -x "$PY" ]] || return 0
  apply_ld_path_from_py
  if "$PY" -c "import dgl" 2>/dev/null; then
    return 0
  fi
  local err
  err="$("$PY" -c "import dgl" 2>&1)" || true
  if [[ "$err" == *"torchdata.datapipes"* ]] || [[ "$err" == *"No module named 'torchdata"* ]]; then
    log "安装 torchdata==0.7.1（兼容 DGL datapipes）…"
    if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
    pip_i install --no-cache-dir "torchdata==0.7.1" -i https://pypi.tuna.tsinghua.edu.cn/simple || true
  fi
}

torch_dgl_pin_fix() {
  [[ -x "$PY" ]] || die "GPU 环境 Python 不存在: $PY"
  apply_ld_path_from_py

  local probe
  probe="$(
    "$PY" 2>&1 <<'PY'
import glob
import os
import site
import sys

def main():
    try:
        import torch
    except Exception as e:
        print("NO_TORCH", e)
        return 2
    tv = torch.__version__.split("+")[0].strip()
    sp = site.getsitepackages()[0]
    gb = os.path.join(sp, "dgl", "graphbolt")
    libs = sorted(glob.glob(os.path.join(gb, "libgraphbolt_pytorch_*.so")))
    suffixes = []
    for p in libs:
        base = os.path.basename(p)
        pref = "libgraphbolt_pytorch_"
        suf = base[len(pref) : -len(".so")]
        suffixes.append(suf)
    need = os.path.join(gb, f"libgraphbolt_pytorch_{tv}.so")
    ok_file = os.path.isfile(need)
    print("TORCH_VERSION", torch.__version__)
    print("GRAPHBOLT_LIB_OK", int(ok_file))
    print("GRAPHBOLT_SUFFIXES", ",".join(suffixes))
    try:
        import dgl  # noqa: F401

        print("DGL_IMPORT_OK", 1)
    except Exception as e:
        print("DGL_IMPORT_OK", 0)
        print("DGL_IMPORT_ERR", repr(e))
    return 0


sys.exit(main())
PY
  )" || true
  if echo "$probe" | grep -q '^NO_TORCH'; then
    log "安装 PyTorch cu121（pin ${TORCH_PIN_VERSION}）…"
    if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
    pip_i install --no-cache-dir "torch==$TORCH_PIN_VERSION" "torchvision==$TV_PIN" "torchaudio==$TA_PIN" \
      --index-url https://download.pytorch.org/whl/cu121 || log "torch 安装失败"
    return 0
  fi

  local ok suf_line
  ok="$(echo "$probe" | awk '/^GRAPHBOLT_LIB_OK /{print $2}')"
  suf_line="$(echo "$probe" | awk '/^GRAPHBOLT_SUFFIXES /{$1=""; print substr($0,2)}')"
  if [[ "$ok" == "1" ]]; then
    log "PyTorch 与 DGL GraphBolt .so 一致（可选后缀: $suf_line）"
  else
    log "GraphBolt 缺少当前 torch 对应 .so（已有: $suf_line）→ 降级 torch==${TORCH_PIN_VERSION}+cu121"
    if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
    pip_i install --no-cache-dir "torch==$TORCH_PIN_VERSION" "torchvision==$TV_PIN" "torchaudio==$TA_PIN" \
      --index-url https://download.pytorch.org/whl/cu121 || log "torch 降级失败"
  fi

  apply_ld_path_from_py
  torchdata_compat_fix

  if ! "$PY" -c "import dgl; print('dgl', dgl.__version__)" 2>/dev/null; then
    log "DGL 仍失败：手动执行 — eval \"\$(bash $HERE/mcmipl_gpu_env_autofix.sh --print-ld)\""
  else
    log "DGL 导入 OK"
  fi
}

backup_logs() {
  local ds="$1"
  [[ -n "$ds" ]] || return 0
  mkdir -p "$LOG_ROOT/archive"
  local stamp arch n=0 f
  stamp="$(date +%Y%m%d_%H%M%S)"
  arch="$LOG_ROOT/archive/${stamp}_${ds}"
  mkdir -p "$arch"
  shopt -s nullglob
  for f in "$LOG_ROOT/train_${ds}_s"*.log; do
    cp -a "$f" "$arch/" && n=$((n + 1)) || true
  done
  shopt -u nullglob
  log "已备份 ${ds} 日志 ${n} 个 → $arch"
}

ensure_large_tmp
disk_fix
numpy_fix
torch_dgl_pin_fix

if [[ -n "$BACKUP_DATASET" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "[dry-run] 将备份: $BACKUP_DATASET"
  else
    backup_logs "$BACKUP_DATASET"
  fi
fi

log "结束。训练前可选备份：bash $0 --backup-logs BOOK"
log "当前会话大临时目录（已在脚本内 export）：TMPDIR=$TMPDIR PIP_CACHE_DIR=${PIP_CACHE_DIR:-}"
