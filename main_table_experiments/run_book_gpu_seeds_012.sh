#!/bin/bash
# BOOK（pipeline 中第三个数据集）GPU 串行跑 seed 0/1/2；日志见 logs/train_BOOK_s*.log 与本脚本 tee。
set -euo pipefail
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
exec >> logs/tmux_BOOK_gpu_runner.log 2>&1
echo "====== runner start $(date) ======"
bash scripts/mcmipl_gpu_env_autofix.sh || echo "[runner] WARN: autofix 退出非零，继续" >&2
for s in 0 1 2; do
  echo "====== BOOK seed=${s} $(date) ======"
  bash run_mcmipl.sh BOOK "${s}" 50 100 10
done
echo "====== BOOK seeds 0-2 all finished $(date) ======"
