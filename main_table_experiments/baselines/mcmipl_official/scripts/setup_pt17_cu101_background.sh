#!/usr/bin/env bash
# 后台安装：PyTorch 1.7.1+cu101 + DGL cu101 + baseline pip 依赖
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${BASE}/results/raw_logs/setup_pt17_cu101_background.log"
mkdir -p "$(dirname "$LOG")"
exec >>"${LOG}" 2>&1
echo "===== $(date -Iseconds) setup_pt17_cu101_background start ====="

ENV=mcmipl-pt17-cu101
ROOT=/root/miniconda3
MAMBA="${ROOT}/bin/micromamba"
YML="${BASE}/environment_pt17_cu101.yml"

if [[ ! -x "${MAMBA}" ]]; then
  echo "ERROR: micromamba not found at ${MAMBA}"
  exit 1
fi

"${MAMBA}" env remove -n "${ENV}" -r "${ROOT}" -y 2>/dev/null || true
"${MAMBA}" env create -y -r "${ROOT}" -n "${ENV}" -f "${YML}"

PY="${ROOT}/envs/${ENV}/bin/python"
PIP="${ROOT}/envs/${ENV}/bin/pip"

"${PIP}" install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
"${PIP}" install --no-cache-dir pyyaml easydict ipdb -i https://pypi.tuna.tsinghua.edu.cn/simple

# 与先前在本机验证可行的组合一致：pip 官方 cu101 torch wheel（避免 conda MKL/iJIT 问题）
"${PIP}" install --no-cache-dir \
  "torch==1.7.1+cu101" \
  -f https://download.pytorch.org/whl/torch_stable.html \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

"${PIP}" uninstall -y dgl dgl-cu101 2>/dev/null || true
"${PIP}" install --no-cache-dir "dgl-cu101==0.6.0" -f https://data.dgl.ai/wheels/repo.html

echo "--- smoke ---"
"${PY}" - <<'PY'
import torch
import dgl
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        torch.zeros(2, device="cuda").mul_(2)
        print("cuda_kernel_smoke_ok")
    except Exception as e:
        print("cuda_kernel_smoke_failed", repr(e))
print("dgl", dgl.__version__)
PY

echo "===== $(date -Iseconds) setup_pt17_cu101_background DONE ====="
echo "RTX 40 系：若 cuda_kernel_smoke_failed / no kernel image，属预期；请改用 PyTorch≥2 + CUDA11/12 新环境才能在 Ada 上用 GPU。"
