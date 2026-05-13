#!/usr/bin/env bash
# RTX 4080 等 Ada GPU：需 PyTorch 2.x + CUDA 12.x（无法用 PyTorch 1.7+cu101 在 GPU 上执行内核）。
# SSH 断开后仍可继续：日志见 results/raw_logs/setup_gpu_baseline_background.log
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${BASE}/results/raw_logs/setup_gpu_baseline_background.log"
mkdir -p "$(dirname "$LOG")"
exec >>"${LOG}" 2>&1

echo "===== $(date -Iseconds) setup_gpu_baseline_background START ====="
ENV=mcmipl-baseline-gpu
ROOT=/root/miniconda3
MAMBA="${ROOT}/bin/micromamba"

if [[ ! -x "${MAMBA}" ]]; then
  echo "ERROR: micromamba missing: ${MAMBA}"
  exit 1
fi

"${MAMBA}" env remove -n "${ENV}" -r "${ROOT}" -y 2>/dev/null || true

"${MAMBA}" create -y -r "${ROOT}" -n "${ENV}" \
  python=3.10 pip numpy scipy pandas tqdm networkx scikit-learn \
  pytorch torchvision torchaudio pytorch-cuda=12.1 \
  -c pytorch -c nvidia

PIP="${ROOT}/envs/${ENV}/bin/pip"
PY="${ROOT}/envs/${ENV}/bin/python"

"${PIP}" install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"${PIP}" install --no-cache-dir easydict ipdb pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
"${PIP}" install --no-cache-dir dgl -f https://data.dgl.ai/wheels/cu121/repo.html

echo "--- GPU smoke ---"
"${PY}" - <<'PY'
import torch
import dgl
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(4096, device="cuda")
    y = (x * 2).sum()
    print("cuda_matmul_smoke_ok", float(y))
print("dgl", dgl.__version__)
PY

echo "===== $(date -Iseconds) setup_gpu_baseline_background DONE ====="
echo "激活: conda activate ${ENV}"
echo "run_mcmipl.sh 会在未设置 MCMIPL_FORCE_CPU 且该环境存在时优先使用此 Python。"
