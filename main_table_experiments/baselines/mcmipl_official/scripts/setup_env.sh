#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASELINE_DIR}/results/raw_logs"
ENV_FILE="${BASELINE_DIR}/environment.yml"
ENV_NAME="mcmipl-reproduce"
export CONDARC="${BASELINE_DIR}/condarc"

mkdir -p "${LOG_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda or use Dockerfile." | tee "${LOG_DIR}/env_versions.txt"
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env create -f "${ENV_FILE}" 2>&1 | tee "${LOG_DIR}/conda_create.log"
fi

{
  echo "===== environment versions ====="
  conda run -n "${ENV_NAME}" python - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
try:
    import torch
    print("torch", torch.__version__)
except Exception as exc:
    print("torch_import_error", repr(exc))
try:
    import dgl
    print("dgl", dgl.__version__)
except Exception as exc:
    print("dgl_import_error", repr(exc))
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as exc:
    print("numpy_import_error", repr(exc))
PY
} | tee "${LOG_DIR}/env_versions.txt"
