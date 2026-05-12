#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${BASELINE_DIR}/MCMIPL"
LOG_DIR="${BASELINE_DIR}/results/raw_logs"
CKPT_DIR="${BASELINE_DIR}/results/checkpoints"
DATA_NAME="${1:-LAST_FM_STAR}"
SEED="${2:-0}"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}/${DATA_NAME}/${SEED}"

cd "${REPO_DIR}"

LOG_FILE="${LOG_DIR}/train_${DATA_NAME}_${SEED}.log"
{
  echo "===== train MCMIPL ====="
  echo "data_name=${DATA_NAME}"
  echo "seed=${SEED}"
  echo "cwd=$(pwd)"
  git rev-parse HEAD || true
  date
  python RL_model.py --data_name "${DATA_NAME}" --seed "${SEED}"
  date
} 2>&1 | tee "${LOG_FILE}"

python - <<PY
from pathlib import Path
import shutil

repo = Path("${REPO_DIR}")
dst = Path("${CKPT_DIR}") / "${DATA_NAME}" / "${SEED}"
tmp_map = {
    "LAST_FM_STAR": repo / "tmp" / "last_fm_star" / "RL-agent",
    "YELP_STAR": repo / "tmp" / "yelp_star" / "RL-agent",
    "BOOK": repo / "tmp" / "book" / "RL-agent",
    "MOVIE": repo / "tmp" / "movie" / "RL-agent",
}
src = tmp_map.get("${DATA_NAME}")
if src and src.exists():
    for file in src.glob("*.pkl"):
        shutil.copy2(file, dst / file.name)
PY
