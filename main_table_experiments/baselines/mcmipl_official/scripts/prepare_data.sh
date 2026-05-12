#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${BASELINE_DIR}/MCMIPL"
LOG_DIR="${BASELINE_DIR}/results/raw_logs"
DATA_NAME="${1:-LAST_FM_STAR}"

mkdir -p "${LOG_DIR}"

echo "Preparing official MCMIPL data for ${DATA_NAME}" | tee "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
echo "Repo dir: ${REPO_DIR}" | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"

case "${DATA_NAME}" in
  LAST_FM_STAR)
    DATA_DIR="${REPO_DIR}/data/lastfm_star"
    ;;
  YELP_STAR)
    DATA_DIR="${REPO_DIR}/data/yelp_star"
    ;;
  BOOK)
    DATA_DIR="${REPO_DIR}/data/book"
    ;;
  MOVIE)
    DATA_DIR="${REPO_DIR}/data/movie"
    ;;
  *)
    echo "Unknown data_name: ${DATA_NAME}" | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
    exit 2
    ;;
esac

if [ ! -d "${DATA_DIR}" ]; then
  echo "Missing official data directory: ${DATA_DIR}" | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
  echo "Download official released data and place it under MCMIPL/data/<data_name> before training." | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
  exit 1
fi

echo "Found data directory: ${DATA_DIR}" | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
echo "Data files:" | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
python - <<PY | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
from pathlib import Path
data_dir = Path("${DATA_DIR}")
for path in sorted(data_dir.rglob("*")):
    if path.is_file():
        print(path.relative_to(data_dir))
PY

# Optional: same order as InterRec raw → preprocess (export interactions.csv + items.csv).
# Unset by default; baseline training still uses MCMIPL pickles as before.
#   export EXPORT_INTERREC_CSV=1
#   export INTERREC_REPO=/home/yurh/interrec
#   export INTERREC_CSV_OUT=/home/yurh/interrec/data/raw/mcmipl_lastfm_star  # 按数据集换名，勿用 ${DATA_NAME,,}（会变成 last_fm_star）
if [ "${EXPORT_INTERREC_CSV:-0}" = "1" ]; then
  INTERREC_REPO="${INTERREC_REPO:-/home/yurh/interrec}"
  : "${INTERREC_CSV_OUT:?Set INTERREC_CSV_OUT to the InterRec data/raw/... target dir}"
  case "${DATA_NAME}" in
    LAST_FM_STAR) _DS=lastfm_star ;;
    YELP_STAR) _DS=yelp_star ;;
    BOOK) _DS=book ;;
    MOVIE) _DS=movie ;;
    *) echo "EXPORT_INTERREC_CSV: unknown DATA_NAME ${DATA_NAME}"; exit 2 ;;
  esac
  echo "Exporting InterRec-compatible CSV to ${INTERREC_CSV_OUT} (dataset=${_DS})" \
    | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
  python "${INTERREC_REPO}/scripts/convert_mcmipl_to_interrec_csv.py" \
    --mcmipl-dir "${REPO_DIR}" \
    --dataset "${_DS}" \
    --out "${INTERREC_CSV_OUT}" \
    | tee -a "${LOG_DIR}/prepare_data_${DATA_NAME}.log"
fi
