#!/usr/bin/env bash
# rebuild MCMIPL BOOK / MOVIE interaction splits from InterRec temporal protocol, then rerun graph_init.
# 不是「只改一个 json」：覆盖 review_dict + UI_data pkl 后必须 graph_init；TransE/RL 需另跑。
set -euo pipefail

# 默认路径：假定本仓库布局为 <repo>/interrec + <repo>/main_table_experiments/baselines/mcmipl_official
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../../.." && pwd)"
INTERREC="${INTERREC:-${_REPO_ROOT}/interrec}"
BASELINE="${BASELINE:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
MCMIPL="${MCMIPL:-${BASELINE}/MCMIPL}"
MCMIPL_DATA="${MCMIPL}/data"

# QUICK=1: 仅取前 N 条 session 写入 MCMIPL，用于烟测（graph_init + 可选 env）
QUICK="${QUICK:-0}"
QUICK_N="${QUICK_N:-120}"

usage() {
  echo "Usage: QUICK=1 QUICK_N=80 $0   # smoke test"
  echo "       QUICK=0 $0              # full InterRec sessions -> MCMIPL"
}

cd "${INTERREC}"
mkdir -p data/raw_mcmipl_book data/raw_mcmipl_movie

echo "== [1/6] MCMIPL official merged dicts -> InterRec CSV (list order = time proxy) =="
python scripts/convert_mcmipl_to_interrec_csv.py \
  --mcmipl-dir "${MCMIPL}" \
  --dataset book \
  --out data/raw_mcmipl_book

python scripts/convert_mcmipl_to_interrec_csv.py \
  --mcmipl-dir "${MCMIPL}" \
  --dataset movie \
  --out data/raw_mcmipl_movie

echo "== [2/6] InterRec preprocess (observed 40% + future 7:1:2) =="
python scripts/preprocess_dataset.py --config configs/preprocess_mcmipl_book.yaml
python scripts/preprocess_dataset.py --config configs/preprocess_mcmipl_movie.yaml

BOOK_SESS="data/processed_mcmipl_book/sessions.json"
MOVIE_SESS="data/processed_mcmipl_movie/sessions.json"

if [[ "${QUICK}" == "1" ]]; then
  echo "== [3/6] QUICK: slice sessions to first ${QUICK_N} users =="
  python scripts/slice_sessions_json.py --in-path "${BOOK_SESS}" --out-path data/processed_mcmipl_book/sessions_smoke.json --max "${QUICK_N}"
  python scripts/slice_sessions_json.py --in-path "${MOVIE_SESS}" --out-path data/processed_mcmipl_movie/sessions_smoke.json --max "${QUICK_N}"
  BOOK_EXPORT="data/processed_mcmipl_book/sessions_smoke.json"
  MOVIE_EXPORT="data/processed_mcmipl_movie/sessions_smoke.json"
else
  echo "== [3/6] full sessions (no slice) =="
  BOOK_EXPORT="${BOOK_SESS}"
  MOVIE_EXPORT="${MOVIE_SESS}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
BK="${MCMIPL_DATA}/_backup_official_split_${TS}"
echo "== [4/6] backup current MCMIPL book+movie UI_Interaction_data + UI_data -> ${BK} =="
mkdir -p "${BK}"
cp -a "${MCMIPL_DATA}/book/UI_Interaction_data" "${BK}/book_UI_Interaction_data" 2>/dev/null || true
cp -a "${MCMIPL_DATA}/book/UI_data" "${BK}/book_UI_data" 2>/dev/null || true
cp -a "${MCMIPL_DATA}/movie/UI_Interaction_data" "${BK}/movie_UI_Interaction_data" 2>/dev/null || true
cp -a "${MCMIPL_DATA}/movie/UI_data" "${BK}/movie_UI_data" 2>/dev/null || true

echo "== [5/6] Export InterRec sessions -> MCMIPL review_dict + train/test.pkl =="
python scripts/export_interrec_sessions_to_mcmipl_book_movie.py \
  --dataset book \
  --sessions "${BOOK_EXPORT}" \
  --dataset-data-dir "${MCMIPL_DATA}/book"

python scripts/export_interrec_sessions_to_mcmipl_book_movie.py \
  --dataset movie \
  --sessions "${MOVIE_EXPORT}" \
  --dataset-data-dir "${MCMIPL_DATA}/movie"

echo "== [6/6] MCMIPL graph_init (重建 Graph_generate_data / user_dict 等与交互一致的图侧) =="
cd "${MCMIPL}"
python graph_init.py --data_name BOOK
python graph_init.py --data_name MOVIE

echo ""
echo "=== Done. InterRec-aligned temporal splits are now in MCMIPL data/book and data/movie under UI_Interaction_data + UI_data. ==="
echo "=== Item pickles fea_item/*.pkl were NOT overwritten (see export_interrec_sessions_to_mcmipl_book_movie.py header). ==="
echo "=== Next for full baseline: TransE/OpenKE embeddings -> RL_model.py (same as official pipeline). ==="
echo "=== Restore official split from: ${BK} ==="
