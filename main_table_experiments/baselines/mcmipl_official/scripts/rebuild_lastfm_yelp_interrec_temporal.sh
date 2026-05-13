#!/usr/bin/env bash
# 将 LAST_FM_STAR、YELP_STAR 的交互划分重建为与 BOOK/MOVIE 一致的 InterRec 时序协议
# （observed 40% + future 内 7:1:2），再写回 MCMIPL 并 graph_init。
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../../.." && pwd)"
INTERREC="${INTERREC:-${_REPO_ROOT}/interrec}"
BASELINE="${BASELINE:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
MCMIPL="${MCMIPL:-${BASELINE}/MCMIPL}"
MCMIPL_DATA="${MCMIPL}/data"

QUICK="${QUICK:-0}"
QUICK_N="${QUICK_N:-120}"

cd "${INTERREC}"
mkdir -p data/raw_mcmipl_lastfm_star data/raw_mcmipl_yelp_star

echo "== [1/6] MCMIPL lastfm_star / yelp_star -> InterRec CSV =="
python scripts/convert_mcmipl_to_interrec_csv.py \
  --mcmipl-dir "${MCMIPL}" \
  --dataset lastfm_star \
  --out data/raw_mcmipl_lastfm_star

python scripts/convert_mcmipl_to_interrec_csv.py \
  --mcmipl-dir "${MCMIPL}" \
  --dataset yelp_star \
  --out data/raw_mcmipl_yelp_star

echo "== [2/6] InterRec preprocess (observed 40% + future 7:1:2) =="
python scripts/preprocess_dataset.py --config configs/preprocess_mcmipl_lastfm_star.yaml
python scripts/preprocess_dataset.py --config configs/preprocess_mcmipl_yelp_star.yaml

LF_SESS="data/processed_mcmipl_lastfm_star/sessions.json"
Y_SESS="data/processed_mcmipl_yelp_star/sessions.json"

if [[ "${QUICK}" == "1" ]]; then
  echo "== [3/6] QUICK: slice sessions to first ${QUICK_N} users =="
  python scripts/slice_sessions_json.py --in-path "${LF_SESS}" --out-path data/processed_mcmipl_lastfm_star/sessions_smoke.json --max "${QUICK_N}"
  python scripts/slice_sessions_json.py --in-path "${Y_SESS}" --out-path data/processed_mcmipl_yelp_star/sessions_smoke.json --max "${QUICK_N}"
  LF_EXPORT="data/processed_mcmipl_lastfm_star/sessions_smoke.json"
  Y_EXPORT="data/processed_mcmipl_yelp_star/sessions_smoke.json"
else
  echo "== [3/6] full sessions (no slice) =="
  LF_EXPORT="${LF_SESS}"
  Y_EXPORT="${Y_SESS}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
BK="${MCMIPL_DATA}/_backup_official_split_${TS}"
echo "== [4/6] backup current lastfm_star + yelp_star UI -> ${BK} =="
mkdir -p "${BK}"
cp -a "${MCMIPL_DATA}/lastfm_star/UI_Interaction_data" "${BK}/lastfm_UI_Interaction_data" 2>/dev/null || true
cp -a "${MCMIPL_DATA}/lastfm_star/UI_data" "${BK}/lastfm_UI_data" 2>/dev/null || true
cp -a "${MCMIPL_DATA}/yelp_star/UI_Interaction_data" "${BK}/yelp_UI_Interaction_data" 2>/dev/null || true
cp -a "${MCMIPL_DATA}/yelp_star/UI_data" "${BK}/yelp_UI_data" 2>/dev/null || true

echo "== [5/6] InterRec sessions -> MCMIPL review_dict + train/test.pkl =="
python scripts/export_interrec_sessions_to_mcmipl_book_movie.py \
  --dataset lastfm_star \
  --sessions "${LF_EXPORT}" \
  --dataset-data-dir "${MCMIPL_DATA}/lastfm_star"

python scripts/export_interrec_sessions_to_mcmipl_book_movie.py \
  --dataset yelp_star \
  --sessions "${Y_EXPORT}" \
  --dataset-data-dir "${MCMIPL_DATA}/yelp_star"

echo "== [6/6] MCMIPL graph_init =="
cd "${MCMIPL}"
python graph_init.py --data_name LAST_FM_STAR
python graph_init.py --data_name YELP_STAR

echo ""
echo "=== Done. LAST_FM_STAR / YELP_STAR 已与 BOOK/MOVIE 使用同一 InterRec 时序切分协议。==="
echo "=== 下一步：重训 TransE（OpenKE），再跑 RL_model.py；备份目录：${BK} ==="
