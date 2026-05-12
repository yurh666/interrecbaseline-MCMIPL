#!/usr/bin/env bash
# 将 InterRec 的 sessions.json 写成 MCMIPL BOOK 或 MOVIE 的 review_dict + UI_data pkl（时序对齐）。
# 仅支持 BOOK / MOVIE。用法见 interrec/scripts/export_interrec_sessions_to_mcmipl_book_movie.py
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <book|movie> <sessions.json> <MCMIPL_data_dir>" >&2
  echo "Example: $0 book /home/yurh/interrec/data/processed/sessions_book.json /home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL/data/book" >&2
  exit 2
fi

DS="$1"
SESSIONS="$2"
MCMIPL_DATA="$3"
INTERREC_REPO="${INTERREC_REPO:-/home/yurh/interrec}"

python "${INTERREC_REPO}/scripts/export_interrec_sessions_to_mcmipl_book_movie.py" \
  --dataset "${DS}" \
  --sessions "${SESSIONS}" \
  --dataset-data-dir "${MCMIPL_DATA}"
