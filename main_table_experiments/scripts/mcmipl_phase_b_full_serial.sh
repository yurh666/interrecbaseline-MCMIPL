#!/usr/bin/env bash
# Full CPU Phase-B serial queue (single RL process at a time).
#
# Order (YELP s0 must already be DONE + archived before this script starts):
#   BOOK 0,1 → LAST_FM_STAR 0,1 → MOVIE 0,1 → YELP_STAR 1,2
#
# Skips any seed that already has a complete archive (see mcmipl_seed_archive_status.sh).
#
# Run in screen after YELP s0 handoff, e.g. screen mcmipl_phase_b_serial
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH_ROOT="${MCMIPL_ARCHIVE_ROOT:-${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives}"

export MCMIPL_FORCE_CPU="${MCMIPL_FORCE_CPU:-1}"
export MCMIPL_CPU_PYTHON="${MCMIPL_CPU_PYTHON:-/home/yrh666/venvs/mcmipl-cpu/bin/python}"
export MCMIPL_SAVE_NUM="${MCMIPL_SAVE_NUM:-10}"

MAX_STEPS="${MAX_STEPS:-50}"
SAMPLE_TIMES="${SAMPLE_TIMES:-100}"
EVAL_NUM="${EVAL_NUM:-10}"

PIPELINE_ID="full_serial_$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="${MAIN}/logs/phase_b_full_serial_${PIPELINE_ID}.log"
SCHEDULE_MD="${ARCH_ROOT}/PHASE_B_SERIAL_SCHEDULE.md"

exec > >(tee -a "$PIPELINE_LOG") 2>&1

QUEUE=(
  "BOOK 0"
  "BOOK 1"
  "LAST_FM_STAR 0"
  "LAST_FM_STAR 1"
  "MOVIE 0"
  "MOVIE 1"
  "YELP_STAR 1"
  "YELP_STAR 2"
)

{
  echo "# Phase B 全串行计划"
  echo ""
  echo "- **pipeline_id:** \`${PIPELINE_ID}\`"
  echo "- **启动时间:** $(date -Iseconds)"
  echo "- **说明:** YELP s0 已在 handoff 阶段完成并归档；本流水线从 BOOK 开始。"
  echo ""
  echo "| 序号 | 数据集 | seed | 状态 | 开始 | 结束 | run_id |"
  echo "|------|--------|------|------|------|------|--------|"
} > "$SCHEDULE_MD"

_idx=0
for line in "${QUEUE[@]}"; do
  read -r DATASET SEED <<< "$line"
  _idx=$((_idx + 1))
  # shellcheck source=/dev/null
  source "${MAIN}/scripts/mcmipl_data_slug.sh"
  SLUG="$(data_slug "$DATASET")"

  if bash "${MAIN}/scripts/mcmipl_seed_archive_status.sh" "$DATASET" "$SEED" >/dev/null 2>&1; then
    echo "[skip] #${_idx} ${DATASET} seed=${SEED} — already archived"
    echo "| ${_idx} | ${DATASET} | ${SEED} | skipped (archived) | — | — | — |" >> "$SCHEDULE_MD"
    continue
  fi

  export MCMIPL_RUN_ID="${SLUG}_s${SEED}_${PIPELINE_ID}"
  _start="$(date -Iseconds)"
  echo ""
  echo "=== [#${_idx}] ${DATASET} seed=${SEED} | run_id=${MCMIPL_RUN_ID} | start=${_start} ==="

  bash "${MAIN}/scripts/mcmipl_archive_phase_a.sh" "$DATASET" || true
  bash "${MAIN}/scripts/mcmipl_record_run_timing.sh" start "$DATASET" "$SEED" "$MCMIPL_RUN_ID" "pipeline=${PIPELINE_ID}"

  bash "${MAIN}/scripts/mcmipl_run_seed_with_archive.sh" \
    "$DATASET" "$SEED" "$MAX_STEPS" "$SAMPLE_TIMES" "$EVAL_NUM"

  _end="$(date -Iseconds)"
  bash "${MAIN}/scripts/mcmipl_record_run_timing.sh" end "$DATASET" "$SEED" "$MCMIPL_RUN_ID" "pipeline=${PIPELINE_ID}"

  echo "| ${_idx} | ${DATASET} | ${SEED} | done | ${_start} | ${_end} | \`${MCMIPL_RUN_ID}\` |" >> "$SCHEDULE_MD"
done

{
  echo ""
  echo "**流水线结束:** $(date -Iseconds)"
  echo ""
  echo "明细 CSV: \`archives/phase_b_run_timeline.csv\`"
  echo "各 seed: \`archives/checkpoints/<slug>/seed_<N>/TIMING.txt\` + \`MANIFEST.txt\`"
} >> "$SCHEDULE_MD"

echo ""
echo "=== FULL SERIAL PIPELINE COMPLETE | ${PIPELINE_ID} | $(date) ==="
