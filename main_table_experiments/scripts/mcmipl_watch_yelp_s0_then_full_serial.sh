#!/usr/bin/env bash
# 1) Wait for YELP_STAR seed=0 to finish (running in mcmipl_phase_b_cpu).
# 2) Archive s0 + record timing.
# 3) Stop old phase_b_cpu screen (avoid re-entering YELP s0 in old resume script).
# 4) Launch ONE screen running full serial: BOOK→LAST_FM→MOVIE→YELP s1,s2.
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MAIN}/logs/train_YELP_STAR_s0.log"
ARCH_ROOT="${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives"
HANDOFF_FLAG="${ARCH_ROOT}/.yelp_s0_full_serial_handoff_done"
POLL_SEC="${POLL_SEC:-120}"
SERIAL_SCREEN="${MCMIPL_SERIAL_SCREEN:-mcmipl_phase_b_serial}"

echo "=== watch: YELP s0 → full serial pipeline | poll=${POLL_SEC}s | $(date) ==="
echo "Log: ${LOG}"
echo "After handoff screen: ${SERIAL_SCREEN}"

if [[ -f "$HANDOFF_FLAG" ]]; then
  echo "[handoff] already done: $(cat "$HANDOFF_FLAG")"
  exit 0
fi

while true; do
  if [[ -f "$LOG" ]] && grep -q "=== DONE: YELP_STAR seed=0" "$LOG"; then
    echo "[handoff] YELP s0 DONE at $(date)"
    break
  fi
  echo "[handoff] waiting for YELP s0... $(date) (log lines: $(wc -l < "$LOG" 2>/dev/null || echo 0))"
  sleep "$POLL_SEC"
done

export MCMIPL_RUN_ID="yelp_star_s0_handoff_$(date +%Y%m%d_%H%M%S)"
bash "${MAIN}/scripts/mcmipl_record_run_timing.sh" end YELP_STAR 0 "$MCMIPL_RUN_ID" "yelp_s0_completed_in_phase_b_cpu"
bash "${MAIN}/scripts/mcmipl_archive_seed.sh" YELP_STAR 0
bash "${MAIN}/scripts/mcmipl_archive_phase_a.sh" YELP_STAR

# Append to serial schedule doc (s0 block)
SCHEDULE_MD="${ARCH_ROOT}/PHASE_B_SERIAL_SCHEDULE.md"
{
  echo "# Phase B 全串行计划"
  echo ""
  echo "## YELP s0（handoff 前已在 mcmipl_phase_b_cpu 完成）"
  echo ""
  echo "| 序号 | 数据集 | seed | 状态 | 备注 |"
  echo "|------|--------|------|------|------|"
  echo "| 0 | YELP_STAR | 0 | done (pre-pipeline) | archived at handoff $(date -Iseconds) |"
  echo ""
  echo "## 后续队列（screen \`${SERIAL_SCREEN}\`）"
  echo ""
} > "$SCHEDULE_MD"

# Stop parallel / old screens that would compete for CPU
for scr in mcmipl_phase_b_cpu mcmipl_lane_3ds mcmipl_lane_BOOK mcmipl_lane_LAST_FM_STAR mcmipl_lane_MOVIE mcmipl_lane_YELP_STAR; do
  if screen -ls | grep -qE "[0-9]+\.${scr}[[:space:]]"; then
    echo "[handoff] stopping screen ${scr}"
    screen -S "$scr" -X quit || true
  fi
done
sleep 3
pkill -f "run_phase_b_resume_lastfm_yelp.sh" 2>/dev/null || true
pkill -f "mcmipl_lane_three_datasets_serial" 2>/dev/null || true
# Do not kill YELP s0 python — should have exited on DONE; clean orphans only
pkill -f "RL_model.py --data_name YELP_STAR --embed transe --seed 0" 2>/dev/null || true
sleep 2

if screen -ls | grep -qE "[0-9]+\.${SERIAL_SCREEN}[[:space:]]"; then
  echo "[handoff] ${SERIAL_SCREEN} already running"
else
  SCR_LOG="${MAIN}/logs/screen_${SERIAL_SCREEN}_$(date +%Y%m%d_%H%M%S).log"
  screen -dmS "${SERIAL_SCREEN}" bash -lc \
    "bash ${MAIN}/scripts/mcmipl_phase_b_full_serial.sh 2>&1 | tee -a ${SCR_LOG}"
  echo "[handoff] launched ${SERIAL_SCREEN}"
  echo "          log: ${SCR_LOG}"
  echo "          attach: screen -r ${SERIAL_SCREEN}"
fi

date -Iseconds > "$HANDOFF_FLAG"
echo "[handoff] complete -> ${HANDOFF_FLAG}"
