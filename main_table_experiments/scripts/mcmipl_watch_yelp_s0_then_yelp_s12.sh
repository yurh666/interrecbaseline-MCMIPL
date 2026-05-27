#!/usr/bin/env bash
# Wait for YELP s0 (mcmipl_phase_b_cpu) → archive s0 → stop old screen → serial YELP s1 → s2.
# Does NOT start BOOK/LAST_FM/MOVIE (those run on another server via rl_supplement).
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MAIN}/logs/train_YELP_STAR_s0.log"
ARCH_ROOT="${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives"
HANDOFF_FLAG="${ARCH_ROOT}/.yelp_s0_yelp_s12_handoff_done"
POLL_SEC="${POLL_SEC:-120}"
YELP_SCREEN="${MCMIPL_YELP_SCREEN:-mcmipl_lane_YELP_STAR}"

echo "=== watch: YELP s0 → archive → YELP s1,s2 | poll=${POLL_SEC}s | $(date) ==="

if [[ -f "$HANDOFF_FLAG" ]]; then
  echo "[handoff] already done: $(cat "$HANDOFF_FLAG")"
  exit 0
fi

while true; do
  if [[ -f "$LOG" ]] && grep -q "=== DONE: YELP_STAR seed=0" "$LOG"; then
    echo "[handoff] YELP s0 DONE at $(date)"
    break
  fi
  echo "[handoff] waiting YELP s0... $(date) (lines: $(wc -l < "$LOG" 2>/dev/null || echo 0))"
  sleep "$POLL_SEC"
done

export MCMIPL_RUN_ID="yelp_star_s0_handoff_$(date +%Y%m%d_%H%M%S)"
bash "${MAIN}/scripts/mcmipl_record_run_timing.sh" end YELP_STAR 0 "$MCMIPL_RUN_ID" "phase_b_cpu"
bash "${MAIN}/scripts/mcmipl_archive_seed.sh" YELP_STAR 0
bash "${MAIN}/scripts/mcmipl_archive_phase_a.sh" YELP_STAR

for scr in mcmipl_phase_b_cpu mcmipl_phase_b_serial mcmipl_lane_3ds; do
  screen -ls | grep -qE "[0-9]+\.${scr}[[:space:]]" && screen -S "$scr" -X quit || true
done
sleep 2
pkill -f "run_phase_b_resume_lastfm_yelp.sh" 2>/dev/null || true
pkill -f "RL_model.py --data_name YELP_STAR --embed transe --seed 0" 2>/dev/null || true
sleep 2

if screen -ls | grep -qE "[0-9]+\.${YELP_SCREEN}[[:space:]]"; then
  echo "[handoff] ${YELP_SCREEN} already running"
else
  SCR_LOG="${MAIN}/logs/screen_${YELP_SCREEN}_$(date +%Y%m%d_%H%M%S).log"
  screen -dmS "${YELP_SCREEN}" bash -lc \
    "bash ${MAIN}/scripts/mcmipl_lane_yelp_serial_all_seeds.sh 1 2 2>&1 | tee -a ${SCR_LOG}"
  echo "[handoff] started ${YELP_SCREEN} seeds 1,2"
fi

date -Iseconds > "$HANDOFF_FLAG"
echo "[handoff] done -> $HANDOFF_FLAG"
