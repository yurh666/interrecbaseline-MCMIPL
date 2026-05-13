#!/usr/bin/env bash
# BOOK seed 0/1/2 全部结束后，对「除 BOOK 外」三个数据集按顺序跑 **TransE/OpenKE 段 → Phase B（RL）**：
#   LAST_FM_STAR → YELP_STAR → MOVIE。
#
# Phase A 不在这里做（已由 CPU / graph_init 完成）。
# TransE：仓库 **无一键 OpenKE**；默认只检查 `MCMIPL/tmp/<slug>/embeds/transe.pkl`，缺则退出并提示按官方 README 训练。
# 若你自己封装了训练命令，可设 TRANSE_TRAIN_CMD，会在每个数据集跑 RL 前对应当前 slug 执行一次。
#
# 超参默认与 run_book_gpu_seeds_012.sh / Phase B 一致：
#   MAX_STEPS=50 SAMPLE_TIMES=100 EVAL_NUM=10 SEEDS=0 1 2
#
# 环境变量：
#   SKIP_WAIT=1                 不等待 BOOK（调试）
#   POLL_SEC=120                等待 BOOK 时的轮询间隔
#   TRANSE_TRAIN_CMD='...'      非空时：每个数据集在 Phase B 前执行
#                               `bash -c "$TRANSE_TRAIN_CMD"`，并 export：
#                               MCMIPL_TRANSE_DATA_NAME、MCMIPL_TRANSE_SLUG、MCMIPL_ROOT、MCMIPL_TRANSE_OUT
#   ALLOW_MISSING_TRANSE=1      未设 TRANSE_TRAIN_CMD 且缺 transe.pkl 时仍继续（RL 侧可能 WARN）
#   REQUIRE_TRANSE=0|1          传给每次调用的 `run_pipeline_phase_gpu.sh`
#   其余同 Phase B：MCMIPL_GPU_PYTHON、MAX_STEPS、SAMPLE_TIMES、EVAL_NUM、SEEDS
#
# 交接说明见：artifacts/PHASE_A_LASTFM_YELP_INTERREC_TEMPORAL_20260513.md（LAST_FM kg 更新后 TransE 一致性）。
#
set -euo pipefail

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_MAIN_EXP="${_REPO}/main_table_experiments"
MCMIPL="${MCMIPL:-${_MAIN_EXP}/baselines/mcmipl_official/MCMIPL}"
cd "${_MAIN_EXP}"

LOG_DIR="${MCMIPL_LOG_DIR:-${_MAIN_EXP}/logs}"
POLL_SEC="${POLL_SEC:-120}"
SKIP_WAIT="${SKIP_WAIT:-0}"
ALLOW_MISSING_TRANSE="${ALLOW_MISSING_TRANSE:-0}"

_data_to_slug() {
  case "$1" in
    LAST_FM_STAR) printf '%s' 'last_fm_star' ;;
    YELP_STAR) printf '%s' 'yelp_star' ;;
    BOOK) printf '%s' 'book' ;;
    MOVIE) printf '%s' 'movie' ;;
    *)
      echo "Unknown DATA_NAME: $1" >&2
      return 2
      ;;
  esac
}

_transe_pkl() {
  local slug
  slug="$(_data_to_slug "$1")" || return 2
  printf '%s' "${MCMIPL}/tmp/${slug}/embeds/transe.pkl"
}

_book_done() {
  local s="$1"
  local f="${LOG_DIR}/train_BOOK_s${s}.log"
  [[ -f "${f}" ]] || return 1
  grep -q "=== DONE: BOOK seed=${s} at" "${f}"
}

_wait_book_three_seeds() {
  echo "=== 等待 BOOK seed 0/1/2 在日志中出现 DONE：${LOG_DIR}/train_BOOK_s*.log ==="
  while true; do
    if _book_done 0 && _book_done 1 && _book_done 2; then
      echo "=== BOOK 三 seed 均已 DONE | $(date) ==="
      return 0
    fi
    for s in 0 1 2; do
      if _book_done "${s}"; then
        echo "  [$(date +%H:%M:%S)] BOOK seed=${s} OK"
      else
        echo "  [$(date +%H:%M:%S)] BOOK seed=${s} 尚未 DONE → ${LOG_DIR}/train_BOOK_s${s}.log"
      fi
    done
    echo "  ${POLL_SEC}s 后重试…"
    sleep "${POLL_SEC}"
  done
}

_trans_openke_step() {
  local d="$1"
  local slug tp embed_dir
  slug="$(_data_to_slug "$d")" || return 2
  tp="$(_transe_pkl "$d")"
  embed_dir="$(dirname "${tp}")"
  mkdir -p "${embed_dir}"

  echo ""
  echo "=== [TransE / OpenKE] 数据集=${d} | slug=${slug} ==="
  echo "=== 期望产物（与官方 MCMIPL README 一致）：${tp} ==="
  echo "=== 说明：本仓库不封装 OpenKE；请手工按 OpenKE + 官方说明训练，或设置 TRANSE_TRAIN_CMD 调用你的命令。==="

  if [[ -n "${TRANSE_TRAIN_CMD:-}" ]]; then
    echo "=== 执行 TRANSE_TRAIN_CMD（当前数据集 ${d}）==="
    export MCMIPL_ROOT="${MCMIPL}"
    export MCMIPL_TRANSE_DATA_NAME="${d}"
    export MCMIPL_TRANSE_SLUG="${slug}"
    export MCMIPL_TRANSE_OUT="${tp}"
    bash -c "${TRANSE_TRAIN_CMD}"
  fi

  if [[ ! -f "${tp}" ]]; then
    echo "[ERROR] 仍缺少 embedding: ${tp}" >&2
    echo "  - 按 baselines/mcmipl_official/MCMIPL/README.md 使用 OpenKE 生成 TransE，或" >&2
    echo "  - 设置 TRANSE_TRAIN_CMD 指向你的训练脚本/命令；或" >&2
    echo "  - 临时设置 ALLOW_MISSING_TRANSE=1 强行进入 RL（与「图/embed 一致性」要求可能冲突）。" >&2
    if [[ "${ALLOW_MISSING_TRANSE}" != "1" ]]; then
      return 1
    fi
    echo "[WARN] ALLOW_MISSING_TRANSE=1：继续 Phase B。" >&2
  else
    echo "=== OK：已存在 ${tp} ==="
  fi
}

if [[ "${SKIP_WAIT}" != "1" ]]; then
  _wait_book_three_seeds
else
  echo "=== SKIP_WAIT=1：不等待 BOOK ==="
fi

echo ""
echo "=== 队列：LAST_FM_STAR → YELP_STAR → MOVIE（各：TransE 检查/钩子 → Phase B）| $(date) ==="

for d in LAST_FM_STAR YELP_STAR MOVIE; do
  _trans_openke_step "${d}"
  echo ""
  echo "=== [Phase B / RL] 数据集=${d} | $(date) ==="
  REQUIRE_TRANSE="${REQUIRE_TRANSE:-0}" bash "${_MAIN_EXP}/run_pipeline_phase_gpu.sh" "${d}"
done

echo ""
echo "=== 队列完成 | $(date) ==="
