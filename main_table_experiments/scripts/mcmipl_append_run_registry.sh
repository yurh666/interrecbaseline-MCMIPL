#!/usr/bin/env bash
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH_ROOT="${MCMIPL_ARCHIVE_ROOT:-${MAIN}/../experiments/mcmipl_interrec_protocol_eval/archives}"
REG="${ARCH_ROOT}/run_registry.jsonl"

DATASET="$1"
SEED="$2"
RUN_ID="$3"
DEST="$4"
STATUS="${5:-archived}"

mkdir -p "$ARCH_ROOT"
printf '{"ts":"%s","dataset":"%s","seed":%s,"run_id":"%s","archive_dir":"%s","status":"%s"}\n' \
  "$(date -Iseconds)" "$DATASET" "$SEED" "$RUN_ID" "$DEST" "$STATUS" >> "$REG"
