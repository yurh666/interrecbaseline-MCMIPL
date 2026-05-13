#!/usr/bin/env bash
# 转发到主入口：main_table_experiments/run_pipeline_phase_cpu.sh
# 若你从 baselines/mcmipl_official/scripts 里找 Phase A，用这个即可。
set -euo pipefail
_MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${_MAIN}/run_pipeline_phase_cpu.sh" "$@"
