#!/usr/bin/env bash
# 转发到主入口：main_table_experiments/run_pipeline_phase_rl_cpu.sh
set -euo pipefail
_MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${_MAIN}/run_pipeline_phase_rl_cpu.sh" "$@"
