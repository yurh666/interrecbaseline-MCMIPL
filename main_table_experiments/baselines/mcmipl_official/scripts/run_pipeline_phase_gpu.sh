#!/usr/bin/env bash
# 转发到主入口：main_table_experiments/run_pipeline_phase_gpu.sh
# 多数据集用法（由主脚本循环）：bash run_pipeline_phase_gpu.sh BOOK MOVIE
set -euo pipefail
_MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${_MAIN}/run_pipeline_phase_gpu.sh" "$@"
