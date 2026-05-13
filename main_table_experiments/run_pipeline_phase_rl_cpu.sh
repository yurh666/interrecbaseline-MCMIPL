#!/usr/bin/env bash
# 与 run_pipeline_phase_gpu.sh 相同调度，但强制走 CPU Python（第二台无卡机器跑 RL 用）
#
# 用法：
#   bash run_pipeline_phase_rl_cpu.sh BOOK MOVIE
#   MCMIPL_CPU_PYTHON=/path/to/python bash run_pipeline_phase_rl_cpu.sh BOOK
#
set -euo pipefail
export MCMIPL_FORCE_CPU=1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pipeline_phase_gpu.sh" "$@"
