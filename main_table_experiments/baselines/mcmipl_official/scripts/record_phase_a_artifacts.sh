#!/usr/bin/env bash
# 转发到：main_table_experiments/scripts/record_phase_a_artifacts.sh
set -euo pipefail
_MTE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${_MTE}/scripts/record_phase_a_artifacts.sh" "$@"
