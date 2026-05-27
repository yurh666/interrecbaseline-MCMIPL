#!/usr/bin/env bash
# Yelp only: seed 0 → 1 → 2 serial, per-seed archive (fresh run from scratch).
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mcmipl_lane_sequential.sh" YELP_STAR 0 1 2
