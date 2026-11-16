#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/training/config.env.sh"
export SEQUENCE_FORMAT=chat
# Keep the earlier run; use reserves verified for this single-save profile.
export OUTPUT_SPACE_PROFILE=balanced-single-save
exec "$PYTHON_BIN" "$MATH_RUN_ROOT/preparation/training/pipeline.py" "${REPAIR_STAGE:-train}" \
  --reuse-train-run "${REPAIR_CACHE_SOURCE:-$MATH_RUN_ROOT/preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json}" "$@"
