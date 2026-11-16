#!/usr/bin/env bash
# Same AIME protocol as the completed distilled-model evaluation; no training.
set -euo pipefail
export MATH_EVAL_DATASETS=aime24,aime25
source "$(dirname "${BASH_SOURCE[0]}")/training/config.env.sh"
exec "$PYTHON_BIN" "$MATH_RUN_ROOT/preparation/training/base_evaluation.py" "$@"
