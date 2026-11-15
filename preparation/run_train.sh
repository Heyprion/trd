#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/training/config.env.sh"
exec "$PYTHON_BIN" "$MATH_RUN_ROOT/preparation/training/pipeline.py" train "$@"
