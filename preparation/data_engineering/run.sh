#!/usr/bin/env bash
set -euo pipefail
DATA_ENTRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DATA_ENGINEERING_CONFIG:-$DATA_ENTRY_DIR/config.env.sh}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OTEL_SDK_DISABLED=true
DATA_STAGE=all
case "${1:-}" in
  all|clean|label) DATA_STAGE="$1"; shift ;;
esac
case "$DATA_STAGE" in
  all) DATA_ENTRY=pipeline.py ;;
  clean) DATA_ENTRY=clean.py ;;
  label) DATA_ENTRY=stratify.py ;;
esac
exec "$PYTHON_BIN" "$DATA_ENTRY_DIR/$DATA_ENTRY" "$@"
