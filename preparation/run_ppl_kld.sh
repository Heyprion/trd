#!/usr/bin/env bash
# Run from any directory. --dry-run checks all 512 inputs without scoring/CUDA/SwanLab.
set -euo pipefail
PPL_KLD_ENTRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PPL_KLD_CONFIG:-$PPL_KLD_ENTRY_DIR/ppl_kld/config.env.sh}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$PPL_KLD_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$PPL_KLD_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi
export PYTHONPATH="$PPL_KLD_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OTEL_SDK_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$PYTHON_BIN" "$PPL_KLD_ENTRY_DIR/ppl_kld/evaluate.py" "$@"
