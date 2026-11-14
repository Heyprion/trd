#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VERL_ROOT"
VARIANT="${1:-forward_kl_y_r}"
case "$VARIANT" in
    forward_kl|forward_kl_y_r|forward_kl_clip|reverse_kl|reverse_kl_topk) ;;
    *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
# Use one run timestamp for output staging and for the upstream entrypoint.
export RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M%S)}"
# Load the original wrapper's experiment defaults without starting its entrypoint.
source <(sed '/^exec bash /d' "recipe/opd/run/opd/math/qwen3-4b-instruct/$VARIANT.sh" | sed '/^SCRIPT_DIR=/,/^export PYTHONPATH=/d')
if [[ "${2:-}" == --dry-run ]]; then
    python recipe/opd/run/utils/run_config.py print
    exit
fi
python preparation/stage_outputs.py
python preparation/check_ready.py
exec bash "recipe/opd/run/opd/math/qwen3-4b-instruct/$VARIANT.sh"
