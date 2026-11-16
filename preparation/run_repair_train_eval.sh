#!/usr/bin/env bash
set -euo pipefail
export REPAIR_STAGE=all
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_repair_train.sh" "$@"
