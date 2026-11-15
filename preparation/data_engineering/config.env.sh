#!/usr/bin/env bash
# Environment overrides and CLI options take precedence over these defaults.
DATA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$DATA_SCRIPT_DIR/../../recipe/opd" ]]; then
  DATA_DEFAULT_ROOT="$(cd "$DATA_SCRIPT_DIR/../.." && pwd)"
else
  DATA_DEFAULT_ROOT="$DATA_SCRIPT_DIR"
fi
export PROJECT_ROOT="${PROJECT_ROOT:-$DATA_DEFAULT_ROOT}"
export INPUT_PATH="${INPUT_PATH:-$PROJECT_ROOT/data/train/DeepScaleR}"
export EVAL_DIR="${EVAL_DIR:-$PROJECT_ROOT/data/eval}"
export LABEL_MODEL="${LABEL_MODEL:-$PROJECT_ROOT/models/Qwen3-8B}"
export OMNI_PATH="${OMNI_PATH:-$PROJECT_ROOT/data/raw/KbsdJames/Omni-MATH/test.jsonl}"
export DOWNLOAD_OMNI="${DOWNLOAD_OMNI:-true}"
export DATA_VERSION="${DATA_VERSION:-math-v1}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/processed/$DATA_VERSION}"
export HISTORY_FILE="${HISTORY_FILE:-$PROJECT_ROOT/HISTORY.md}"
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export CPU_WORKERS="${CPU_WORKERS:-16}"
export GPU_MAX_NUM_SEQS="${GPU_MAX_NUM_SEQS:-128}"
export GPU_BATCHED_TOKENS="${GPU_BATCHED_TOKENS:-32768}"
export GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
# Short-output jobs benefit from avoiding first-run torch.compile/cudagraph cost.
export ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
export TOTAL_SECONDS="${TOTAL_SECONDS:-3600}"
export CLEAN_SECONDS="${CLEAN_SECONDS:-1800}"
export EXPORT_RESERVE="${EXPORT_RESERVE:-180}"
export NEAR_THRESHOLD="${NEAR_THRESHOLD:-0.82}"
export EVAL_NEAR_THRESHOLD="${EVAL_NEAR_THRESHOLD:-0.45}"
export MAX_NEAR_PAIRS="${MAX_NEAR_PAIRS:-20000}"
export SEED="${SEED:-20261115}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-trd-math-opd}"
export RUN_NAME="${RUN_NAME:-math-data-engineering}"
