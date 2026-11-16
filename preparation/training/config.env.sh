#!/usr/bin/env bash
# Production entrypoints always load the agreed profile and frozen first subset.
MATH_RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$MATH_RUN_ROOT/preparation/env.sh"
source "${MATH_TRAIN_PROFILE:-$MATH_RUN_ROOT/preparation/pilot/profiles/balanced.env.sh}"
source "$MATH_RUN_ROOT/preparation/data_engineering/splits.env.sh"
# Read upstream algorithm/length defaults without launching the recipe.
source <(sed '/^exec bash /d' "$MATH_RUN_ROOT/recipe/opd/run/opd/math/qwen3-4b-instruct/forward_kl_y_r.sh" \
    | sed '/^SCRIPT_DIR=/,/^export PYTHONPATH=/d')
export RUN_EVAL_AFTER_TRAINING=false SAVE_MERGED_MODEL=true SWANLAB_MODE=online PASS_K=16
export MERGE_LORA_ON_EXPORT=true
# Current experiment scope: other prepared benchmarks require an explicit override.
export EVAL_DATASETS="${MATH_EVAL_DATASETS:-aime25}"
# Preserve the upstream benchmark sampling protocol (separate from training rollout).
export EVAL_PROMPT_LENGTH="${EVAL_PROMPT_LENGTH:-4096}"
export EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-38912}"
export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-$((EVAL_PROMPT_LENGTH + EVAL_RESPONSE_LENGTH))}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.85}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-16}"
export EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-$EVAL_MAX_MODEL_LEN}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
