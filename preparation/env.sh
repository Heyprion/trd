#!/usr/bin/env bash
# Source from any directory: source /root/autodl-tmp/trd/preparation/env.sh
export VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$VERL_ROOT/.venv/bin/activate"
export PYTHON_BIN="$VERL_ROOT/.venv/bin/python"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$VERL_ROOT/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export DATA_ROOT="$VERL_ROOT/data"
# Activate the frozen provisional subset once it has been generated. Preserve
# explicit overrides when this file is sourced again by run_math_opd.sh.
if [[ -z "${TRAIN_DATA_PATH:-}" && -f "$DATA_ROOT/processed/math-v1/splits-v1/manifest.json" ]]; then
    source "$VERL_ROOT/preparation/data_engineering/splits.env.sh"
fi
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATA_ROOT/train/DeepScaleR}"
export EVAL_DATASETS_DIR="$DATA_ROOT/eval"
export EVAL_DATASETS="${MATH_EVAL_DATASETS:-aime25}"
export MODEL_PATH="$VERL_ROOT/models/Qwen3-4B-Instruct-2507"
export TEACHER_MODEL_PATH="$VERL_ROOT/models/Qwen3-8B"
export MODEL_NAME="Qwen3-4B-Instruct-2507"
export STUDENT_MODEL="$MODEL_NAME"
export TEACHER_MODEL="Qwen3-8B"
export TASK=math DISTILL_MODE=opd TRAIN_DATA_SOURCE=deepscaleR
export EVAL_RECIPE_DIR="$VERL_ROOT/recipe/math_evaluation"
export EVAL_BENCHMARK_SCRIPT=benchmark_kl_model.sh
# Keep the full FSDP checkpoint on the 30GB system disk, and the merged model
# on the 50GB data disk. Both copies together would overflow the data disk.
export MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-/root/trd-checkpoints}"
export HF_EXPORT_ROOT="${HF_EXPORT_ROOT:-$VERL_ROOT/checkpoints_merged}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}" NNODES="${NNODES:-1}"
export EVAL_GEN_TP="${EVAL_GEN_TP:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
# Original wrappers: 8 GPUs * batch 1 * accumulation 16 = global batch 128.
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export OTEL_SDK_DISABLED=true OTEL_METRICS_EXPORTER=none OTEL_TRACES_EXPORTER=none OTEL_LOGS_EXPORTER=none
export TOKENIZERS_PARALLELISM=false
