#!/usr/bin/env bash
# Current provisional data split; source after env.sh and balanced.env.sh.
DATA_SPLIT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MATH_SPLIT_ROOT="${MATH_SPLIT_ROOT:-$DATA_SPLIT_PROJECT_ROOT/data/processed/math-v1/splits-v1}"
export MATH_SPLIT_MANIFEST="$MATH_SPLIT_ROOT/manifest.json"
export VALIDATION_DATA_PATH="$MATH_SPLIT_ROOT/validation/dataset"
export TRAIN_POOL_DATA_PATH="$MATH_SPLIT_ROOT/train_pool/dataset"
export TRAIN_RESERVE_DATA_PATH="$MATH_SPLIT_ROOT/train_reserve/dataset"
export TRAIN_DATA_PATH="${MATH_TRAIN_DATA_PATH:-$MATH_SPLIT_ROOT/train_9984/dataset}"
DATA_SPLIT_TRAIN_TAG="$(basename "$(dirname "$TRAIN_DATA_PATH")")"
export GEN_RESULTS_ROOT="${MATH_GEN_RESULTS_ROOT:-$DATA_SPLIT_PROJECT_ROOT/gen_results/math-v1-splits-v1/$DATA_SPLIT_TRAIN_TAG}"
