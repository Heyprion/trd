#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VERL_ROOT"
python preparation/download_assets.py --only datasets
python recipe/opd/dataset/prepare_deepscaler.py \
    --dataset "$DATA_ROOT/raw/agentica-org/DeepScaleR-Preview-Dataset" --output_dir "$TRAIN_DATA_PATH"
python recipe/math_evaluation/datasets/prepare_aime.py \
    --local_dataset_path "$DATA_ROOT/raw" --local_save_dir "$EVAL_DATASETS_DIR"
# Prepare just HMMT25; the upstream CLI also downloads unused HMMT23/24.
python - <<'PY'
import os
from recipe.math_evaluation.datasets.prepare_hmmt import process_hmmt_dataset
result = process_hmmt_dataset(os.environ['DATA_ROOT'] + '/raw/PraMamba/HMMT-202502', '25', os.environ['EVAL_DATASETS_DIR'])
if not result:
    raise SystemExit('HMMT25 preparation failed')
PY
python recipe/math_evaluation/datasets/prepare_additional_eval_datasets.py \
    --datasets beyondaime,amobench --local_dataset_path "$DATA_ROOT/raw" --local_save_dir "$EVAL_DATASETS_DIR"
