#!/usr/bin/env bash
# One short student generation through the actual Ray + verl + vLLM entrypoint.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VERL_ROOT"
mkdir -p preparation/logs
python recipe/opd/generation/y_o_prepare.py \
    --input_path "$TRAIN_DATA_PATH" --output_file "$VERL_ROOT/preparation/logs/smoke-prompts.parquet" \
    --max_samples 1 --task math --data_source deepscaleR
python -m verl.trainer.main_generation_server \
    trainer.nnodes=1 trainer.n_gpus_per_node=1 \
    "actor_rollout_ref.model.path=$MODEL_PATH" actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.prompt_length=2048 actor_rollout_ref.rollout.response_length=32 \
    actor_rollout_ref.rollout.max_model_len=2112 actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.n=1 \
    "data.train_files=['$VERL_ROOT/preparation/logs/smoke-prompts.parquet']" data.prompt_key=prompt \
    "+data.output_path=$VERL_ROOT/preparation/logs/smoke-responses.parquet"
