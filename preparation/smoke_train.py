#!/usr/bin/env python3
"""One synthetic two-GPU TRD KL update; no checkpoints or benchmark scores.

Run after asset preparation with:
torchrun --standalone --nproc-per-node=2 preparation/smoke_train.py
"""
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["WANDB_MODE"] = "disabled"

import pandas as pd
import torch.distributed as dist
from recipe.opd.config import KLTrainingConfig
from recipe.opd.kl_trainer import KLTrainer

output = ROOT / "preparation/logs/smoke-train"
output.mkdir(parents=True, exist_ok=True)
rank = int(os.environ.get("RANK", "0"))
# Each rank uses a separate identical input to avoid a filesystem write race.
path = output / f"input-rank{rank}.parquet"
pd.DataFrame([
    {"responses": [r"1 + 1 = 2. Therefore \boxed{2}."],
     "extra_info": {"problem": "What is 1+1?", "answer": "2", "expert_cot": "1+1=2",
                    "initial_response": r"The answer is \boxed{3}.", "index": i}}
    for i in range(2)
]).to_parquet(path)
config = KLTrainingConfig(
    distill_mode="opd", task="math", kl_type="forward", kl_method="full_vocab",
    temperature=1.0, kl_token_clip=0,
    student_model_path=str(ROOT / "models/Qwen3-4B-Instruct-2507"),
    teacher_model_path=str(ROOT / "models/Qwen3-8B"),
    use_lora=True, lora_rank=64, lora_alpha=128,
    learning_rate=5e-6, train_batch_size=1, gradient_accumulation_steps=1,
    max_length=512, max_token_len_per_gpu=512, use_initial_response=True,
    data_path=str(path), num_workers=0, n_gpus_per_node=2,
    fsdp_strategy="fsdp2", use_torch_compile=False,
    save_steps=100000, save_merged_model=False, run_eval_after_training=False,
    output_dir=str(output), model_save_dir=str(output), logging_steps=1,
)
trainer = KLTrainer(config)
try:
    metrics = trainer.train_epoch()
    assert trainer.global_step == 1
    assert all(math.isfinite(v) for v in metrics.values()), metrics
    (output / f"result-rank{rank}.json").write_text(json.dumps({"status": "passed", "optimizer_steps": trainer.global_step, **metrics}, indent=2) + "\n")
    print(f"Rank {rank}: one synthetic TRD KL optimizer update passed: {metrics}", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
trainer.close()
