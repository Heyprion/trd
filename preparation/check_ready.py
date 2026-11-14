#!/usr/bin/env python3
"""Fail early on missing/incompatible dependencies, assets, and GPU kernels."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Also execute CUDA matmul and FlashAttention forward/backward")
    args = parser.parse_args()
    report = {"dependencies": {}, "models": {}, "datasets": {}}
    for name in ["torch", "vllm", "transformers", "peft", "datasets", "ray", "flash-attn",
                 "tensordict", "torchdata", "math-verify", "numpy", "pyarrow", "verl"]:
        report["dependencies"][name] = importlib.metadata.version(name)
    for module in ["vllm._C", "flash_attn", "recipe.opd.kl_trainer",
                   "verl.trainer.main_generation_server", "recipe.math_evaluation.run_eval_suite",
                   "verl.workers.rollout.vllm_rollout.vllm_async_server",
                   "verl.utils.reward_score.amobench_parser_reward"]:
        importlib.import_module(module)
        print(f"Import OK: {module}", flush=True)

    import torch
    import pyarrow.parquet as pq
    from datasets import load_from_disk
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    required_gpus = int(os.environ.get("NGPUS_PER_NODE", "2"))
    assert torch.cuda.device_count() >= required_gpus, f"Need {required_gpus} visible GPUs"
    report["gpus"] = [{"name": torch.cuda.get_device_name(i),
                       "capability": list(torch.cuda.get_device_capability(i)),
                       "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)}
                      for i in range(torch.cuda.device_count())]

    tokenizers = []
    for name in ["Qwen3-4B-Instruct-2507", "Qwen3-8B"]:
        path = ROOT / "models" / name
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizers.append(tokenizer)
        index = json.loads((path / "model.safetensors.index.json").read_text())
        weights = set(index["weight_map"].values())
        for filename in weights:
            with safe_open(path / filename, framework="pt", device="cpu") as f:
                assert len(list(f.keys())) > 0
        report["models"][name] = {"path": str(path), "vocab_size": config.vocab_size,
                                  "shards": len(weights), "bytes": sum((path / f).stat().st_size for f in weights)}
    assert tokenizers[0].get_vocab() == tokenizers[1].get_vocab(), "Full-vocab KL requires matching token IDs"

    train_path = ROOT / "data/train/DeepScaleR"
    train = load_from_disk(str(train_path))
    assert len(train) > 0 and {"problem", "answer", "solution"} <= set(train.column_names)
    report["datasets"]["DeepScaleR"] = len(train)
    for name in ["aime24", "aime25", "hmmt25", "beyondaime", "amobench"]:
        path = ROOT / f"data/eval/{name}/{name}_test.parquet"
        table = pq.read_table(path)
        assert len(table) > 0 and {"prompt", "reward_model", "data_source"} <= set(table.column_names)
        assert all(row["prompt"] and row["reward_model"]["ground_truth"] is not None for row in table.to_pylist())
        report["datasets"][name] = len(table)

    if args.gpu:
        from flash_attn import flash_attn_func
        for i in range(required_gpus):
            with torch.cuda.device(i):
                a = torch.randn(32, 32, device=f"cuda:{i}", dtype=torch.bfloat16)
                assert torch.isfinite(a @ a).all()
                q = torch.randn(1, 128, 4, 64, device=f"cuda:{i}", dtype=torch.bfloat16, requires_grad=True)
                out = flash_attn_func(q, q, q, causal=True)
                out.float().square().mean().backward()
                assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
                torch.cuda.synchronize()
        report["cuda_matmul_and_flash_attention_backward"] = "passed"

    report["status"] = "ready"
    destination = ROOT / "preparation/readiness.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
