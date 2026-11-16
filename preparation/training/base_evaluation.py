#!/usr/bin/env python3
"""Evaluate the original 4B student under the completed distilled AIME protocol."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from preparation.training import pipeline as p  # noqa: E402

DEFAULT_REFERENCE = ROOT / "preparation/logs/math-runs/20261115-185850-279780-53607/eval/summary.json"
DATASETS = ("aime24", "aime25")
GENERATION_KEYS = ("EVAL_PROMPT_LENGTH", "EVAL_RESPONSE_LENGTH", "EVAL_MAX_MODEL_LEN",
                   "EVAL_GPU_MEMORY_UTILIZATION", "EVAL_MAX_NUM_SEQS", "EVAL_MAX_NUM_BATCHED_TOKENS")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_protocol(path, benchmarks):
    reference = p.read_json(path)
    if reference.get("stage") != "eval" or reference.get("status") != "completed":
        raise ValueError("Reference must be a completed distilled evaluation summary")
    cfg = reference["config"]
    if tuple(cfg["datasets"]) != DATASETS or tuple(benchmarks) != DATASETS:
        raise ValueError("Base comparison is restricted to AIME24 then AIME25")
    if any(benchmarks[name] != cfg["datasets"][name] for name in DATASETS):
        raise ValueError("Benchmark rows/checksums differ from the distilled evaluation")
    if (cfg["pass_k"], cfg["temperature"], cfg["top_p"], cfg["tp"]) != (16, .6, .95, 2):
        raise ValueError("Reference is incompatible with the shared pass16/temp0.6/top_p0.95/TP2 runner")
    for name in DATASETS:
        for metric in ("pass@16", "avg@16", "pass@8"):
            key = f"eval/{name}/{metric}"
            if key not in reference["result"]["metrics"]:
                raise ValueError(f"Reference lacks final {name}/{metric}")
            value = float(reference["result"]["metrics"][key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Reference has invalid {name}/{metric}")
    # Pin recorded limits/resources rather than inheriting stale shell overrides.
    for key in GENERATION_KEYS:
        os.environ[key] = str(cfg["generation"][key])
    os.environ.update(EVAL_GEN_TP="2", NNODES="1", NGPUS_PER_NODE="2",
                      EVAL_DATASETS=",".join(DATASETS), MATH_EVAL_DATASETS=",".join(DATASETS))
    return reference


def check_tokenizer_protocol(base, distilled, dataset_dir):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    def generation_config(path):
        return {key: value for key, value in p.read_json(path / "generation_config.json").items()
                if key != "transformers_version" and not key.startswith("_")}

    if generation_config(base) != generation_config(distilled):
        raise ValueError("Base/finished model generation defaults differ")
    left = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=True)
    right = AutoTokenizer.from_pretrained(distilled, local_files_only=True, trust_remote_code=True)
    if left.get_vocab() != right.get_vocab() or left.special_tokens_map != right.special_tokens_map:
        raise ValueError("Base/finished model tokenizer vocabulary or special tokens differ")
    prompts = []
    for name in DATASETS:
        rows = pq.read_table(dataset_dir / name / f"{name}_test.parquet", columns=["prompt"]).to_pylist()
        for row in rows:
            a = left.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            b = right.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            if a != b:
                raise ValueError(f"Base/finished model chat-template token IDs differ: {name}")
            prompts.append(a)
    return {"vocab_sha256": fingerprint(left.get_vocab()), "prompt_token_ids_sha256": fingerprint(prompts),
            "compared_prompts": len(prompts), "generation_defaults_sha256": fingerprint(generation_config(base)),
            "seed": 0, "seed_origin": "Original rollout replica_rank=0, default seed=0; TP=2 uses one replica"}


def run_name(output_root):
    prefix = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d") + "-评测-AIME24和25-pass16-基线"
    used = set()
    if output_root.exists():
        for path in output_root.glob("*/eval/summary.json"):
            used.add(p.read_json(path).get("experiment_name", ""))
    number, candidate = 1, prefix
    while candidate in used:
        number += 1
        candidate = f"{prefix}-r{number:02d}"
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-eval", type=Path, default=Path(os.environ.get("BASE_EVAL_REFERENCE", DEFAULT_REFERENCE)))
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("base-%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}")
    parser.add_argument("--output-root", type=Path, default=ROOT / "preparation/logs/math-base-eval")
    parser.add_argument("--dry-run", action="store_true", help="Check protocol/assets without GPU inference or SwanLab")
    args = parser.parse_args(argv)
    import re

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id) or args.run_id in (".", ".."):
        parser.error("--run-id must be a safe directory name")
    args.output_root = args.output_root.expanduser().resolve()
    args.reference_eval = args.reference_eval.expanduser().resolve()
    base = ROOT / "models/Qwen3-4B-Instruct-2507"
    p.BENCHMARKS = DATASETS
    p.validate_model(base)
    if (base / "lora_adapter").exists() or (base / "adapter_config.json").exists():
        raise ValueError("Base evaluation requires the original student without any adapter")
    benchmarks = p.validate_benchmarks()
    reference = load_protocol(args.reference_eval, benchmarks)
    compatibility = check_tokenizer_protocol(base, Path(reference["config"]["model_path"]),
                                            Path(os.environ["EVAL_DATASETS_DIR"]))
    context = {"summary_path": str(args.reference_eval), "summary_sha256": p.file_hash(args.reference_eval),
               "swanlab_url": reference["swanlab_url"], "model_path": reference["config"]["model_path"],
               "metrics": reference["result"]["metrics"], "compatibility": compatibility}
    run_dir = args.output_root / args.run_id
    if run_dir.exists():
        raise ValueError(f"Output already exists; choose a fresh run ID: {run_dir}")
    print(json.dumps({"base_model": str(base), "datasets": benchmarks, "pass_k": 16, "temperature": .6,
                      "top_p": .95, "tp": 2, "generation": {k: os.environ[k] for k in GENERATION_KEYS},
                      "seed": 0, "reference_eval": str(args.reference_eval), "compatibility": compatibility,
                      "output": str(run_dir / "eval")}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("DRY RUN passed; no generation, model scoring, or SwanLab experiment started.")
        return
    lock_path = ROOT / "preparation/logs/math-training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir.mkdir(parents=True)
        os.environ["RUN_DATE"] = args.run_id
        p.evaluation(args.run_id, run_dir / "eval", base, None, benchmarks, model_label="base",
                     reference=context, run_name=run_name(args.output_root))
    print(f"Base evaluation completed: {run_dir / 'eval/summary.json'}", flush=True)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGTERM, p.interrupted)
    main()
