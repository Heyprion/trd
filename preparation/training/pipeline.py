#!/usr/bin/env python3
"""Train the frozen 9,984 examples and/or benchmark the resulting merged model."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recipe.opd.run.utils.run_config import build_env, print_config  # noqa: E402
from recipe.opd.swanlab_relay import METRIC_PREFIX  # noqa: E402

BENCHMARKS = tuple(os.environ.get("MATH_EVAL_DATASETS", "aime24,aime25").split(","))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def append_history(message):
    with (ROOT / "HISTORY.md").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(message + "\n")


def unique_run_name(prefix, output_root):
    used = {read_json(path).get("experiment_name") for path in Path(output_root).rglob("summary.json")}
    number, candidate = 1, prefix
    while candidate in used:
        number += 1
        candidate = f"{prefix}-r{number:02d}"
    return candidate


def validate_train_split(values):
    import pyarrow.parquet as pq
    from datasets import load_from_disk

    split_root = Path(os.environ["MATH_SPLIT_ROOT"]).resolve()
    expected_path = split_root / "train_9984/dataset"
    if Path(values["TRAIN_DATA_PATH"]).resolve() != expected_path.resolve():
        raise ValueError(f"This entrypoint requires the frozen 9984 subset: {expected_path}")
    for key in ("DATA_PATH", "MAX_SAMPLES", "CORRECTED_RESPONSES_PATH"):
        if values[key]:
            raise ValueError(f"Unset {key}: it can bypass or shorten the frozen 9984 subset")
    manifest = read_json(split_root / "manifest.json")
    if not manifest.get("completed") or manifest["splits"]["train_9984"]["rows"] != 9984:
        raise ValueError("The completed frozen 9984 split manifest is required")
    rows = {}
    for name, count in (("train_9984", 9984), ("validation", 512)):
        folder = split_root / name
        declared = manifest["splits"][name]
        for filename, field in (("data.parquet", "parquet_sha256"), ("ids.json", "ids_sha256"),
                                ("sample_ids.json", "sample_ids_sha256")):
            if file_hash(folder / filename) != declared[field]:
                raise ValueError(f"Frozen split checksum mismatch: {folder / filename}")
        rows[name] = pq.read_table(folder / "data.parquet").to_pylist()
        if len(rows[name]) != count or declared["rows"] != count:
            raise ValueError(f"Frozen {name} row count mismatch")
        if list(load_from_disk(str(folder / "dataset"))) != rows[name]:
            raise ValueError(f"HF dataset differs from frozen parquet: {folder}")
        for key, filename in (("raw_id", "ids.json"), ("sample_id", "sample_ids.json")):
            if [r[key] for r in rows[name]] != read_json(folder / filename):
                raise ValueError(f"Frozen {name} {key} order mismatch")
    for key in ("raw_id", "sample_id", "duplicate_group_id"):
        train_ids = {row[key] for row in rows["train_9984"]}
        val_ids = {row[key] for row in rows["validation"]}
        if len(train_ids) != 9984 or len(val_ids) != 512 or train_ids & val_ids:
            raise ValueError(f"Training/validation identity is duplicated or overlaps: {key}")
    world = int(values["NGPUS_PER_NODE"]) * int(values["NNODES"])
    effective = (world // int(values["SP_SIZE"])) * int(values["TRAIN_BATCH_SIZE"]) * int(
        values["GRADIENT_ACCUMULATION_STEPS"])
    return {"rows": 9984, "validation_rows": 512, "effective_batch": effective,
            "nominal_updates_per_epoch": math.ceil(9984 / effective),
            "manifest_sha256": file_hash(split_root / "manifest.json"),
            "split_version": manifest["config"]["split_version"],
            "parquet_sha256": manifest["splits"]["train_9984"]["parquet_sha256"]}


def validate_generated(path, values):
    """Reject incomplete caches or generated data from any other split before updating."""
    import pyarrow.parquet as pq

    split_root = Path(os.environ["MATH_SPLIT_ROOT"])
    original = pq.read_table(split_root / "train_9984/data.parquet").to_pylist()
    generated = pq.read_table(path).to_pylist()
    if len(generated) != 9984:
        raise ValueError(f"Expected all 9984 generated rows, got {len(generated)}: {path}")
    for row, source in zip(generated, original, strict=True):
        extra = row["extra_info"]
        for key in ("raw_id", "sample_id", "duplicate_group_id", "split_version"):
            if extra.get(key) != source[key]:
                raise ValueError(f"Generated training identity/order mismatch: {key}")
        responses = row.get("responses")
        if responses is None or len(responses) != 1 or not str(responses[0]).strip():
            raise ValueError(f"Missing/empty training response for {extra['sample_id']}")
    return {"rows": len(generated), "sha256": file_hash(path)}


def validate_model(path):
    path = Path(path)
    if not (path / "config.json").is_file() or (path / "adapter_config.json").is_file():
        raise ValueError(f"Expected an exported, merged HF model: {path}")
    read_json(path / "config.json")
    if (path / "lora_adapter").exists() and not (
        (path / "lora_merge.json").is_file() and read_json(path / "lora_merge.json").get("merged") is True
    ):
        raise ValueError(f"LoRA adapter has not been folded into exported HF weights: {path}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weights = [path / name for name in set(read_json(index)["weight_map"].values())]
    else:
        weights = [path / "model.safetensors"]
    if not weights or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise ValueError(f"Missing/incomplete merged HF weights: {path}")


def validate_benchmarks():
    import pyarrow.parquet as pq

    counts = {}
    for name in BENCHMARKS:
        path = Path(os.environ["EVAL_DATASETS_DIR"]) / name / f"{name}_test.parquet"
        table = pq.read_table(path)
        if not len(table) or not {"prompt", "reward_model", "data_source"} <= set(table.column_names):
            raise ValueError(f"Missing/invalid benchmark: {path}")
        counts[name] = {"rows": len(table), "sha256": file_hash(path)}
    return counts


def consume_metric(line, run, metrics_file):
    # torchrun may prefix stdout with a rank; the marker itself is unambiguous.
    if METRIC_PREFIX not in line:
        return False
    payload = json.loads(line.split(METRIC_PREFIX, 1)[1])
    run.log(payload)
    with metrics_file.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")
    return True


def run_command(command, folder, run, env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        with (folder / "console.log").open("a", buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                if not consume_metric(line, run, folder / "metrics.jsonl"):
                    print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        process.stdout.close()


def experiment(stage, run_id, folder, config, action, *, name=None):
    import swanlab

    folder.mkdir(parents=True, exist_ok=False)
    summary_path = folder / "summary.json"
    summary = {"stage": stage, "run_id": run_id, "status": "starting", "config": config,
               "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if name:
        summary["experiment_name"] = name
    write_json(summary_path, summary)
    run = None
    start = time.monotonic()
    try:
        # SwanLab 0.10.5 parses SWANLAB_PROJECT as a nested JSON object.
        # Pass our legacy plain project name explicitly and hide it while the
        # SDK reads settings. Restore it for the next stage and child processes.
        legacy_project = os.environ.pop("SWANLAB_PROJECT", None)
        try:
            run = swanlab.init(project=legacy_project or "trd-math-opd",
                               name=name or f"{run_id}-{stage}", job_type=stage, group=run_id,
                               mode="online", on_online_error="raise", config=config,
                               log_dir=str(folder / "swanlab"), reinit=True)
        finally:
            if legacy_project is not None:
                os.environ["SWANLAB_PROJECT"] = legacy_project
        summary.update(status="running", swanlab_url=run.url)
        write_json(summary_path, summary)
        append_history(f"\n## {summary['started_at']}：数学 TRD {stage} / {run_id}\n\n"
                       f"- SwanLab：{run.url}；配置/日志/状态：`{folder}`。")
        result = action(run)
        summary.update(result=result, status="completed", wall_seconds=time.monotonic() - start)
        run.log({"pipeline/completed": 1, "pipeline/wall_seconds": summary["wall_seconds"]})
        run.finish()
        append_history(f"- {stage} 完成：`{summary_path}`；结果：{json.dumps(result, ensure_ascii=False)}。")
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc(), wall_seconds=time.monotonic() - start)
        print(summary["error"], file=sys.stderr, flush=True)
        append_history(f"- {stage} 失败：`{summary_path}`；错误见 `{folder / 'console.log'}` 和 summary.json。")
        if run is not None:
            run.finish(state="crashed", error=summary["error"])
        raise
    finally:
        write_json(summary_path, summary)
    return summary


def train(run_id, folder, values, split, *, cache_source=None):
    values = values.copy()
    if cache_source:
        values["TRAINING_DATA_PATH"] = str(folder / "cache/chat-training.parquet")

    def action(run):
        output = Path(values["EPOCH_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "swanlab.json", {"url": run.url})
        env = os.environ.copy()
        env.update(values)
        env.update(SWANLAB_RELAY="1", RUN_EVAL_AFTER_TRAINING="false")
        derived = None
        if cache_source:
            from preparation.training.cached_repair import materialize

            cache = folder / "cache/chat-training.parquet"
            derived = materialize(cache_source, cache, values["MODEL_PATH"])
            # launch_training rebuilds values in its child; DATA_PATH is set only
            # here after the frozen split/source checks, never as a user bypass.
            env.update(DATA_PATH=str(cache), TRAINING_DATA_PATH=str(cache))
            run.log({"cache/reused_rows": derived["rows"], "cache/new_generation": 0,
                     "cache/complete_rows_with_eos": derived["complete_rows_with_eos"]})
        scripts = ["preparation/stage_outputs.py", "preparation/check_ready.py"]
        if not cache_source:
            scripts.append("recipe/opd/run/prepare_training_data.py")
        for script in scripts:
            run_command([sys.executable, str(ROOT / script)], folder, run, env)
        generated = validate_generated(env["TRAINING_DATA_PATH"], values)
        run.log({"data/source_rows": 9984, "data/generated_rows": generated["rows"]})
        run_command([sys.executable, str(ROOT / "recipe/opd/run/launch_training.py")], folder, run, env)
        model = str(Path(values["EPOCH_MODEL_SAVE_DIR"]) / "hf_merged")
        validate_model(model)
        return {"model_path": model, "training_data": generated, "output_dir": str(output),
                **({"cache_reuse": derived} if derived else {})}

    config = {"training": values, "split": split, "merge_lora_on_export": True,
              "profile": os.environ.get("MATH_TRAIN_PROFILE", str(ROOT / "preparation/pilot/profiles/balanced.env.sh")),
              "kl_full_vocab_chunk_size": os.environ["KL_FULL_VOCAB_CHUNK_SIZE"]}
    if cache_source:
        config["cache_reuse"] = cache_source
        config["code_sha256"] = {path: file_hash(ROOT / path) for path in (
            "recipe/opd/dataset/data_utils.py", "recipe/opd/config.py", "recipe/opd/kl_trainer.py",
            "recipe/opd/run_training.py", "recipe/opd/run/launch_training.py",
            "preparation/training/cached_repair.py", "preparation/training/pipeline.py",
        )}
    name = None
    if cache_source:
        name = unique_run_name(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d") +
                               "-训练-TRD聊天格式修复-9984题", folder.parent.parent)
    return experiment("train", run_id, folder, config, action, **({"name": name} if name else {}))


def evaluation(run_id, folder, model, source, benchmark_info, *, model_label="distilled", reference=None, run_name=None):
    if model_label not in ("base", "distilled") or (reference and model_label != "base"):
        raise ValueError("A reference comparison is supported only for the original base student")
    values = build_env()
    config = {"model_path": str(model), "source_training": source, "datasets": benchmark_info,
              "pass_k": 16, "temperature": 0.6, "top_p": 0.95, "tp": int(values["EVAL_GEN_TP"]),
              "generation": {key: value for key, value in os.environ.items() if key.startswith("EVAL_")}}
    config["model_label"] = model_label
    if reference:
        config["reference_evaluation"] = reference

    def action(run):
        validate_model(model)
        env = os.environ.copy()
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.update(DATASETS=" ".join(BENCHMARKS), PASS_K="16", GEN_TP=values["EVAL_GEN_TP"],
                   EVAL_MODEL_NAME=f"{run_id}_base" if model_label == "base" else f"{run_id}-distilled",
                   EVAL_BASE_MODEL_NAME=values["MODEL_NAME"],
                   EVAL_RESULTS_FILE=str(folder / "results.json"), EVAL_OUTPUT_DIR=str(folder / "generations"),
                   WRITE_PASS16_AGGREGATES="true", WRITE_RESULTS_CSV="true")
        run_command(["bash", str(ROOT / "recipe/math_evaluation/benchmark_kl_model.sh"), str(model)],
                    folder, run, env)
        entry = read_json(folder / "results.json")[env["EVAL_MODEL_NAME"]]
        metrics = {}
        import pyarrow.parquet as pq

        for name in BENCHMARKS:
            generated = pq.read_table(folder / "generations" / f"{name}_pass16_generation.parquet")
            if len(generated) != benchmark_info[name]["rows"]:
                raise ValueError(f"Incomplete evaluation rows: {name}")
            if any(len(responses) != 16 for responses in generated["responses"].to_pylist()):
                raise ValueError(f"Expected exactly 16 responses for each {name} problem")
            for kind, key in (("pass@16", f"{name}_pass16_generation_pass_16"),
                              ("avg@16", f"{name}_avg_pass1_generation_pass_16"),
                              ("pass@8", f"{name}_pass8_generation_pass_16")):
                value = float(entry[key])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid benchmark metric: {key}={value}")
                metrics[f"eval/{name}/{kind}"] = value
        run.log(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        result = {"model_path": str(model), "model_label": model_label,
                  "metrics": metrics, "results_file": str(folder / "results.json")}
        if reference:
            comparison = {key: {"base": value, "distilled": reference["metrics"][key],
                                "distilled_minus_base_pp": 100 * (reference["metrics"][key] - value)}
                          for key, value in metrics.items()}
            write_json(folder / "comparison.json", {"reference": reference["summary_path"], "metrics": comparison})
            import csv

            with (folder / "comparison.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["metric", "base", "distilled", "distilled_minus_base_pp"])
                for key, row in comparison.items():
                    writer.writerow([key, row["base"], row["distilled"], row["distilled_minus_base_pp"]])
            run.log({f"comparison/{key.removeprefix('eval/')}/distilled_minus_base_pp": row["distilled_minus_base_pp"]
                     for key, row in comparison.items()})
            result["comparison_file"] = str(folder / "comparison.json")
        return result

    return experiment("eval", run_id, folder, config, action, **({"name": run_name} if run_name else {}))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "eval", "all"))
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}")
    parser.add_argument("--output-root", type=Path, default=ROOT / "preparation/logs/math-runs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="Merged HF model for standalone evaluation")
    source.add_argument("--train-run", type=Path, help="Successful train/summary.json for standalone evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration/data; no GPU work or SwanLab run")
    parser.add_argument("--reuse-train-run", type=Path, help="Reuse verified trajectories from a completed frozen training run")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id) or args.run_id in (".", ".."):
        parser.error("--run-id must be a safe directory name")
    if args.stage == "eval" and not (args.model or args.train_run):
        parser.error("eval requires --model <merged HF directory> or --train-run <train/summary.json>")
    if args.stage != "eval" and (args.model or args.train_run):
        parser.error("--model/--train-run are only for standalone eval")
    if args.stage == "eval" and args.reuse_train_run:
        parser.error("--reuse-train-run is for training, not standalone evaluation")
    args.output_root = args.output_root.expanduser().resolve()
    return args


def main(argv=None):
    args = arguments(argv)
    os.environ["RUN_DATE"] = args.run_id
    values = build_env()
    os.environ.update(values)
    run_dir = args.output_root / args.run_id
    source, model, split, benchmarks = None, None, None, None
    cache_source = None
    if args.stage in ("train", "all"):
        split = validate_train_split(values)
        if args.reuse_train_run:
            from preparation.training.cached_repair import validate_source

            if values.get("SEQUENCE_FORMAT") != "chat":
                raise ValueError("Cached repair requires SEQUENCE_FORMAT=chat")
            cache_source = validate_source(args.reuse_train_run, values, split)
            print("Verified cache reuse:", json.dumps(cache_source, ensure_ascii=False), flush=True)
        print_config(values)
        print("Frozen training split:", json.dumps(split, ensure_ascii=False), flush=True)
        model = Path(values["EPOCH_MODEL_SAVE_DIR"]) / "hf_merged"
        if Path(values["EPOCH_OUTPUT_DIR"]).exists() or model.exists():
            raise ValueError("Training outputs already exist for this --run-id; choose a new run ID")
    elif args.train_run:
        source = read_json(args.train_run.expanduser().resolve())
        if source.get("stage") != "train" or source.get("status") != "completed":
            raise ValueError("--train-run must point to a successfully completed training summary")
        model = Path(source["result"]["model_path"]).resolve()
    else:
        model = args.model.expanduser().resolve()
    if args.stage in ("eval", "all"):
        benchmarks = validate_benchmarks()
        if args.stage == "eval":
            validate_model(model)
        print(f"Evaluation model: {model}", flush=True)
        print(f"Benchmarks: {', '.join(BENCHMARKS)}; pass@16; TP={values['EVAL_GEN_TP']}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    if args.dry_run:
        print("DRY RUN passed; no generation, training, scoring, or SwanLab experiment started.")
        return
    if run_dir.exists():
        raise ValueError(f"Run directory already exists; use a new --run-id: {run_dir}")
    run_dir.mkdir(parents=True)
    # One machine shares the training/evaluation GPUs and checkpoint disks.
    lock_path = ROOT / "preparation/logs/math-training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ("train", "all"):
            source = train(args.run_id, run_dir / "train", values, split,
                           **({"cache_source": cache_source} if cache_source else {}))
        if args.stage in ("eval", "all"):
            evaluation(args.run_id, run_dir / "eval", model,
                       {"summary_path": str(args.train_run) if args.train_run else str(run_dir / "train/summary.json"),
                        "swanlab_url": source["swanlab_url"]} if source else None, benchmarks,
                       **({"run_name": unique_run_name(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d") +
                           "-评测-AIME24和25-pass16-聊天格式修复", args.output_root)} if cache_source else {}))
    print(f"Completed: {run_dir}", flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    main()
