#!/usr/bin/env python3
"""Score the frozen math validation split; parent logs to SwanLab, GPU workers only score."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

from scoring import PROTOCOL, aggregate, encode_row, grouped_metrics, make_batches, paired_comparison

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_history(path, message):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(message + "\n")


def swanlab_run_name(label, output_root, day=None):
    day = day or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    purpose = {"baseline": "基线", "distilled": "蒸馏后", "calibration": "校准",
               "chat_repaired": "聊天格式修复后"}.get(label, label)
    previous = sum((folder / "swanlab.json").is_file()
                   for folder in Path(output_root).glob(f"{label}-{day}-*"))
    suffix = f"-r{previous + 1:02d}" if previous else ""
    return f"{day}-评测-PPL和KL-{purpose}{suffix}"


def arguments(argv=None):
    def env(name, default):
        return os.environ.get("PPL_KLD_" + name, str(default))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=Path(env("SPLIT_ROOT", ROOT / "data/processed/math-v1/splits-v1")))
    parser.add_argument("--base-model", type=Path, default=Path(env("BASE_MODEL", ROOT / "models/Qwen3-4B-Instruct-2507")),
                        help="Original student and tokenizer; also base weights when student is a LoRA adapter")
    parser.add_argument("--student-model", type=Path, default=Path(env("STUDENT_MODEL", env("BASE_MODEL", ROOT / "models/Qwen3-4B-Instruct-2507"))),
                        help="Local HF model or LoRA adapter directory; sharded FSDP checkpoint is not an HF model")
    parser.add_argument("--teacher-model", type=Path, default=Path(env("TEACHER_MODEL", ROOT / "models/Qwen3-8B")))
    parser.add_argument("--output-root", type=Path, default=Path(env("OUTPUT_ROOT", ROOT / "preparation/logs/ppl-kld")))
    parser.add_argument("--label", default="baseline", help="Result tag; use distilled for post-training checkpoints")
    parser.add_argument("--compare-to", type=Path, help="Baseline run directory, summary.json, or latest-baseline.json")
    parser.add_argument("--gpu-ids", default=env("GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")))
    parser.add_argument("--batch-size", type=int, default=int(env("BATCH_SIZE", 8)))
    parser.add_argument("--batch-tokens", type=int, default=int(env("BATCH_TOKENS", 8192)), help="Padded tokens per batch per GPU")
    parser.add_argument("--chunk-size", type=int, default=int(env("CHUNK_SIZE", 512)), help="Response positions per full-vocabulary projection/KL chunk")
    parser.add_argument("--max-length", type=int, default=int(env("MAX_LENGTH", 18432)), help="Overflow fails; references are never truncated")
    parser.add_argument("--memory-fraction", type=float, default=float(env("MEMORY_FRACTION", .92)))
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--max-samples", type=int, help="Optional calibration prefix; full frozen validation is the default")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--project", default=env("PROJECT", "trd-math-opd"))
    parser.add_argument("--history-file", type=Path, default=ROOT / "HISTORY.md")
    parser.add_argument("--dry-run", action="store_true", help="Validate/tokenize inputs without model inference, CUDA, or SwanLab")
    a = parser.parse_args(argv)
    for field in ("split_root", "base_model", "student_model", "teacher_model", "output_root", "history_file", "compare_to"):
        value = getattr(a, field)
        if value is not None:
            setattr(a, field, value.expanduser().resolve())
    a.gpu_ids = [value.strip() for value in a.gpu_ids.split(",") if value.strip()]
    if not a.gpu_ids or len(set(a.gpu_ids)) != len(a.gpu_ids) or "-1" in a.gpu_ids:
        parser.error("--gpu-ids must list distinct available GPU IDs or UUIDs")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.label):
        parser.error("--label accepts letters, digits, underscore and hyphen")
    if min(a.batch_size, a.batch_tokens, a.chunk_size, a.max_length) < 1 or not 0 < a.memory_fraction < 1:
        parser.error("Batch/length/chunk sizes must be positive and memory fraction must be in (0,1)")
    if a.max_samples is not None and a.max_samples < 1:
        parser.error("--max-samples must be positive")
    if a.bootstrap_repeats < 100:
        parser.error("Use at least 100 paired bootstrap repeats")
    return a


def load_validation(split_root):
    """Verify both file integrity and the actual HF directory, plus training-group isolation."""
    import pyarrow.parquet as pq
    from datasets import load_from_disk
    manifest = read_json(split_root / "manifest.json")
    if not manifest.get("completed"):
        raise ValueError("Split manifest is not completed; run the data split pipeline first")
    for name in ("validation", "train_pool"):
        folder, declared = split_root / name, manifest["splits"][name]
        for filename, key in (("data.parquet", "parquet_sha256"), ("ids.json", "ids_sha256"),
                              ("sample_ids.json", "sample_ids_sha256")):
            if file_hash(folder / filename) != declared[key]:
                raise ValueError(f"Frozen split checksum mismatch: {folder / filename}")
    validation = list(load_from_disk(str(split_root / "validation/dataset")))
    parquet = pq.read_table(split_root / "validation/data.parquet").to_pylist()
    if digest(validation) != digest(parquet):
        raise ValueError("Validation HF directory differs from its frozen parquet")
    expected = manifest["splits"]["validation"]["rows"]
    if len(validation) != expected or not validation:
        raise ValueError("Frozen validation row count mismatch")
    if [r["raw_id"] for r in validation] != read_json(split_root / "validation/ids.json"):
        raise ValueError("Frozen validation raw-ID order mismatch")
    if [r["sample_id"] for r in validation] != read_json(split_root / "validation/sample_ids.json"):
        raise ValueError("Frozen validation sample-ID order mismatch")
    pool = pq.read_table(split_root / "train_pool/data.parquet",
                         columns=["raw_id", "sample_id", "duplicate_group_id"]).to_pylist()
    if len(pool) != manifest["splits"]["train_pool"]["rows"]:
        raise ValueError("Training pool row count mismatch")
    for key in ("raw_id", "sample_id", "duplicate_group_id"):
        left, right = {r[key] for r in validation}, {r[key] for r in pool}
        if len(left) != len(validation) or left & right or None in left or "" in left:
            raise ValueError(f"Validation IDs are not unique/isolated: {key}")
    return validation, manifest


def model_identity(path, hash_weights=False):
    adapter = (path / "adapter_config.json").is_file()
    config_path = path / ("adapter_config.json" if adapter else "config.json")
    if not config_path.is_file():
        raise ValueError(f"Not an HF model or LoRA directory: {path}; export/merge FSDP checkpoints first")
    if adapter and read_json(config_path).get("peft_type") != "LORA":
        raise ValueError(f"Only LoRA adapters are supported by the hidden-state scoring path: {path}")
    weights = sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin")) | set(path.glob("adapter_model*.bin")))
    if not weights:
        raise ValueError(f"No HF/adapter weights in {path}")
    files = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
              **({"sha256": file_hash(p)} if hash_weights else {})} for p in weights]
    content = {"config_sha256": file_hash(config_path),
               "files": [{k: r[k] for k in ("name", "bytes", "sha256")} for r in files]} if hash_weights else None
    return {"path": str(path), "kind": "lora_adapter" if adapter else "hf_model", "config_sha256": file_hash(config_path),
            "weights": files, "content_sha256": digest(content) if content else None}


def preflight(a):
    from transformers import AutoConfig, AutoTokenizer
    rows, manifest = load_validation(a.split_root)
    if a.max_samples is not None and a.max_samples > len(rows):
        raise ValueError("--max-samples exceeds frozen validation size")
    tokenizer = AutoTokenizer.from_pretrained(str(a.base_model), local_files_only=True)
    vocab_hash = digest(tokenizer.get_vocab())
    for path in {a.base_model, a.teacher_model, a.student_model}:
        if path == a.student_model and (path / "adapter_config.json").is_file():
            cfg_path = a.base_model
        else:
            cfg_path = path
        identity = model_identity(path)
        cfg = AutoConfig.from_pretrained(str(cfg_path), local_files_only=True)
        if cfg.model_type != "qwen3":
            raise ValueError(f"Unsupported model_type={cfg.model_type}: {path}")
        if path != a.base_model and (path / "tokenizer_config.json").exists():
            other = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            if digest(other.get_vocab()) != vocab_hash:
                raise ValueError(f"Tokenizer ID mapping differs: {path}")
        if path == a.teacher_model and identity["kind"] != "hf_model":
            raise ValueError("Use the fixed full teacher model, not a teacher adapter")
    base_cfg = AutoConfig.from_pretrained(str(a.base_model), local_files_only=True)
    teacher_cfg = AutoConfig.from_pretrained(str(a.teacher_model), local_files_only=True)
    student_cfg = AutoConfig.from_pretrained(str(a.base_model if (a.student_model / "adapter_config.json").exists()
                                                    else a.student_model), local_files_only=True)
    if len({base_cfg.vocab_size, teacher_cfg.vocab_size, student_cfg.vocab_size}) != 1:
        raise ValueError("Model output vocabularies differ")
    encoded = [encode_row(row, tokenizer, a.max_length) for row in rows]
    for row in encoded:
        row["tokens_sha256"] = digest(row["input_ids"])
    expected_tokens = manifest.get("validation_token_stats", {}).get("response_tokens")
    if expected_tokens is not None and sum(r["response_tokens"] for r in encoded) != expected_tokens:
        raise ValueError("Reference token count differs from the frozen tokenizer/split")
    encoded = encoded if a.max_samples is None else encoded[:a.max_samples]
    if max(max(r["input_ids"]) for r in encoded) >= base_cfg.vocab_size:
        raise ValueError("Input token is outside the shared model output vocabulary")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("No padding/EOS token available")
    batches = make_batches(encoded, a.batch_size, a.batch_tokens)
    data = {"split_version": manifest["config"]["split_version"],
            "manifest_sha256": file_hash(a.split_root / "manifest.json"),
            "validation_parquet_sha256": manifest["splits"]["validation"]["parquet_sha256"],
            "frozen_rows": len(rows), "scored_rows": len(encoded),
            "full_validation": len(encoded) == len(rows),
            "response_tokens": sum(r["response_tokens"] for r in encoded),
            "input_tokens": sum(len(r["input_ids"]) for r in encoded),
            "max_input_tokens": max(len(r["input_ids"]) for r in encoded),
            "tokenizer_vocab_sha256": vocab_hash,
            "scored_token_ids_sha256": digest([{k: row[k] for k in ("sample_id", "tokens_sha256", "prompt_tokens", "response_tokens")} for row in encoded]),
            "train_pool_overlap": 0}
    protocol = {**PROTOCOL, "max_length": a.max_length, "tokenizer_vocab_sha256": vocab_hash,
                "scored_token_ids_sha256": data["scored_token_ids_sha256"],
                "validation_parquet_sha256": data["validation_parquet_sha256"]}
    config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "data": data, "protocol": protocol, "protocol_sha256": digest(protocol),
              "execution": {"parallelism": "independent GPU replicas; disjoint validation batches; no gradients",
                            "model_dtype": "BF16; adapter tensors retain PEFT FP32 inference casting",
                            "pad_token_id": pad, "batches": len(batches)},
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "swanlab")},
              "code_sha256": digest({p.name: file_hash(p) for p in sorted(FOLDER.glob("*.py")) if not p.name.startswith("test_")}),
              "models": {"base": model_identity(a.base_model), "student": model_identity(a.student_model),
                         "teacher": model_identity(a.teacher_model)}}
    return config, encoded, batches


def load_baseline(path):
    path = path / "summary.json" if path.is_dir() else path
    summary = read_json(path)
    if "summary_path" in summary:
        path = Path(summary["summary_path"])
        summary = read_json(path)
    if not summary.get("completed"):
        raise ValueError("Baseline summary is not completed")
    records_path = path.parent / summary["records_file"]
    if file_hash(records_path) != summary["records_sha256"]:
        raise ValueError("Baseline per-sample results checksum mismatch")
    return summary, read_jsonl(records_path), str(path)


def check_comparable(config, baseline, *, check_teacher=True):
    if baseline["config"]["protocol_sha256"] != config["protocol_sha256"]:
        raise ValueError("Baseline scoring data/tokenization/protocol differs; comparison is refused")
    if check_teacher and baseline["config"]["models"]["teacher"]["content_sha256"] != config["models"]["teacher"]["content_sha256"]:
        raise ValueError("Baseline teacher weights/config differ; comparison is refused")


def load_model(path, base_path, attention, *, device="cuda:0", dtype=None):
    import torch
    from transformers import AutoModelForCausalLM
    is_adapter = (path / "adapter_config.json").is_file()
    model = AutoModelForCausalLM.from_pretrained(str(base_path if is_adapter else path), local_files_only=True,
                                                dtype=dtype or torch.bfloat16, device_map={"": device},
                                                attn_implementation=attention)
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(path), is_trainable=False, autocast_adapter_dtype=True,
                                          device_map={"": device}, torch_device=device)
    model.eval()
    model.requires_grad_(False)
    return model


def worker(config_path, batch_path, output, rank):
    """Each subprocess sees exactly its assigned CUDA_VISIBLE_DEVICES device."""
    import torch
    from scoring import score_batch
    config, batches = read_json(config_path), read_json(batch_path)
    a = config["arguments"]
    torch.set_num_threads(4)
    torch.manual_seed(a["seed"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU worker must see exactly one CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(a["memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    then = time.monotonic()
    student = load_model(Path(a["student_model"]), Path(a["base_model"]), a["attention"])
    teacher = load_model(Path(a["teacher_model"]), Path(a["base_model"]), a["attention"])
    load_seconds = time.monotonic() - then
    scored, response_tokens = 0, 0
    score_start = time.monotonic()
    with (output / f"rank-{rank}.jsonl").open("w") as stream:
        for batch in batches:
            result = score_batch(student, teacher, batch, config["execution"]["pad_token_id"], a["chunk_size"])
            for row in result:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            scored += len(result)
            response_tokens += sum(r["response_tokens"] for r in result)
            write_json(output / f"progress-{rank}.json", {"rows": scored, "response_tokens": response_tokens})
            print(f"GPU {rank}: {scored} rows / {response_tokens} response tokens", flush=True)
    torch.cuda.synchronize()
    summary = {"rows": scored, "response_tokens": response_tokens, "load_seconds": load_seconds,
               "score_seconds": time.monotonic() - score_start,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
               "gpu_name": torch.cuda.get_device_name(0)}
    write_json(output / f"worker-{rank}.json", summary)


def run_workers(a, config, batches, output, progress_callback):
    # Largest padded batches first, assigned greedily to balance work across GPUs.
    active = min(len(a.gpu_ids), len(batches))
    assignments, loads = [[] for _ in range(active)], [0] * active
    for batch in sorted(batches, key=lambda b: len(b) * max(len(r["input_ids"]) for r in b), reverse=True):
        rank = min(range(active), key=lambda i: loads[i])
        assignments[rank].append(batch)
        loads[rank] += len(batch) * max(len(r["input_ids"]) for r in batch)
    processes, streams = [], []
    try:
        for rank, assignment in enumerate(assignments):
            batch_file = output / f"batches-{rank}.json"
            write_json(batch_file, assignment)
            stream = (output / f"gpu-{rank}.log").open("w")
            streams.append(stream)
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu_ids[rank], "TOKENIZERS_PARALLELISM": "false"}
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(output / "config.json"),
                       str(batch_file), str(output), str(rank)]
            processes.append(subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT))
        previous, last_log = -1, 0.0
        while True:
            codes = [p.poll() for p in processes]
            for rank, code in enumerate(codes):
                if code not in (None, 0):
                    raise RuntimeError(f"GPU worker {rank} exited with {code}; inspect {output / f'gpu-{rank}.log'}")
            count = sum(read_json(p)["rows"] for p in output.glob("progress-*.json"))
            if count != previous and time.monotonic() - last_log >= 5:
                progress_callback(count)
                previous, last_log = count, time.monotonic()
            if all(code == 0 for code in codes):
                break
            time.sleep(1)
        records = []
        for rank in range(active):
            records.extend(read_jsonl(output / f"rank-{rank}.jsonl"))
        return records, [read_json(output / f"worker-{rank}.json") for rank in range(active)]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in streams:
            stream.close()


def report_markdown(summary):
    overall = summary["overall"]
    lines = [f"# {summary['label']} validation PPL/KLD", "", f"SwanLab: {summary['swanlab_url']}", "",
             f"Rows: {overall['rows']}; response tokens: {overall['response_tokens']}; full validation: {summary['config']['data']['full_validation']}", "",
             "Same-input fixed-reference prefixes; temperature 1, full vocabulary, response-only token-weighted means.", "",
             "| Metric | Value |", "| --- | ---: |"]
    for key in ("student_nll", "student_ppl", "teacher_nll", "teacher_ppl", "forward_kl", "reverse_kl"):
        lines.append(f"| {key} | {overall[key]:.8f} |")
    if "comparison" in summary:
        lines.extend(["", "Delta = current minus baseline; 95% intervals use paired problem bootstrap.", "",
                      "| Metric | Baseline | Current | Delta | Delta 95% CI |", "| --- | ---: | ---: | ---: | --- |"])
        for key, row in summary["comparison"]["metrics"].items():
            lines.append(f"| {key} | {row['baseline']:.8f} | {row['current']:.8f} | {row['delta']:.8f} | {row['delta_ci95']} |")
    lines.extend(["", "This measures reference prediction/distribution matching, not mathematical accuracy or live on-policy KL."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    started = time.monotonic()
    a = arguments(argv)
    config, encoded, batches = preflight(a)
    baseline = load_baseline(a.compare_to) if a.compare_to else None
    if baseline:
        check_comparable(config, baseline[0], check_teacher=False)
    if a.dry_run:
        print(json.dumps({"dry_run": True, **config}, ensure_ascii=False, indent=2))
        return
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = a.output_root / f"{a.label}-{timestamp}-{os.getpid()}"
    output.mkdir(parents=True)
    logger = logging.getLogger("ppl-kld")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handlers = (logging.StreamHandler(sys.stdout), logging.FileHandler(output / "evaluation.log"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # SwanLab 0.10.5 parses a flat SWANLAB_PROJECT env var as nested JSON.
    os.environ.pop("SWANLAB_PROJECT", None)
    import swanlab
    run_name = swanlab_run_name(a.label, a.output_root)
    config["swanlab_run_name"] = run_name
    run = swanlab.init(project=a.project, name=run_name, mode="online", on_online_error="raise",
                       log_dir=str(output / "swanlab"), job_type="ppl-kld", config=config)
    write_json(output / "swanlab.json", {"url": run.url, "id": run.id, "name": run_name})
    a.history_file.parent.mkdir(parents=True, exist_ok=True)
    append_history(a.history_file, f"\n## {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S UTC}：验证集PPL/KLD {a.label}\n\n"
                   f"- SwanLab：{run.url}；输出/日志：`{output}`；学生：`{a.student_model}`；"
                   f"验证题数{len(encoded)}/{config['data']['frozen_rows']}。")
    logger.info("SwanLab: %s", run.url)
    logger.info("Output: %s", output)
    def handle_sigterm(*_):
        raise KeyboardInterrupt("Evaluation received SIGTERM")
    previous_sigterm = signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        # Strong content hashes are computed only for real runs, before any GPU work.
        logger.info("Hashing local model weights for provenance and fixed-teacher comparison")
        identities = {}
        for key, path in (("base", a.base_model), ("student", a.student_model), ("teacher", a.teacher_model)):
            if path not in identities:
                identities[path] = model_identity(path, hash_weights=True)
            config["models"][key] = identities[path]
        if baseline:
            check_comparable(config, baseline[0])
        write_json(output / "config.json", config)
        swanlab.log({"artifacts/config": swanlab.Text(json.dumps(config, ensure_ascii=False))})
        def progress(count):
            logger.info("Scored %d/%d validation questions", count, len(encoded))
            swanlab.log({"progress/rows": count, "progress/fraction": count / len(encoded)})
        records, workers = run_workers(a, config, batches, output, progress)
        records = sorted(records, key=lambda r: r["sample_id"])
        expected = {r["sample_id"]: r for r in encoded}
        if len(records) != len(expected) or {r["sample_id"] for r in records} != expected.keys():
            raise ValueError("Worker results are duplicated, missing or unexpected")
        for row in records:
            source = expected[row["sample_id"]]
            if any(row[k] != source[k] for k in ("tokens_sha256", "response_tokens", "prompt_tokens")):
                raise ValueError("Worker response alignment changed")
        overall = aggregate(records)
        if overall["response_tokens"] != config["data"]["response_tokens"]:
            raise ValueError("Worker token count differs from preflight")
        records_path = output / "per-sample.jsonl"
        records_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
        summary = {"completed": True, "label": a.label, "swanlab_url": run.url, "config": config,
                   "overall": overall, "groups": grouped_metrics(records), "workers": workers,
                   "wall_seconds": time.monotonic() - started,
                   "records_file": records_path.name, "records_sha256": file_hash(records_path)}
        if baseline:
            summary["comparison"] = {"baseline_summary": baseline[2], **paired_comparison(baseline[1], records,
                                                                                         a.bootstrap_repeats, a.seed)}
        write_json(output / "summary.json", summary)
        (output / "REPORT.md").write_text(report_markdown(summary))
        swanlab.log({**{"validation/" + k: v for k, v in overall.items()},
                     "runtime/wall_seconds": summary["wall_seconds"],
                     "artifacts/summary": swanlab.Text(json.dumps(summary, ensure_ascii=False)),
                     "artifacts/report": swanlab.Text((output / "REPORT.md").read_text())})
        for field, groups in summary["groups"].items():
            swanlab.log({f"{field}/{name}/{metric}": values[metric] for name, values in groups.items()
                         for metric in ("student_ppl", "teacher_ppl", "forward_kl", "reverse_kl")})
        if baseline:
            swanlab.log({f"comparison/{name}/delta": value["delta"] for name, value in summary["comparison"]["metrics"].items()})
        for path in [output / "evaluation.log", *output.glob("gpu-*.log")]:
            swanlab.log({"logs/" + path.stem: swanlab.Text(path.read_text(errors="replace")[-60000:])})
        swanlab.finish()
        if a.label == "baseline" and config["data"]["full_validation"]:
            write_json(a.output_root / "latest-baseline.json", {"summary_path": str(output / "summary.json"),
                                                                "swanlab_url": run.url})
        append_history(a.history_file, f"- 完成：学生PPL={overall['student_ppl']:.8f}，教师PPL={overall['teacher_ppl']:.8f}，"
                       f"forward KL(T||S)={overall['forward_kl']:.8f}，reverse KL(S||T)={overall['reverse_kl']:.8f} nats/token；"
                       f"有效response tokens={overall['response_tokens']}；耗时{summary['wall_seconds']:.1f}s；"
                       f"汇总`{output / 'summary.json'}`。")
        logger.info("Complete: student PPL=%.8f; teacher PPL=%.8f; FKL=%.8f; RKL=%.8f", overall["student_ppl"],
                    overall["teacher_ppl"], overall["forward_kl"], overall["reverse_kl"])
        logger.info("Report: %s", output / "REPORT.md")
    except BaseException:
        error = traceback.format_exc()
        write_json(output / "failure.json", {"error": error, "swanlab_url": run.url})
        if (output / "summary.json").exists():
            failed_summary = read_json(output / "summary.json")
            failed_summary["completed"] = False
            write_json(output / "summary.json", failed_summary)
        logger.error(error)
        try:
            swanlab.log({"artifacts/failure": swanlab.Text(error)})
            for path in output.glob("gpu-*.log"):
                swanlab.log({"logs/" + path.stem: swanlab.Text(path.read_text(errors="replace")[-60000:])})
            swanlab.finish(state="crashed", error=error[-2000:])
        finally:
            append_history(a.history_file, f"- 失败/中断：`{output / 'failure.json'}`；SwanLab：{run.url}。")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]))
    else:
        main()
