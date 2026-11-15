#!/usr/bin/env python3
"""One-hour math cleaning + labeling, or either stage independently."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import signal
import time
import traceback
import urllib.request

from core import (DOMAINS, LEVELS, SCHEMA_VERSION, decision, digest, dump_json, initial_filter,
                  load_rows, native_index, near_pairs, prepare_rows, public_row, read_eval,
                  save_table, task)
from engine import InferencePool

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1] if (FOLDER.parents[1] / "recipe/opd").is_dir() else FOLDER
OMNI_REVISION = "40ba231d8f16e29ecd40e6407e2c8640145a8f62"
QUALITY_REASONS = {"R1": "missing_diagram", "R2": "missing_conditions", "R3": "answer_leakage",
                   "R4": "solution_fragment", "R5": "broken_task_or_answer", "R6": "not_established_usable"}


def env(name, default):
    return os.environ.get(name, str(default))


def boolean(value):
    return str(value).lower() in ("true", "1", "yes")


def arguments(default_stage="all", argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "clean", "label"), default=default_stage)
    p.add_argument("--input", type=Path, default=Path(env("INPUT_PATH", ROOT / "data/train/DeepScaleR")))
    p.add_argument("--eval-dir", type=Path, default=Path(env("EVAL_DIR", ROOT / "data/eval")))
    p.add_argument("--model", type=Path, default=Path(env("LABEL_MODEL", ROOT / "models/Qwen3-8B")))
    p.add_argument("--output", type=Path, default=Path(env("OUTPUT_DIR", ROOT / "data/processed/math-v1")))
    p.add_argument("--omni", type=Path, default=Path(env("OMNI_PATH", ROOT / "data/raw/KbsdJames/Omni-MATH/test.jsonl")))
    p.add_argument("--download-omni", type=boolean, default=boolean(env("DOWNLOAD_OMNI", "true")))
    p.add_argument("--gpu-ids", default=env("GPU_IDS", env("CUDA_VISIBLE_DEVICES", "0,1")))
    p.add_argument("--workers", type=int, default=int(env("CPU_WORKERS", 16)))
    p.add_argument("--max-num-seqs", type=int, default=int(env("GPU_MAX_NUM_SEQS", 128)))
    p.add_argument("--batched-tokens", type=int, default=int(env("GPU_BATCHED_TOKENS", 32768)))
    p.add_argument("--memory-fraction", type=float, default=float(env("GPU_MEMORY_FRACTION", .85)))
    p.add_argument("--max-model-len", type=int, default=int(env("MAX_MODEL_LEN", 4096)))
    p.add_argument("--enforce-eager", type=boolean, default=boolean(env("ENFORCE_EAGER", "true")))
    p.add_argument("--total-seconds", type=int, help="Default: combined 3600s; standalone stages split that budget")
    p.add_argument("--clean-seconds", type=int, default=int(env("CLEAN_SECONDS", 1800)))
    p.add_argument("--export-reserve", type=int, default=int(env("EXPORT_RESERVE", 180)))
    p.add_argument("--near-threshold", type=float, default=float(env("NEAR_THRESHOLD", .82)))
    p.add_argument("--eval-near-threshold", type=float, default=float(env("EVAL_NEAR_THRESHOLD", .45)))
    p.add_argument("--max-near-pairs", type=int, default=int(env("MAX_NEAR_PAIRS", 20000)))
    p.add_argument("--seed", type=int, default=int(env("SEED", 20261115)))
    p.add_argument("--max-samples", type=int, help="Smoke-test limit; never used by default")
    p.add_argument("--project", default=env("SWANLAB_PROJECT", "trd-math-opd"))
    p.add_argument("--run-name", default=env("RUN_NAME", "math-data-engineering"))
    p.add_argument("--data-version", default=env("DATA_VERSION", "math-v1"))
    p.add_argument("--history-file", type=Path, default=Path(env("HISTORY_FILE", ROOT / "HISTORY.md")))
    p.add_argument("--resume", action="store_true", help="Reuse completed, fingerprint-checked stage outputs")
    p.add_argument("--dry-run", action="store_true", help="Preflight and print config, without CUDA/SwanLab")
    a = p.parse_args(argv)
    if a.total_seconds is None:
        combined = int(env("TOTAL_SECONDS", 3600))
        a.total_seconds = (combined if a.stage == "all" else min(a.clean_seconds, combined) if a.stage == "clean"
                           else max(60, combined-a.clean_seconds))
    for name in ("input", "eval_dir", "model", "output", "omni", "history_file"):
        setattr(a, name, getattr(a, name).expanduser().resolve())
    if a.total_seconds < 60 or a.export_reserve < 10 or a.export_reserve >= a.total_seconds:
        p.error("total-seconds must be >=60, and 10 <= export-reserve < total-seconds")
    if a.workers < 1 or a.max_num_seqs < 1 or not 0 < a.memory_fraction < 1:
        p.error("Invalid workers, GPU concurrency or memory fraction")
    if not 0 < a.near_threshold <= 1 or not 0 < a.eval_near_threshold <= 1 or a.max_near_pairs < 1:
        p.error("Invalid duplicate thresholds or candidate budget")
    if a.max_samples is not None and a.max_samples < 1:
        p.error("max-samples must be positive")
    return a


def append_history(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(text + "\n")


def download_native(a, logger):
    if a.omni.exists() or not a.download_omni:
        return
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{base}/datasets/KbsdJames/Omni-MATH/resolve/{OMNI_REVISION}/test.jsonl"
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError("Unexpected native metadata size")
        # Pin the bytes verified in the preceding audit, not just the URL.
        if digest(data) != "7c87be8ee41ac7c7a597ef5a5500e84bd2b639a85a06db3da7f69bf9a32ef168":
            raise ValueError("Omni-MATH checksum mismatch")
        a.omni.parent.mkdir(parents=True, exist_ok=True)
        temp = a.omni.with_suffix(".download")
        temp.write_bytes(data)
        temp.replace(a.omni)
        logger.info("Downloaded pinned Omni-MATH metadata")
    except Exception as exc:
        # No native annotations are asserted when source metadata is unavailable.
        # All remaining usable examples receive model labels instead.
        logger.warning("Native metadata unavailable (%s); use automatic labels", type(exc).__name__)


def model_identity(path):
    return {"path": str(path), "weight_metadata": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.glob("*.safetensors"))],
            "config": digest((path / "config.json").read_bytes()),
            "tokenizer": digest((path / "tokenizer_config.json").read_bytes())}


def fingerprint(a, rows, evaluation):
    # Content hash is independent of JSON whitespace or Arrow serialization.
    source_hash = digest(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    code_hash = digest("".join(p.read_text() for p in sorted(FOLDER.glob("*.py"))))
    policy = {"schema": SCHEMA_VERSION, "code_hash": code_hash, "input_hash": source_hash,
              "eval_hash": digest(json.dumps(evaluation, sort_keys=True, ensure_ascii=False)),
              "native_hash": digest(a.omni.read_bytes()) if a.omni.exists() else None,
              "near_threshold": a.near_threshold, "eval_near_threshold": a.eval_near_threshold,
              "max_near_pairs": a.max_near_pairs, "seed": a.seed, "data_version": a.data_version,
              "model_identity": model_identity(a.model), "max_model_len": a.max_model_len,
              "native_mapping": "<=3 easy; <=5 medium; >5 hard"}
    return source_hash, policy, digest(json.dumps(policy, sort_keys=True))


def clean_stage(a, rows, evaluation, native, source_hash, pool, deadline, progress):
    then = time.monotonic()
    prepared = prepare_rows(rows, source_hash, a.data_version, a.workers)
    active, decisions, members = initial_filter(prepared, evaluation, native)
    progress("cpu_filter", {"input_rows": len(rows), "representatives": len(active), "excluded": len(decisions)})
    pairs, overflow, pair_stats = near_pairs(active, evaluation, a.workers, a.near_threshold,
                                           a.eval_near_threshold, a.max_near_pairs)
    progress("near_candidates", pair_stats)
    rejected = {active[i]["raw_id"]: "near_candidate_budget" for i in overflow}
    pair_tasks = []
    for index, pair in enumerate(pairs):
        left = active[pair["left"]]
        right = evaluation[pair["right"]] if pair["eval"] else active[pair["right"]]
        pair_tasks.append(task("pair", str(index), {"left": left["problem"], "right": right["problem"]}))
    answers = pool.infer(pair_tasks, deadline, "near_pair_judgment")
    parent = list(range(len(active)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for index, pair in enumerate(pairs):
        result = answers[str(index)]
        code = result["text"] if result["status"] == "ok" else "R"
        left = active[pair["left"]]
        if code == "D":
            continue
        technical_reason = "near_pair_" + result["status"] if result["status"] != "ok" else None
        if pair["eval"]:
            rejected[left["raw_id"]] = technical_reason or ("eval_near_overlap" if code == "S" else "eval_near_not_resolved")
        else:
            right = active[pair["right"]]
            if code == "S" and left["_answer_key"] == right["_answer_key"]:
                parent[find(pair["right"])] = find(pair["left"])
            else:
                reason = technical_reason or ("near_answer_conflict" if code == "S" else "near_pair_not_resolved")
                rejected[left["raw_id"]] = reason
                rejected[right["raw_id"]] = reason
    clusters = {}
    for i, row in enumerate(active):
        clusters.setdefault(find(i), []).append(row)
    survivors = []
    for cluster in clusters.values():
        cluster.sort(key=lambda r: r["raw_index"])
        representative = cluster[0]
        group_id = representative["sample_id"]
        reason = next((rejected[r["raw_id"]] for r in cluster if r["raw_id"] in rejected), None)
        # Native labels must also agree after merging paraphrases.
        native_signatures = {(r["native_domain"], r["native_difficulty"]) for r in cluster if r["native_level"]}
        if len(native_signatures) > 1:
            reason = "native_label_conflict"
        if not representative["native_level"] and native_signatures:
            source = next(r for r in cluster if r["native_level"])
            for key in ("native_domain", "native_level", "native_difficulty", "native_source"):
                representative[key] = source[key]
        all_members = [r for item in cluster for r in members[item["raw_id"]]]
        for row in all_members:
            row["duplicate_group_id"] = group_id
            if reason:
                decisions[row["raw_id"]] = decision(row, "excluded", reason)
            elif row["raw_id"] != representative["raw_id"]:
                decisions[row["raw_id"]] = decision(row, "excluded", "near_or_exact_duplicate", representative["raw_id"])
        if not reason:
            survivors.append(representative)
    quality_tasks = [task("quality", r["raw_id"], {"problem": r["problem"], "gold_answer": r["answer"]}) for r in survivors]
    group_members = {}
    for item, _ in prepared:
        group_members.setdefault(item.get("duplicate_group_id"), []).append(item)
    random.Random(a.seed).shuffle(quality_tasks)
    pool.calibrate(quality_tasks, deadline)
    judged = pool.infer(quality_tasks, deadline, "content_cleaning")
    usable = []
    for row in survivors:
        result = judged[row["raw_id"]]
        if result["status"] == "ok" and result["text"] == "A":
            usable.append(public_row(row))
            decisions[row["raw_id"]] = decision(row, "usable", "passed_cleaning")
        else:
            reason = QUALITY_REASONS.get(result["text"], "quality_" + result["status"])
            # Propagate rejection to the complete duplicate group.
            group = row["duplicate_group_id"]
            for item in group_members[group]:
                decisions[item["raw_id"]] = decision(item, "excluded", reason)
    if len(decisions) != len(rows):
        raise AssertionError("Each input must have exactly one cleaning decision")
    progress("clean_complete", {"usable": len(usable), "excluded": len(rows)-len(usable), "seconds": time.monotonic()-then})
    return usable, decisions, {**pair_stats, "seconds": time.monotonic()-then,
                               "quality_output_tokens": sum(r["tokens"] for r in judged.values())}


def label_stage(a, rows, decisions, pool, deadline, progress):
    todo, labeled = [], []
    for original in rows:
        row = dict(original)
        if row.get("native_domain") in DOMAINS and row.get("native_level") in LEVELS:
            row.update(domain=DOMAINS[row["native_domain"]], difficulty=LEVELS[row["native_level"]], label_source="native")
            labeled.append(row)
        else:
            todo.append(row)
    tasks = [task("label", r["raw_id"], {"problem": r["problem"]}) for r in todo]
    random.Random(a.seed).shuffle(tasks)
    results = pool.infer(tasks, deadline, "difficulty_labeling")
    for row in todo:
        result = results[row["raw_id"]]
        if result["status"] == "ok":
            domain, level = result["text"].split(",")
            row.update(domain=domain, difficulty=level, label_source="qwen3_8b_rubric")
            labeled.append(row)
        else:
            decisions[row["raw_id"]] = decision(row, "excluded", "label_" + result["status"])
    labeled.sort(key=lambda r: r["raw_index"])
    for row in labeled:
        decisions[row["raw_id"]] = decision(row, "usable", "passed_cleaning_and_labeling")
    progress("label_complete", {"usable": len(labeled), "native": sum(r["label_source"] == "native" for r in labeled)})
    return labeled, decisions


def export(a, folder, rows, decisions, manifest, final=False):
    ordered = sorted(decisions.values(), key=lambda r: r["raw_index"])
    if len({r["raw_id"] for r in ordered}) != manifest["input_rows"]:
        raise AssertionError("Incomplete or duplicated input decisions")
    if {r["status"] for r in ordered} - {"usable", "excluded"}:
        raise AssertionError("Invalid final cleaning state")
    accepted_ids = {r["raw_id"] for r in ordered if r["status"] == "usable"}
    if accepted_ids != {r["raw_id"] for r in rows} or len(accepted_ids) != len(rows):
        raise AssertionError("Usable rows and decisions disagree")
    if len({r["duplicate_group_id"] for r in rows}) != len(rows):
        raise AssertionError("Duplicate group appears twice in usable data")
    if final:
        for row in rows:
            if row.get("domain") not in DOMAINS.values() or row.get("difficulty") not in LEVELS.values():
                raise AssertionError("Every usable row requires valid domain and three-tier difficulty")
    folder.mkdir(parents=True, exist_ok=True)
    save_table(folder / "usable.parquet", rows, ("raw_id", "problem", "answer", "solution", "domain", "difficulty"))
    save_table(folder / "decisions.parquet", ordered)
    excluded = [r for r in ordered if r["status"] == "excluded"]
    save_table(folder / "excluded.parquet", excluded)
    stats = {"input_rows": manifest["input_rows"], "usable_rows": len(rows), "excluded_rows": len(excluded),
             "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
             "technical_exclusions": sum(any(k in r["reason"] for k in ("time_budget", "inference_error", "invalid_output", "input_too_long", "candidate_budget")) for r in excluded)}
    if final:
        stats.update(difficulty_counts=dict(Counter(r["difficulty"] for r in rows)),
                     domain_counts=dict(Counter(r["domain"] for r in rows)), label_source_counts=dict(Counter(r["label_source"] for r in rows)))
        if rows:
            from datasets import Dataset
            Dataset.from_list(rows).save_to_disk(str(folder / "dataset"))
    manifest = {**manifest, "completed": True, "stats": stats}
    dump_json(folder / "manifest.json", manifest)  # Completion marker is written last.
    return stats


def main(default_stage="all", argv=None):
    started = time.monotonic()
    a = arguments(default_stage, argv)
    if not (a.model / "config.json").is_file() or not (a.model / "tokenizer_config.json").is_file():
        raise SystemExit(f"Local Qwen3 model is required at {a.model}; configure LABEL_MODEL")
    if a.stage != "label" and not a.input.exists():
        raise SystemExit(f"Input does not exist: {a.input}")
    if not a.eval_dir.is_dir():
        raise SystemExit(f"Evaluation directory does not exist: {a.eval_dir}")
    devices = [s.strip() for s in a.gpu_ids.split(",") if s.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise SystemExit("GPU_IDS must contain distinct visible GPU IDs or UUIDs")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if a.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    import fcntl
    a.output.mkdir(parents=True, exist_ok=True)
    output_lock = (a.output / ".pipeline.lock").open("a")
    try:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another pipeline is writing {a.output}")
    marker = a.output / ("final" if a.stage in ("label", "all") else "clean") / "manifest.json"
    if marker.exists() and not a.resume:
        raise SystemExit(f"Completed output exists: {marker}; choose a new OUTPUT_DIR or use --resume")
    log_dir = a.output / "logs" / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + a.stage)
    log_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("math_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / "pipeline.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # SwanLab 0.10.5's nested Settings interprets this legacy flat env var as
    # JSON. We already captured its plain project name in a.project.
    os.environ.pop("SWANLAB_PROJECT", None)
    import swanlab
    run = swanlab.init(project=a.project, name=f"{a.run_name}-{a.stage}", mode="online", on_online_error="raise",
                       log_dir=str(log_dir / "swanlab"), job_type="data-engineering", config=config)
    dump_json(log_dir / "swanlab.json", {"url": run.url, "id": run.id})
    append_history(a.history_file, f"\n## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}：数据工程 {a.stage}\n\n- SwanLab：{run.url}；配置及日志：`{log_dir}`；输出：`{a.output}`。")
    pool = None
    def timed_out(signum, frame):
        raise TimeoutError("The configured end-to-end time budget expired")
    previous_alarm = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(max(1, int(a.total_seconds - (time.monotonic()-started))))

    def progress(event, values):
        logger.info("%s %s", event, json.dumps(values, ensure_ascii=False))
        numeric = {f"{event}/{k}": v for k, v in values.items() if isinstance(v, (int, float))}
        if numeric:
            swanlab.log(numeric)

    try:
        download_native(a, logger)
        evaluation = read_eval(a.eval_dir)
        clean_marker = a.output / "clean/manifest.json"
        if a.stage == "label":
            if not clean_marker.exists():
                raise ValueError("Run clean.py first, or use run.sh for both stages")
            clean_manifest = json.loads(clean_marker.read_text())
            rows = load_rows(a.output / "clean/usable.parquet")
            decisions = {r["raw_id"]: r for r in load_rows(a.output / "clean/decisions.parquet")}
            if not clean_manifest.get("completed"):
                raise ValueError("Cleaning did not complete")
            # The source input path need not remain available for standalone stage 2.
            config["clean_manifest_hash"] = digest(clean_marker.read_bytes())
            manifest = {**clean_manifest, "swanlab_url": run.url}
            label_config = {"clean_manifest_hash": config["clean_manifest_hash"], "model": model_identity(a.model),
                            "code": digest("".join(p.read_text() for p in sorted(FOLDER.glob("*.py")))),
                            "max_model_len": a.max_model_len, "seed": a.seed}
            manifest["label_config_hash"] = digest(json.dumps(label_config, sort_keys=True))
            if marker.exists() and a.resume:
                previous = json.loads(marker.read_text())
                if previous.get("label_config_hash") != manifest["label_config_hash"]:
                    raise ValueError("Final labeling output fingerprint does not match this configuration")
                swanlab.log({"resume/reused_completed": 1, "artifacts/summary": swanlab.Text(json.dumps(previous["stats"]))})
                swanlab.finish()
                append_history(a.history_file, f"- 已核对并复用完成分层产物：{marker}；未运行GPU。")
                return
            clean_needed = False
        else:
            source_rows = load_rows(a.input)
            if a.max_samples:
                source_rows = source_rows[:a.max_samples]
            if not source_rows:
                raise ValueError("Training input is empty")
            source_hash, policy, signature = fingerprint(a, source_rows, evaluation)
            manifest = {"schema_version": SCHEMA_VERSION, "input_rows": len(source_rows), "input_hash": source_hash,
                        "policy": policy, "policy_hash": signature, "swanlab_url": run.url}
            dump_json(log_dir / "config.json", {**config, "policy": policy})
            if marker.exists() and a.resume:
                previous = json.loads(marker.read_text())
                if previous.get("policy_hash") != signature:
                    raise ValueError("Completed output fingerprint does not match current input/model/policy")
                logger.info("Verified completed output; no GPU work required: %s", marker)
                swanlab.log({"resume/reused_completed": 1, "artifacts/summary": swanlab.Text(json.dumps(previous["stats"]))})
                swanlab.finish()
                append_history(a.history_file, f"- 已核对并复用完成产物：{marker}；未运行GPU。")
                return
            clean_needed = True
            if clean_marker.exists():
                if not a.resume:
                    raise ValueError("Cleaning output already exists; use --resume after verifying configuration")
                previous = json.loads(clean_marker.read_text())
                if previous.get("policy_hash") != signature:
                    raise ValueError("Cleaning checkpoint does not match current input/model/policy")
                rows = load_rows(a.output / "clean/usable.parquet")
                decisions = {r["raw_id"]: r for r in load_rows(a.output / "clean/decisions.parquet")}
                clean_needed = False
        work_deadline = started + a.total_seconds - a.export_reserve
        engine_config = {"model": str(a.model), "log_dir": str(log_dir), "memory_fraction": a.memory_fraction,
                         "max_model_len": a.max_model_len, "max_num_seqs": a.max_num_seqs,
                         "batched_tokens": a.batched_tokens, "enforce_eager": a.enforce_eager, "seed": a.seed}
        # Start loading now; CPU preparation continues concurrently.
        pool = InferencePool(devices, engine_config, progress)
        if clean_needed:
            deadline = min(work_deadline, started + a.clean_seconds) if a.stage == "all" else work_deadline
            rows, decisions, detail = clean_stage(a, source_rows, evaluation, native_index(a.omni), source_hash, pool, deadline, progress)
            manifest["clean_detail"] = detail
            clean_stats = export(a, a.output / "clean", rows, decisions, manifest)
            swanlab.log({"artifacts/clean_summary": swanlab.Text(json.dumps(clean_stats, ensure_ascii=False))})
        if a.stage in ("all", "label"):
            label_config = {"clean_manifest_hash": digest(clean_marker.read_bytes()), "model": model_identity(a.model),
                            "code": digest("".join(p.read_text() for p in sorted(FOLDER.glob("*.py")))),
                            "max_model_len": a.max_model_len, "seed": a.seed}
            manifest["label_config_hash"] = digest(json.dumps(label_config, sort_keys=True))
            rows, decisions = label_stage(a, rows, decisions, pool, work_deadline, progress)
            stats = export(a, a.output / "final", rows, decisions, manifest, final=True)
        else:
            stats = clean_stats
        stats.update(wall_seconds=time.monotonic()-started, gpu=pool.stats, swanlab_url=run.url)
        dump_json(log_dir / "summary.json", stats)
        dump_json(a.output / ("summary.json" if a.stage != "clean" else "clean-summary.json"), stats)
        swanlab.log({"result/usable_rows": stats["usable_rows"], "result/excluded_rows": stats["excluded_rows"],
                     "result/technical_exclusions": stats["technical_exclusions"], "result/wall_seconds": stats["wall_seconds"],
                     "artifacts/summary": swanlab.Text(json.dumps(stats, ensure_ascii=False)),
                     "artifacts/config": swanlab.Text(json.dumps(config, ensure_ascii=False))})
        for file in log_dir.glob("gpu-*.log"):
            swanlab.log({"logs/" + file.stem: swanlab.Text(file.read_text(errors="replace")[-30000:])})
        swanlab.log({"logs/pipeline": swanlab.Text((log_dir / "pipeline.log").read_text()[-30000:])})
        if not stats["usable_rows"]:
            raise RuntimeError("No usable rows; see exclusion reasons and GPU logs. Do not start training.")
        append_history(a.history_file, f"- 完成：输入{stats['input_rows']}，可用{stats['usable_rows']}，排除{stats['excluded_rows']}，技术排除{stats['technical_exclusions']}，耗时{stats['wall_seconds']:.1f}s；汇总`{log_dir / 'summary.json'}`。")
        swanlab.finish()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except BaseException:
        error = traceback.format_exc()
        logger.error(error)
        dump_json(log_dir / "failure.json", {"error": error, "swanlab_url": run.url})
        swanlab.log({"artifacts/error": swanlab.Text(error)})
        swanlab.finish(state="crashed")
        append_history(a.history_file, f"- 运行失败，详见`{log_dir / 'failure.json'}`；SwanLab：{run.url}。")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if pool:
            pool.close()


if __name__ == "__main__":
    main()
