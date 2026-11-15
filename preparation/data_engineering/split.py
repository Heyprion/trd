#!/usr/bin/env python3
"""Freeze a stratified reference validation set and an expandable training pool."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import time
import traceback

from core import DOMAINS, LEVELS, answer_key, digest, dump_json, load_rows, save_table

ROOT = Path(__file__).resolve().parents[2]
STRATA = [(level, domain) for level in LEVELS.values() for domain in DOMAINS.values()]


def quota(size, weights, capacities, minimum=0):
    """Bounded proportional apportionment, with deterministic tie breaking."""
    keys = sorted(weights)
    if size < 0 or size > sum(capacities.get(k, 0) for k in keys):
        raise ValueError("Requested sample exceeds available candidates")
    target = {k: size*weights[k]/sum(weights.values()) for k in keys}
    lower = {k: min(minimum, capacities.get(k, 0)) for k in keys}
    if sum(lower.values()) > size:
        raise ValueError("Requested sample cannot cover all required strata")
    count = {k: min(capacities.get(k, 0), max(lower[k], math.floor(target[k]))) for k in keys}
    while sum(count.values()) > size:
        key = max((k for k in keys if count[k] > lower[k]), key=lambda k: (count[k]-target[k], k))
        count[key] -= 1
    while sum(count.values()) < size:
        key = max((k for k in keys if count[k] < capacities.get(k, 0)), key=lambda k: (target[k]-count[k], k))
        count[key] += 1
    return count


def aligned_size(target, batch):
    if target < 1 or batch < 1:
        raise ValueError("Training target and effective batch must be positive")
    return max(batch, ((target + batch//2)//batch)*batch)


def rank(row, seed, purpose):
    return digest(f"{seed}:{purpose}:{row['raw_id']}")


def last_box(text):
    positions = [(text.rfind(marker), marker) for marker in (r"\boxed", r"\fbox")]
    index, marker = max(positions)
    if index < 0:
        return None
    start = index + len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start+1:end]
    return None


def terminal_answer(text):
    # Remove only an explicit leading multiple-choice annotation, not variables.
    return re.sub(r"^\s*\\(?:textbf|text|mathrm)\{\(?[A-E]\)?\}\s*(?:\\[,;! ]\s*)*", "", text).strip()


def fast_reference(row):
    box = last_box(row["solution"])
    if box is not None and answer_key(terminal_answer(box)) == answer_key(row["answer"]):
        return "last_box_equivalent"
    return None


def symbolic_reference(row):
    """Bounded terminal answer consistency; no proof of reasoning correctness."""
    from math_verify import parse, verify
    box = last_box(row["solution"])
    prediction = r"\boxed{" + terminal_answer(box) + "}" if box is not None else row["solution"][-2048:]
    try:
        gold = parse(r"\boxed{" + row["answer"] + "}", fallback_mode="no_fallback", parsing_timeout=1)
        target = parse(prediction, fallback_mode="no_fallback", parsing_timeout=1)
        return "math_verify_terminal" if gold and target and verify(gold, target, timeout_seconds=1) else None
    except Exception:
        return None


def validation_selection(rows, size, seed, pilot_indices, reference_budget=120):
    all_counts = Counter((r["difficulty"], r["domain"]) for r in rows)
    reference = defaultdict(list)
    for row in rows:
        if row.get("solution", "").strip() and row["raw_index"] not in pilot_indices:
            reference[(row["difficulty"], row["domain"])].append(row)
    by_level = Counter(r["difficulty"] for r in rows)
    capacities = {level: sum(len(reference[(level, domain)]) for domain in DOMAINS.values()) for level in LEVELS.values()}
    level_quotas = quota(size, by_level, capacities)
    desired = {}
    for level, number in level_quotas.items():
        weights = {domain: all_counts[(level, domain)] for domain in DOMAINS.values() if all_counts[(level, domain)]}
        caps = {domain: len(reference[(level, domain)]) for domain in weights}
        desired.update({(level, domain): n for domain, n in quota(number, weights, caps, minimum=1).items()})
    eligible, failed = defaultdict(list), defaultdict(list)
    methods = Counter()
    for cell in STRATA:
        for row in sorted(reference[cell], key=lambda r: rank(r, seed, "validation")):
            method = fast_reference(row)
            if method:
                eligible[cell].append({**row, "reference_check_method": method})
                methods[method] += 1
            else:
                failed[cell].append(row)
    # Symbolic tools are used only where fast candidates cannot fill a quota.
    then, attempts = time.monotonic(), 0
    for cell in STRATA:
        for row in failed[cell]:
            if len(eligible[cell]) >= desired.get(cell, 0):
                break
            if time.monotonic()-then >= reference_budget:
                raise TimeoutError("Reference consistency budget exhausted; split was not frozen")
            attempts += 1
            method = symbolic_reference(row)
            if method:
                eligible[cell].append({**row, "reference_check_method": method})
                methods[method] += 1
    deficits = {"/".join(k): desired[k]-len(eligible[k]) for k in desired if len(eligible[k]) < desired[k]}
    if deficits:
        raise ValueError(f"Insufficient consistent references for fixed strata: {deficits}")
    selected = []
    for cell in STRATA:
        ordered = sorted(eligible[cell], key=lambda r: rank(r, seed, "validation"))
        selected.extend(ordered[:desired.get(cell, 0)])
    selected.sort(key=lambda r: rank(r, seed, "validation_output"))
    return selected, {"nonempty_references_excluding_pilot": sum(map(len, reference.values())),
                      "eligible_checked_references": sum(map(len, eligible.values())),
                      "eligible_check_methods": dict(methods), "symbolic_fallback_attempts": attempts,
                      "selected_check_methods": dict(Counter(r["reference_check_method"] for r in selected)),
                      "validation_stratum_quotas": {"/".join(k): v for k, v in desired.items()}}


def training_order(rows, seed):
    """Interleave fixed per-stratum shuffles; any longer prefix retains old IDs."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["difficulty"], row["domain"])].append(row)
    keys = sorted(groups)
    for key in keys:
        groups[key].sort(key=lambda r: rank(r, seed, "training"))
    counts = {key: 0 for key in keys}
    weights = {key: len(groups[key])/len(rows) for key in keys}
    ordered = []
    for position in range(len(rows)):
        key = max((k for k in keys if counts[k] < len(groups[k])),
                  key=lambda k: (weights[k]*(position+1)-counts[k], k))
        row = dict(groups[key][counts[key]])
        row["sampling_rank"] = position
        ordered.append(row)
        counts[key] += 1
    return ordered


def stats(rows):
    return {"rows": len(rows), "difficulty_counts": dict(Counter(r["difficulty"] for r in rows)),
            "domain_counts": dict(Counter(r["domain"] for r in rows)),
            "stratum_counts": dict(sorted(Counter(r["difficulty"]+"/"+r["domain"] for r in rows).items())),
            "nonempty_references": sum(bool(r.get("solution", "").strip()) for r in rows)}


def check_splits(source, validation, pool, selected, reserve):
    def ids(rows, key):
        values = [r[key] for r in rows]
        if len(set(values)) != len(values):
            raise AssertionError(f"Repeated {key} within a split")
        return set(values)
    for key in ("raw_id", "sample_id", "duplicate_group_id"):
        all_ids = ids(source, key)
        val_ids, pool_ids = ids(validation, key), ids(pool, key)
        train_ids, reserve_ids = ids(selected, key), ids(reserve, key)
        assert not val_ids & pool_ids
        assert val_ids | pool_ids == all_ids
        assert train_ids <= pool_ids
        assert not train_ids & reserve_ids
        assert train_ids | reserve_ids == pool_ids
    assert [r["raw_id"] for r in selected] == [r["raw_id"] for r in pool[:len(selected)]]
    assert all(r.get("solution", "").strip() and r.get("reference_check_method") for r in validation)


def write_split(folder, rows):
    from datasets import Dataset
    folder.mkdir(parents=True, exist_ok=True)
    save_table(folder / "data.parquet", rows)
    dump_json(folder / "ids.json", [r["raw_id"] for r in rows])
    dump_json(folder / "sample_ids.json", [r["sample_id"] for r in rows])
    Dataset.from_list(rows).save_to_disk(str(folder / "dataset"))
    return {"parquet_sha256": digest((folder / "data.parquet").read_bytes()),
            "ids_sha256": digest((folder / "ids.json").read_bytes()),
            "sample_ids_sha256": digest((folder / "sample_ids.json").read_bytes()), **stats(rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/processed/math-v1/final/usable.parquet")
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/math-v1/splits-v1")
    p.add_argument("--validation-size", type=int, default=512)
    p.add_argument("--target-train-size", type=int, default=10000)
    p.add_argument("--global-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--split-version", default="math-v1-splits-v1")
    p.add_argument("--pilot-config", type=Path, default=ROOT / "preparation/logs/pilot/real-rollout/config.json")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen3-4B-Instruct-2507")
    a = p.parse_args(argv)
    count = aligned_size(a.target_train_size, a.global_batch)
    a.input, a.output = a.input.resolve(), a.output.resolve()
    if a.output.exists():
        raise SystemExit(f"Frozen split output already exists: {a.output}; use it without re-sampling")
    pilot = set(range(32))
    if a.pilot_config.exists():
        pilot.update(json.loads(a.pilot_config.read_text())["selected_dataset_indices"])
    config = {"input": str(a.input), "input_sha256": digest(a.input.read_bytes()), "output": str(a.output),
              "validation_size": a.validation_size, "train_target": a.target_train_size, "train_selected": count,
              "effective_batch": a.global_batch, "seed": a.seed, "split_version": a.split_version,
              "pilot_excluded_from_validation": sorted(pilot), "validation_reference_policy": "last box or bounded Math-Verify terminal equivalence",
              "training_order": "deterministic per-stratum hash order, weighted deficit interleave; nested prefixes",
              "code_sha256": digest(Path(__file__).read_bytes())}
    os.environ.pop("SWANLAB_PROJECT", None)
    import swanlab
    logdir = ROOT / "preparation/logs/data-engineering" / ("split-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
    logdir.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(logdir / "split.log")])
    logger = logging.getLogger("split")
    run = swanlab.init(project="trd-math-opd", name=f"math-v1-validation{a.validation_size}-train{count}-split", mode="online",
                       on_online_error="raise", log_dir=str(logdir / "swanlab"), job_type="data-split", config=config)
    dump_json(logdir / "swanlab.json", {"url": run.url})
    dump_json(logdir / "config.json", config)
    with (ROOT / "HISTORY.md").open("a") as f:
        f.write(f"\n- 数据分层划分：{run.url}；配置及日志：`{logdir}`；目标输出：`{a.output}`。\n")
    started = time.monotonic()
    temp = a.output.with_name("." + a.output.name + ".tmp-" + run.id)
    try:
        source = load_rows(a.input)
        if count > len(source)-a.validation_size:
            raise ValueError("Selected training set exceeds remaining training pool")
        if len({r["duplicate_group_id"] for r in source}) != len(source):
            raise ValueError("Source contains repeated duplicate groups; resolve before splitting")
        logger.info("Input rows=%s, validation=%s, selected train=%s", len(source), a.validation_size, count)
        validation, ref_stats = validation_selection(source, a.validation_size, a.seed, pilot)
        held_out = {r["raw_id"] for r in validation}
        pool = training_order([r for r in source if r["raw_id"] not in held_out], a.seed)
        for row in validation + pool:
            row["split_version"] = a.split_version
        selected, reserve = pool[:count], pool[count:]
        check_splits(source, validation, pool, selected, reserve)
        outputs = {}
        for name, rows in (("validation", validation), ("train_pool", pool), (f"train_{count}", selected), ("train_reserve", reserve)):
            logger.info("Writing %s: %s rows", name, len(rows))
            outputs[name] = write_split(temp / name, rows)
        token_stats = None
        if a.tokenizer.is_dir():
            from transformers import AutoTokenizer
            import numpy as np
            tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
            lengths = [len(x) for x in tok([r["solution"] for r in validation], add_special_tokens=False)["input_ids"]]
            token_stats = {"tokenizer": str(a.tokenizer), "response_tokens": sum(lengths), "mean": float(np.mean(lengths)),
                           "median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)), "max": max(lengths)}
        summary = {"completed": True, "config": config, "source_stats": stats(source), "reference_selection": ref_stats,
                   "splits": outputs, "validation_token_stats": token_stats,
                   "nominal_optimizer_steps_one_epoch": count//a.global_batch,
                   "checks": {"unique_ids": True, "validation_train_overlap": 0, "validation_pool_group_overlap": 0,
                              "selected_is_pool_prefix": True, "selected_reserve_overlap": 0, "all_source_rows_accounted": True},
                   "swanlab_url": run.url, "wall_seconds": time.monotonic()-started}
        dump_json(temp / "manifest.json", summary)
        temp.rename(a.output)
        dump_json(logdir / "summary.json", summary)
        swanlab.log({"split/validation": len(validation), "split/train_pool": len(pool), "split/train_selected": count,
                     "split/reserve": len(reserve), "split/overlap": 0, "split/wall_seconds": summary["wall_seconds"],
                     "artifacts/summary": swanlab.Text(json.dumps(summary, ensure_ascii=False)),
                     "logs/split": swanlab.Text((logdir / "split.log").read_text())})
        with (ROOT / "HISTORY.md").open("a") as f:
            f.write(f"- 划分完成：验证{len(validation)}、训练池{len(pool)}、首轮训练{count}、后备{len(reserve)}，按有效batch{a.global_batch}为{count//a.global_batch}步/epoch；零交集检查通过；SwanLab {run.url}。\n")
        swanlab.finish()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        error = traceback.format_exc()
        dump_json(logdir / "failure.json", {"error": error, "url": run.url})
        swanlab.log({"error": swanlab.Text(error)})
        swanlab.finish(state="crashed")
        raise
    finally:
        if temp.exists():
            shutil.rmtree(temp)


if __name__ == "__main__":
    main()
