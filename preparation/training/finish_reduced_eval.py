#!/usr/bin/env python3
"""Finish reporting a live legacy evaluation after its requested final benchmark."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recipe.math_evaluation.runtime_control import read_control  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def process_alive(pid, created_at):
    import psutil

    try:
        proc = psutil.Process(pid)
        return abs(proc.create_time() - created_at) < .01 and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def selected_metrics(folder, datasets, model_name, expected_rows):
    import pyarrow.parquet as pq

    entry = read_json(folder / "results.json")[model_name]
    metrics = {}
    for name in datasets:
        table = pq.read_table(folder / "generations" / f"{name}_pass16_generation.parquet")
        if len(table) != expected_rows[name] or any(len(r) != 16 for r in table["responses"].to_pylist()):
            raise ValueError(f"Incomplete {name} pass@16 responses")
        for label, key in (("pass@16", "pass16"), ("avg@16", "avg_pass1"), ("pass@8", "pass8")):
            value = float(entry[f"{name}_{key}_generation_pass_16"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid result {name}/{label}")
            metrics[f"eval/{name}/{label}"] = value
    return metrics


def finalize(folder, control):
    original = read_json(folder / "summary.json")
    write_json(folder / "summary-before-scope-finalization.json", original)
    model_name = original["run_id"] + "-distilled"
    model = original["config"]["model_path"]
    datasets = control["datasets"]
    subprocess.run([sys.executable, str(ROOT / "recipe/math_evaluation/compute_pass_at_k_from_gen.py"),
                    "--gen_dir", str(folder / "generations"), "--results_file", str(folder / "results.json"),
                    "--model_name", model_name, "--model_path", model, "--datasets", ",".join(datasets)],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "recipe/math_evaluation/results_json_to_csv.py"),
                    "--results_file", str(folder / "results.json"), "--output_file", str(folder / "results.csv")],
                   cwd=ROOT, check=True)
    expected = {name: original["config"]["datasets"][name]["rows"] for name in datasets}
    metrics = selected_metrics(folder, datasets, model_name, expected)
    revised = dict(original)
    revised.pop("error", None)
    revised["config"] = dict(original["config"])
    previous = list(original["config"]["datasets"])
    revised["config"]["original_datasets"] = previous
    revised["config"]["datasets"] = {name: original["config"]["datasets"][name] for name in datasets}
    revised["config"]["generation"] = {**original["config"].get("generation", {}),
                                        "EVAL_DATASETS": ",".join(datasets)}
    control_path = folder / "evaluation-control.json"
    if not control_path.is_file():
        control_path = folder.parent / "evaluation-control.json"
    revised["scope_change"] = {"reason": control.get("reason", "user_requested_scope_reduction"), "datasets": datasets,
                               "skipped_datasets": [name for name in previous if name not in datasets],
                               "request_file": str(control_path)}
    import swanlab

    os.environ.pop("SWANLAB_PROJECT", None)
    run_id = original["swanlab_url"].rstrip("/").split("/")[-1]
    # Resume this evaluation's existing record after its former owner exits.
    # The model is never reloaded or evaluated by this reporting process.
    run = swanlab.init(project="trd-math-opd", id=run_id, resume="must", mode="online",
                       on_online_error="raise", log_dir=str(folder / "swanlab-scope-finalize"),
                       config=revised["config"], reinit=True)
    try:
        swanlab.config.update({"scope_change": revised["scope_change"]})
        run.log({**metrics, "pipeline/completed": 1, "eval/selected_dataset_count": len(datasets)})
        run.finish()
    except BaseException:
        run.finish(state="crashed", error=traceback.format_exc())
        raise
    revised.update(status="completed", completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   result={"model_path": model, "metrics": metrics, "results_file": str(folder / "results.json")})
    write_json(folder / "summary.json", revised)
    with (ROOT / "HISTORY.md").open("a") as history:
        fcntl.flock(history, fcntl.LOCK_EX)
        history.write(f"\n- 用户缩减后的{'/'.join(datasets)}评测完成：{original['swanlab_url']}；"
                      f"跳过{revised['scope_change']['skipped_datasets']}；汇总`{folder / 'summary.json'}`；"
                      f"指标{json.dumps(metrics)}。\n")
    return metrics


def finish_after_controller(folder, control, pid, created_at):
    """Finalize after the old owner's natural exit; never signal training."""
    status_path = folder.parent / "evaluation-scope-status.json"
    write_json(status_path, {"status": "armed", "datasets": control["datasets"],
                            "watcher_pid": os.getpid(), "controller_pid": pid})
    try:
        while process_alive(pid, created_at):
            time.sleep(2)
        if read_json(folder.parent / "train/summary.json").get("status") != "completed":
            raise RuntimeError("Training did not complete; evaluation cannot be finalized")
        if not (folder / "summary.json").is_file():
            raise RuntimeError("Controller exited before creating an evaluation record")
        write_json(status_path, {"status": "finalizing", "datasets": control["datasets"]})
        metrics = finalize(folder, control)
        write_json(status_path, {"status": "completed", "datasets": control["datasets"], "metrics": metrics})
        return metrics
    except BaseException:
        write_json(status_path, {"status": "failed", "datasets": control["datasets"], "error": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--wait-for-controller", type=int)
    parser.add_argument("--controller-created-at", type=float)
    args = parser.parse_args()
    folder = args.eval_dir.resolve()
    control = read_control(folder / "results.json")
    if args.wait_for_controller:
        if not control or args.controller_created_at is None:
            raise ValueError("A scope request and exact controller creation time are required")
        finish_after_controller(folder, control, args.wait_for_controller, args.controller_created_at)
        return
    if not control or not control.get("legacy_suite_pid"):
        raise ValueError("A PID-specific legacy boundary control is required")
    status_path = folder / "scope-finalizer-status.json"
    write_json(status_path, {"status": "armed", "datasets": control["datasets"], "pid": os.getpid()})
    while not (folder / "scope-boundary.json").is_file():
        if not process_alive(control["legacy_suite_pid"], control["legacy_suite_created_at"]):
            raise RuntimeError("Evaluation suite exited before the requested boundary was recorded")
        time.sleep(1)
    # Let the old owner finish its finally blocks and close the online record.
    while process_alive(control["pipeline_pid"], control["pipeline_created_at"]):
        time.sleep(1)
    write_json(status_path, {"status": "finalizing", "datasets": control["datasets"], "pid": os.getpid()})
    metrics = finalize(folder, control)
    write_json(status_path, {"status": "completed", "datasets": control["datasets"], "metrics": metrics})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
