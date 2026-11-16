"""Two real balanced-profile FSDP2 updates on validated cached chat trajectories."""
import argparse
import datetime as dt
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparation.training import pipeline as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-run", type=Path, required=True, help="Completed CPU repair-check summary")
    args = parser.parse_args()
    source = p.read_json(args.validated_run)
    if source.get("status") != "completed" or not source["result"]["aligned_causal_targets"]:
        raise ValueError("A successfully completed cached-target token audit is required")
    cache = source["result"]["cache"]
    if p.file_hash(cache["derived_cache"]) != cache["derived_cache_sha256"]:
        raise ValueError("The audited chat cache has changed")
    values = p.build_env()
    split = p.validate_train_split(values)
    if source["config"]["cache_source"]["source_summary_sha256"] != cache["source_summary_sha256"]:
        raise ValueError("Audit and derived cache have different provenance")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"chat-repair-gpu-{stamp}"
    folder = ROOT / "preparation/logs/training-development" / run_id
    case = {"name": run_id, "output": str(folder), "data": cache["derived_cache"], "steps": 64,
            "batch": int(values["TRAIN_BATCH_SIZE"]), "accum": int(values["GRADIENT_ACCUMULATION_STEPS"]),
            "length": int(values["MAX_LENGTH"]), "budget": int(values["MAX_TOKEN_LEN_PER_GPU"]),
            "sequence_format": "chat", "refine": True, "sp": 1, "checkpoint": True,
            "teacher_cpu_offload": False, "reshard_after_forward": True, "kl_loss_backend": "recompute",
            "use_liger": True, "model_storage_dtype": "bf16", "cuda_memory_fraction": .92, "chunk": 512}
    if (case["batch"], case["accum"]) != (2, 32):
        raise ValueError("GPU check requires the balanced batch2/acc32 profile")

    def action(run):
        import pynvml as nv
        nv.nvmlInit()
        handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(2)]
        stop, telemetry = threading.Event(), []

        def monitor():
            while not stop.is_set():
                telemetry.append([nv.nvmlDeviceGetMemoryInfo(h).used / 2**30 for h in handles])
                stop.wait(.5)

        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        p.write_json(folder / "case.json", case)
        snapshots = folder / "source"
        snapshots.mkdir()
        for relative in ["recipe/opd/dataset/data_utils.py", "recipe/opd/kl_trainer.py", "recipe/opd/config.py",
                         "preparation/pilot/train_probe.py"]:
            shutil.copy2(ROOT / relative, snapshots / relative.replace("/", "__"))
        env = os.environ.copy()
        env.update(WANDB_MODE="disabled", SWANLAB_RELAY="1", PYTHONUNBUFFERED="1",
                   TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        try:
            p.run_command([str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc-per-node=2",
                           str(ROOT / "preparation/pilot/train_probe.py"), "--case", str(folder / "case.json")],
                          folder, run, env)
        finally:
            stop.set()
            worker.join()
            nv.nvmlShutdown()
            p.write_json(folder / "gpu-telemetry.json", telemetry)
        results = [p.read_json(folder / f"result-rank{i}.json") for i in range(2)]
        if not all(r["status"] == "passed" and r["optimizer_steps"] == 2 and len(r["metrics"]) == 64 for r in results):
            raise ValueError("Both ranks must finish all 64 microbatches and 2 optimizer updates")
        for result in results:
            for metric in result["metrics"]:
                assert all(math.isfinite(v) for v in metric.values() if isinstance(v, (float, int)))
        peaks = [max(row[i] for row in telemetry) for i in range(2)]
        for i in range(64):
            run.log({"smoke/microbatch": i, "smoke/kl": results[0]["metrics"][i]["kl_loss"],
                     "smoke/grad_norm": results[0]["metrics"][i]["grad_norm"] or 0})
        report = {"optimizer_steps": 2, "microbatches_per_rank": 64, "sample_visits": 256,
                  "all_values_finite": True, "gpu_peak_gib": peaks,
                  "wall_seconds": time.monotonic() - started, "cache_manifest": cache,
                  "weights_saved": False, "new_generation": False}
        p.write_json(folder / "validation.json", report)
        run.log({"smoke/completed": 1, "smoke/gpu0_peak_gib": peaks[0], "smoke/gpu1_peak_gib": peaks[1]})
        return report

    import fcntl
    started = time.monotonic()
    with (ROOT / "preparation/logs/math-training.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p.experiment("数值校验", run_id, folder, {"case": case, "cache_source": cache,
                     "validated_run": str(args.validated_run), "split": split}, action,
                     name=p.unique_run_name(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d") +
                                            "-数值校验-聊天格式修复-双卡两步", folder.parent))


if __name__ == "__main__":
    main()
