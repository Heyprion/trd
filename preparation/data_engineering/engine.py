"""Persistent independent vLLM replicas, one process per selected GPU."""
from __future__ import annotations

import multiprocessing as mp
import json
import os
from pathlib import Path
import queue
import random
import time
import traceback

from core import valid_codes


def _worker(rank, device, config, incoming, outgoing):
    # CUDA imports must happen after the per-process visibility is set.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    log_path = Path(config["log_dir"]) / f"gpu-{rank}.log"
    with log_path.open("a", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            started = time.monotonic()
            llm = LLM(model=config["model"], dtype="bfloat16", tensor_parallel_size=1,
                      gpu_memory_utilization=config["memory_fraction"],
                      max_model_len=config["max_model_len"], max_num_seqs=config["max_num_seqs"],
                      max_num_batched_tokens=config["batched_tokens"], enable_prefix_caching=True,
                      enforce_eager=config["enforce_eager"], seed=config["seed"] + rank,
                      disable_log_stats=True, trust_remote_code=False)
            tokenizer = llm.get_tokenizer()
            params = {kind: SamplingParams(temperature=0, max_tokens=8,
                       structured_outputs=StructuredOutputsParams(choice=valid_codes(kind)))
                      for kind in ("quality", "pair", "label")}
            outgoing.put({"event": "ready", "rank": rank, "load_seconds": time.monotonic()-started})
            while True:
                message = incoming.get()
                if message is None:
                    break
                job_id, tasks = message
                then = time.monotonic()
                try:
                    rows, prompts, valid = [], [], []
                    # No question or pair is silently truncated to fit context.
                    batches = tokenizer.apply_chat_template([t["messages"] for t in tasks], tokenize=True,
                                add_generation_prompt=True, enable_thinking=False)
                    for item, ids in zip(tasks, batches):
                        if len(ids) + 8 > config["max_model_len"]:
                            rows.append({"id": item["id"], "status": "input_too_long", "text": "", "tokens": 0})
                        else:
                            valid.append(item)
                            prompts.append({"prompt_token_ids": ids})
                    if prompts:
                        answers = llm.generate(prompts, params[tasks[0]["kind"]], use_tqdm=False)
                        for item, answer in zip(valid, answers):
                            output = answer.outputs[0]
                            text = output.text.strip()
                            # A complete member of the finite grammar is valid
                            # even if trailing whitespace consumes the token cap.
                            ok = text in valid_codes(item["kind"])
                            rows.append({"id": item["id"], "status": "ok" if ok else "invalid_output",
                                         "text": text, "tokens": len(output.token_ids),
                                         "prompt_tokens": len(answer.prompt_token_ids or []),
                                         "finish_reason": output.finish_reason})
                    outgoing.put({"event": "result", "rank": rank, "job": job_id, "rows": rows,
                                  "seconds": time.monotonic()-then})
                except Exception:
                    print(traceback.format_exc(), flush=True)
                    outgoing.put({"event": "result", "rank": rank, "job": job_id,
                                  "rows": [{"id": t["id"], "status": "inference_error", "text": "", "tokens": 0} for t in tasks],
                                  "seconds": time.monotonic()-then})
            llm.llm_engine.engine_core.shutdown()
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory()
        except BaseException:
            print(traceback.format_exc(), flush=True)
            outgoing.put({"event": "fatal", "rank": rank, "error": traceback.format_exc()[-6000:]})


class InferencePool:
    def __init__(self, devices, config, progress=None):
        self.context = mp.get_context("spawn")
        self.outgoing = self.context.Queue()
        self.queues, self.processes = [], []
        self.ready = set()
        self.sequence = 0
        self.batch_size = min(128, config["max_num_seqs"])
        self.progress = progress or (lambda *args: None)
        self.stats = {"load_seconds": {}, "batches": 0, "requests": 0, "requests_by_rank": {},
                      "output_tokens": 0, "worker_seconds": 0.0}
        self.config = config
        for rank, device in enumerate(devices):
            incoming = self.context.Queue()
            process = self.context.Process(target=_worker, args=(rank, device, config, incoming, self.outgoing))
            process.start()
            self.queues.append(incoming)
            self.processes.append(process)

    def _receive(self, timeout=1):
        try:
            message = self.outgoing.get(timeout=timeout)
        except queue.Empty:
            dead = [i for i, p in enumerate(self.processes) if p.exitcode is not None]
            if dead:
                raise RuntimeError(f"GPU workers exited: {dead}; inspect gpu-*.log")
            return None
        if message["event"] == "fatal":
            raise RuntimeError(f"GPU {message['rank']} failed: {message['error']}")
        if message["event"] == "ready":
            rank = message["rank"]
            self.ready.add(rank)
            self.stats["load_seconds"][str(rank)] = message["load_seconds"]
            self.progress("gpu_ready", {"rank": rank, "seconds": message["load_seconds"]})
        return message

    def wait_ready(self, deadline):
        last_update = time.monotonic()
        while len(self.ready) != len(self.processes):
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU model loading exceeded the work budget")
            self._receive()
            if time.monotonic()-last_update > 20:
                self.progress("loading", {"ready": len(self.ready), "total": len(self.processes)})
                last_update = time.monotonic()

    def infer(self, tasks, deadline, label="inference", retries=1):
        if not tasks:
            return {}
        self.wait_ready(deadline)
        results = self._run(tasks, deadline, label)
        retry = [t for t in tasks if results[t["id"]]["status"] in ("invalid_output", "inference_error")]
        if retries and retry and time.monotonic() < deadline:
            if any(results[t["id"]]["status"] == "inference_error" for t in retry):
                self.batch_size = max(1, self.batch_size // 2)
            results.update(self._run(retry, deadline, label + "_retry"))
        return results

    def _run(self, tasks, deadline, label):
        if len({t["kind"] for t in tasks}) != 1:
            raise ValueError("An inference call must contain only one task kind")
        results, pending = {}, {}
        cursor = 0
        idle = set(self.ready)
        started = time.monotonic()
        last_update = started
        batch_size = max(1, min(self.batch_size, (len(tasks)+len(self.processes)-1)//len(self.processes)))
        while cursor < len(tasks) or pending:
            while idle and cursor < len(tasks) and time.monotonic() < deadline:
                rank = min(idle)
                idle.remove(rank)
                batch = tasks[cursor:cursor+batch_size]
                cursor += len(batch)
                self.sequence += 1
                pending[self.sequence] = (rank, batch)
                self.queues[rank].put((self.sequence, batch))
            if not pending:
                break
            # Reserve time outside deadline for draining the single batch/rank.
            if time.monotonic() > deadline + 60:
                raise TimeoutError(f"GPU batch did not drain: {label}")
            message = self._receive()
            if message and message["event"] == "result":
                rank, batch = pending.pop(message["job"])
                idle.add(rank)
                for row in message["rows"]:
                    results[row["id"]] = row
                with (Path(self.config["log_dir"]) / "inference-results.jsonl").open("a") as stream:
                    for row in message["rows"]:
                        stream.write(json.dumps({"stage": label, "rank": rank, **row}, ensure_ascii=False) + "\n")
                self.stats["batches"] += 1
                self.stats["requests"] += len(batch)
                self.stats["requests_by_rank"][str(rank)] = self.stats["requests_by_rank"].get(str(rank), 0) + len(batch)
                self.stats["output_tokens"] += sum(r["tokens"] for r in message["rows"])
                self.stats["worker_seconds"] += message["seconds"]
            if time.monotonic()-last_update > 15:
                self.progress(label, {"completed": len(results), "total": len(tasks),
                                     "wall_seconds": time.monotonic()-started})
                last_update = time.monotonic()
        for item in tasks:
            results.setdefault(item["id"], {"id": item["id"], "status": "time_budget", "text": "", "tokens": 0})
        self.progress(label, {"completed": sum(r["status"] != "time_budget" for r in results.values()),
                             "total": len(tasks), "wall_seconds": time.monotonic()-started})
        return results

    def calibrate(self, tasks, deadline):
        """Compare effective batch 64/128 on disjoint samples, without model reload."""
        if len(tasks) < 384 or self.config["max_num_seqs"] < 128 or time.monotonic()+90 >= deadline:
            self.progress("calibration", {"skipped": True, "batch_size": self.batch_size})
            return
        probe = list(tasks)
        random.Random(self.config["seed"]).shuffle(probe)
        self.infer(probe[:16], deadline, "warmup", retries=0)
        rates = {}
        cursor = 16
        for size in (64, 128):
            self.batch_size = size
            sample = probe[cursor:cursor+size*len(self.processes)]
            cursor += len(sample)
            start = time.monotonic()
            got = self.infer(sample, deadline, f"calibration_{size}", retries=0)
            elapsed = time.monotonic()-start
            rates[size] = sum(r["status"] == "ok" for r in got.values()) / max(elapsed, .001)
        self.batch_size = max(rates, key=rates.get)
        self.stats["calibration_requests_per_second"] = rates
        self.stats["chosen_batch_size"] = self.batch_size
        self.progress("calibration", {"batch_size": self.batch_size, "requests_per_second": rates})

    def close(self):
        for incoming in self.queues:
            incoming.put(None)
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
