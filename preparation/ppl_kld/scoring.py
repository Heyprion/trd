"""Response-only teacher-forced scoring; never materialize sequence × vocabulary logits."""
from __future__ import annotations

import math
from collections import defaultdict

import torch


SUM_KEYS = ("student_nll_sum", "teacher_nll_sum", "forward_kl_sum", "reverse_kl_sum")
PROMPT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
PROTOCOL = {
    "version": 1,
    "context": "same problem and reference prefix for teacher and student",
    "prompt": "problem + SPACE + instruction_following + NEWLINE; encode separately from response",
    "instruction": PROMPT_INSTRUCTION,
    "special_tokens": False,
    "eos": "only tokens present in original solution; no appended EOS or chat tags",
    "mask": "response only; causal shift includes first and last reference token",
    "temperature": 1.0,
    "vocabulary": "full model output vocabulary; no top-k or clipping",
    "reduction": "global sum divided by global valid response token count",
    "nll_kl_dtype": "FP32 log_softmax and products, FP64 sums",
    "overflow": "error; never silently truncate or drop rows",
}


def encode_row(row, tokenizer, max_length):
    if not isinstance(row.get("problem"), str) or not row["problem"].strip():
        raise ValueError(f"Empty problem: {row.get('sample_id')}")
    if not isinstance(row.get("solution"), str) or not row["solution"].strip():
        raise ValueError(f"Empty reference: {row.get('sample_id')}")
    prompt = row["problem"] + " " + PROMPT_INSTRUCTION + "\n"
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    response = tokenizer.encode(row["solution"], add_special_tokens=False)
    if not prefix or not response:
        raise ValueError(f"No prompt/reference tokens: {row.get('sample_id')}")
    if len(prefix) + len(response) > max_length:
        raise ValueError(f"{row['sample_id']}: {len(prefix)+len(response)} > max_length={max_length}; no truncation allowed")
    return {"sample_id": row["sample_id"], "raw_id": row["raw_id"],
            "duplicate_group_id": row["duplicate_group_id"], "domain": row["domain"],
            "difficulty": row["difficulty"], "input_ids": prefix + response,
            "prompt_tokens": len(prefix), "response_tokens": len(response)}


def make_batches(rows, batch_size, token_budget):
    batches, batch = [], []
    for row in sorted(rows, key=lambda r: (len(r["input_ids"]), r["sample_id"])):
        if len(row["input_ids"]) > token_budget:
            raise ValueError(f"Single sequence exceeds batch token budget: {row['sample_id']}; increase --batch-tokens")
        if batch and (len(batch) >= batch_size or len(row["input_ids"]) * (len(batch) + 1) > token_budget):
            batches.append(batch)
            batch = []
        batch.append(row)
    if batch:
        batches.append(batch)
    return batches


def core_model(model):
    # PeftModel wraps a Qwen3ForCausalLM whose transformer/head already contain LoRA layers.
    core = model.get_base_model() if hasattr(model, "peft_config") else model
    if core.config.model_type != "qwen3":
        raise ValueError("Hidden-state chunking currently supports Qwen3ForCausalLM only")
    return core


@torch.inference_mode()
def score_batch(student, teacher, rows, pad_token_id, chunk_size):
    """Match ordinary causal-LM logits exactly, but project only supervised positions."""
    device = next(student.parameters()).device
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    # Store the hidden-state positions that predict response token k (position k-1).
    owner, positions, targets = [], [], []
    for i, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[i, :length] = torch.tensor(row["input_ids"], device=device)
        attention[i, :length] = 1
        start = row["prompt_tokens"]
        positions.extend(range(start - 1, length - 1))
        owner.extend([i] * row["response_tokens"])
        targets.extend(row["input_ids"][start:])
    owner = torch.tensor(owner, dtype=torch.long, device=device)
    positions = torch.tensor(positions, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    s_core, t_core = core_model(student), core_model(teacher)
    s_hidden = s_core.model(input_ids=ids, attention_mask=attention, use_cache=False,
                            return_dict=True).last_hidden_state
    t_hidden = t_core.model(input_ids=ids, attention_mask=attention, use_cache=False,
                            return_dict=True).last_hidden_state
    totals = torch.zeros((len(rows), len(SUM_KEYS)), dtype=torch.float64, device=device)
    for begin in range(0, targets.numel(), chunk_size):
        end = min(begin + chunk_size, targets.numel())
        batch_ids, seq_ids = owner[begin:end], positions[begin:end]
        s_logits = s_core.get_output_embeddings()(s_hidden[batch_ids, seq_ids]).float()
        t_logits = t_core.get_output_embeddings()(t_hidden[batch_ids, seq_ids]).float()
        s_logp, t_logp = s_logits.log_softmax(-1), t_logits.log_softmax(-1)
        del s_logits, t_logits
        labels = targets[begin:end, None]
        difference = t_logp - s_logp
        values = torch.stack((
            -s_logp.gather(-1, labels).squeeze(-1).double(),
            -t_logp.gather(-1, labels).squeeze(-1).double(),
            (t_logp.exp() * difference).sum(-1, dtype=torch.float64),
            (s_logp.exp() * -difference).sum(-1, dtype=torch.float64),
        ), dim=-1)
        totals.index_add_(0, batch_ids, values)
        del s_logp, t_logp, difference, values
    if not torch.isfinite(totals).all().item():
        raise FloatingPointError("Non-finite NLL/KL; no result has been accepted")
    result = []
    for row, sums in zip(rows, totals.cpu().tolist()):
        result.append({**{k: row[k] for k in ("sample_id", "raw_id", "duplicate_group_id", "domain", "difficulty",
                                             "prompt_tokens", "response_tokens", "tokens_sha256")},
                       **dict(zip(SUM_KEYS, sums))})
    return result


def aggregate(rows):
    count = sum(r["response_tokens"] for r in rows)
    if count <= 0:
        raise ValueError("No valid response tokens")
    sums = {key: math.fsum(r[key] for r in rows) for key in SUM_KEYS}
    s_nll, t_nll = sums["student_nll_sum"] / count, sums["teacher_nll_sum"] / count
    return {"rows": len(rows), "response_tokens": count, **sums,
            "student_nll": s_nll, "student_ppl": math.exp(s_nll),
            "teacher_nll": t_nll, "teacher_ppl": math.exp(t_nll),
            "forward_kl": sums["forward_kl_sum"] / count,
            "reverse_kl": sums["reverse_kl_sum"] / count}


def grouped_metrics(rows):
    result = {}
    for field in ("difficulty", "domain"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result[field] = {name: aggregate(group) for name, group in sorted(groups.items())}
    lengths = defaultdict(list)
    for row in rows:
        length = row["response_tokens"]
        lengths["<=256" if length <= 256 else "257-512" if length <= 512 else "513-1024" if length <= 1024 else ">1024"].append(row)
    result["response_length"] = {name: aggregate(group) for name, group in sorted(lengths.items())}
    return result


def paired_comparison(baseline, current, repeats=1000, seed=20261115):
    import numpy as np
    before = {r["sample_id"]: r for r in baseline}
    after = {r["sample_id"]: r for r in current}
    if len(before) != len(baseline) or len(after) != len(current) or before.keys() != after.keys():
        raise ValueError("Comparison requires identical unique sample IDs")
    order = sorted(before)
    for key in order:
        if any(before[key][f] != after[key][f] for f in ("tokens_sha256", "response_tokens", "prompt_tokens")):
            raise ValueError(f"Comparison reference/prefix changed: {key}")
    metrics = ("student_nll", "student_ppl", "forward_kl", "reverse_kl")
    a, b = aggregate(baseline), aggregate(current)
    # Resample problems together; every draw recomputes token-weighted sums and exp(NLL).
    def vector(row):
        return [row["response_tokens"], row["student_nll_sum"], row["forward_kl_sum"], row["reverse_kl_sum"]]
    x = np.array([vector(before[k]) for k in order], dtype=np.float64)
    y = np.array([vector(after[k]) for k in order], dtype=np.float64)
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(repeats):
        indices = rng.integers(0, len(order), size=len(order))
        left, right = x[indices].sum(0), y[indices].sum(0)
        ln, rn = left[1:] / left[0], right[1:] / right[0]
        deltas.append([rn[0] - ln[0], np.exp(rn[0]) - np.exp(ln[0]), rn[1] - ln[1], rn[2] - ln[2]])
    bounds = np.percentile(np.array(deltas), [2.5, 97.5], axis=0)
    return {"direction": "current minus baseline; negative means lower", "bootstrap_repeats": repeats,
            "bootstrap_unit": "paired problem", "seed": seed,
            "metrics": {k: {"baseline": a[k], "current": b[k], "delta": b[k] - a[k],
                            "relative_percent": 100 * (b[k] - a[k]) / a[k] if abs(a[k]) > 1e-12 else None,
                            "delta_ci95": bounds[:, i].tolist()} for i, k in enumerate(metrics)}}
