"""Frozen forward-only KL diagnostics on existing training targets; no generation."""
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparation.training import pipeline as p
from recipe.opd.dataset.data_utils import KLTrainingDataset
from verl.utils.reward_score.math_reward import compute_score

FIRST = ROOT / "preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json"
REPAIR = ROOT / "preparation/logs/math-runs/repair-chat-9984-v1/train/summary.json"


def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True, local_files_only=True)
    return model.to("cuda:0").eval()


@torch.inference_mode()
def hidden(model, item, side):
    ids = item[side + "_input_ids"].to("cuda:0").unsqueeze(0)
    indices = item[side + "_loss_mask"].bool().nonzero().flatten().to("cuda:0")
    output = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0, indices]
    return output.cpu()


@torch.inference_mode()
def score_positions(student_head, sh, teacher_head, th, labels, chunk=128):
    stats = []
    for start in range(0, len(labels), chunk):
        end = min(start + chunk, len(labels))
        slog = student_head(sh[start:end].to("cuda:0")).float().log_softmax(-1)
        tlog = teacher_head(th[start:end].to("cuda:0")).float().log_softmax(-1)
        target = labels[start:end].to("cuda:0").unsqueeze(1)
        ps, pt = slog.exp(), tlog.exp()
        vals = torch.stack([-slog.gather(1, target).squeeze(1), -tlog.gather(1, target).squeeze(1),
                            (pt * (tlog - slog)).sum(-1), (ps - pt).abs().sum(-1)], dim=-1)
        stats.append(vals.double().cpu())
    result = torch.cat(stats).numpy()
    assert np.isfinite(result).all() and result[:, 2].min() > -1e-4
    return result


def segments(labels, tokenizer):
    close = tokenizer.convert_tokens_to_ids("</think>")
    ends = np.flatnonzero(labels.numpy() == close)
    boundary = int(ends[0]) + 1 if len(ends) else len(labels)
    eos = labels.numpy() == tokenizer.eos_token_id
    thought = np.arange(len(labels)) < boundary
    return {"thinking": thought & ~eos, "final_solution": ~thought & ~eos, "eos": eos,
            "first8": np.arange(len(labels)) < 8, "all": np.ones(len(labels), dtype=bool)}


def summarize(values, masks):
    result = {}
    for name, mask in masks.items():
        part = values[mask]
        if len(part):
            result[name] = {"tokens": len(part), "student_nll_sum": float(part[:, 0].sum()),
                "teacher_nll_sum": float(part[:, 1].sum()), "fkl_sum": float(part[:, 2].sum()),
                "logit_gradient_l1_sum": float(part[:, 3].sum()),
                "student_ppl": float(np.exp(part[:, 0].mean())), "teacher_ppl": float(np.exp(part[:, 1].mean())),
                "fkl_mean": float(part[:, 2].mean())}
    fkl = values[:, 2]
    for frac in [.01, .05]:
        n = max(1, int(np.ceil(len(fkl) * frac)))
        result[f"top{int(frac * 100)}_percent_kl_mass"] = float(np.sort(fkl)[-n:].sum() / fkl.sum())
    return result


def main():
    first, repair = p.read_json(FIRST), p.read_json(REPAIR)
    old_cache = Path(first["config"]["training"]["TRAINING_DATA_PATH"])
    new_cache = Path(repair["result"]["cache_reuse"]["derived_cache"])
    source_rows = pq.read_table(new_cache).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-4B-Instruct-2507", local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-8B", local_files_only=True)
    rng = np.random.default_rng(20261116)
    groups = {key: [] for key in ["correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong"]}
    for i in rng.permutation(len(source_rows)):
        row = source_rows[int(i)]
        gold = row["reward_model"]["ground_truth"].strip()
        if not gold.lstrip("+-").isdigit():
            continue
        extra, response = row["extra_info"], row["responses"][0]
        a = bool(compute_score(extra["initial_response"], gold))
        b = bool(compute_score(response, gold))
        key = ("correct" if a else "wrong") + "_" + ("correct" if b else "wrong")
        # A bounded, deliberately balanced diagnostic, not a population estimate.
        if len(groups[key]) < 8 and len(response) < 20000 and len(extra["initial_response"]) < 30000:
            groups[key].append(int(i))
        if all(len(v) == 8 for v in groups.values()):
            break
    assert all(len(v) == 8 for v in groups.values())
    selected = [(i, key) for key, indices in groups.items() for i in indices]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = "target-kl-probe-" + stamp
    folder = ROOT / "preparation/logs/distillation-diagnostics" / run_id
    variants = {"base": ROOT / "models/Qwen3-4B-Instruct-2507",
                "first": Path(first["result"]["model_path"]), "repair": Path(repair["result"]["model_path"])}

    def action(run):
        import swanlab
        torch.cuda.set_per_process_memory_fraction(.85, 0)
        teacher = load_model(ROOT / "models/Qwen3-8B")
        datasets = {
            "chat": KLTrainingDataset(str(new_cache), tokenizer, teacher_tokenizer=teacher_tokenizer,
                kl_type="forward", max_length=18432, use_initial_response=True, prompt_truncation=True,
                distill_mode="opd", sequence_format="chat"),
            "plain": KLTrainingDataset(str(old_cache), tokenizer, kl_type="forward", max_length=18432,
                use_initial_response=True, prompt_truncation=True, distill_mode="opd", sequence_format="plain")}
        items, teacher_hidden = {}, {}
        for fmt, dataset in datasets.items():
            items[fmt], teacher_hidden[fmt] = [], []
            for idx, group in selected:
                item = dataset[idx]
                si = item["student_loss_mask"].bool().nonzero().flatten()
                ti = item["teacher_loss_mask"].bool().nonzero().flatten()
                labels = item["student_input_ids"][si + 1]
                assert torch.equal(labels, item["teacher_input_ids"][ti + 1])
                items[fmt].append((item, labels))
                teacher_hidden[fmt].append(hidden(teacher, item, "teacher"))
        observations = []
        for name, path in variants.items():
            model = load_model(path)
            for fmt in (["chat", "plain"] if name in {"base", "first"} else ["chat"]):
                for j, ((idx, group), (item, labels), th) in enumerate(zip(selected, items[fmt], teacher_hidden[fmt], strict=True)):
                    sh = hidden(model, item, "student")
                    values = score_positions(model.get_output_embeddings(), sh, teacher.get_output_embeddings(), th, labels)
                    record = {"model": name, "format": fmt, "training_row_index": idx, "group": group,
                              "segments": summarize(values, segments(labels, tokenizer)),
                              "first_token_id": int(labels[0]), "first_token_logprob_student": float(-values[0, 0]),
                              "first_token_logprob_teacher": float(-values[0, 1])}
                    observations.append(record)
                    if (j + 1) % 8 == 0:
                        print(name, fmt, j + 1, "/32", flush=True)
                run.log({f"progress/{name}/{fmt}": 32})
            del model
            torch.cuda.empty_cache()
        result = {"swanlab_url": run.url, "groups": {k: len(v) for k, v in groups.items()},
                  "population_representative": False, "variants": {}}
        for name, fmt in [("base", "chat"), ("first", "chat"), ("repair", "chat"), ("base", "plain"), ("first", "plain")]:
            rows = [r for r in observations if r["model"] == name and r["format"] == fmt]
            summary = {"segments": {}, "groups": {}}
            for segment in ["thinking", "final_solution", "eos", "first8", "all"]:
                ss = [r["segments"][segment] for r in rows if segment in r["segments"]]
                if not ss:
                    continue
                totals = {key: sum(s[key] for s in ss) for key in
                    ["tokens", "student_nll_sum", "teacher_nll_sum", "fkl_sum", "logit_gradient_l1_sum"]}
                totals["student_ppl"] = float(np.exp(totals["student_nll_sum"] / totals["tokens"]))
                totals["fkl_mean"] = totals["fkl_sum"] / totals["tokens"]
                summary["segments"][segment] = totals
            for group in groups:
                subset = [r for r in rows if r["group"] == group]
                total = sum(r["segments"]["all"]["tokens"] for r in subset)
                summary["groups"][group] = {
                    "student_ppl": float(np.exp(sum(r["segments"]["all"]["student_nll_sum"] for r in subset) / total)),
                    "teacher_ppl": float(np.exp(sum(r["segments"]["all"]["teacher_nll_sum"] for r in subset) / total)),
                    "fkl": sum(r["segments"]["all"]["fkl_sum"] for r in subset) / total}
            summary["first_token_mean_logprob_student"] = float(np.mean([r["first_token_logprob_student"] for r in rows]))
            summary["first_token_mean_logprob_teacher"] = float(np.mean([r["first_token_logprob_teacher"] for r in rows]))
            result["variants"][name + "_" + fmt] = summary
        result["caveats"] = ["32 bounded integer-gold targets, 8 per correction outcome; deliberately nonrepresentative.",
            "Teacher/forced prefixes differ as in each training protocol; no independent teacher solve accuracy was measured.",
            "Logit-gradient L1 is dKL/dlogits, not a parameter/LoRA gradient norm.",
            "Frozen forward-only scoring in eval mode; no optimizer, parameter updates, or answer generation."]
        p.write_json(folder / "analysis.json", result)
        p.write_json(folder / "per-target.json", observations)
        run.log({"analysis/summary": swanlab.Text(json.dumps(result, ensure_ascii=False))})
        return {"analysis_file": str(folder / "analysis.json"), "rows": 32, "variant_protocol_pairs": 5,
                "new_generation": False, "new_training": False}

    import fcntl
    with (ROOT / "preparation/logs/math-training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p.experiment("数值校验", run_id, folder, {"groups": {k: len(v) for k, v in groups.items()},
                     "method": "Frozen full-vocabulary KL, thinking/final/EOS segments; 32 existing training targets",
                     "model_paths": {k: str(v) for k, v in variants.items()},
                     "new_generation": False, "new_training": False}, action,
                     name=p.unique_run_name("20261116-数值校验-思考与解答KL分段", folder.parent))


if __name__ == "__main__":
    main()
