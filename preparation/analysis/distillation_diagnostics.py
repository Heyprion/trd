"""Reanalyse saved AIME outputs and both KL training runs; no model generation."""
from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparation.training import pipeline as p
from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string, remove_boxed, strip_string

BASE = ROOT / "preparation/logs/math-base-eval/base-20261116-113135-834520-153610/eval"
FIRST = ROOT / "preparation/logs/math-runs/20261115-185850-279780-53607"
REPAIR = ROOT / "preparation/logs/math-runs/repair-chat-9984-v1"


def desc(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "max": float(x.max()), "min": float(x.min())}


def ci(delta):
    x = np.asarray(delta)
    rng = np.random.default_rng(20261116)
    means = x[rng.integers(0, len(x), (20000, len(x)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def boxed(text):
    try:
        answer = last_boxed_only_string(text)
        return strip_string(remove_boxed(answer)) if answer is not None else None
    except Exception:
        return None


def features(tokenizer, texts):
    result = []
    for begin in range(0, len(texts), 16):
        batch = texts[begin:begin + 16]
        encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for text, ids in zip(batch, encoded, strict=True):
            tail = ids[-1024:]
            ngrams = Counter(tuple(tail[i:i + 16]) for i in range(max(0, len(tail) - 15)))
            repeated = sum(n for n in ngrams.values() if n > 1) / max(1, sum(ngrams.values()))
            maximum = max(ngrams.values(), default=0)
            result.append({"tokens": len(ids), "near_cap": len(ids) >= 38910, "answer": boxed(text),
                "think": "<think>" in text or "</think>" in text,
                "tail_repeat16_fraction": repeated, "tail_top16_count": maximum,
                "loop_proxy": maximum >= 8 and repeated >= .5,
                "box_mentions": text.count("\\boxed"), "chars": len(text)})
    return result


def aggregate(scores, feats):
    lens = [r["tokens"] for r in feats]
    correct = scores.reshape(-1) > 0
    return {"questions": len(scores), "responses": scores.size, "correct_responses": int(scores.sum()),
            "avg16": float(scores.mean()), "pass16": float((scores.max(axis=1) > 0).mean()),
            "pass8": float((scores[:, :8].max(axis=1) > 0).mean()),
            "lengths": desc(lens), "lengths_correct": desc(np.asarray(lens)[correct]),
            "lengths_wrong": desc(np.asarray(lens)[~correct]),
            "near_cap": sum(r["near_cap"] for r in feats),
            "near_cap_wrong": sum(r["near_cap"] and not s for r, s in zip(feats, correct)),
            "missing_box": sum(r["answer"] is None for r in feats),
            "loop_proxy": sum(r["loop_proxy"] for r in feats),
            "loop_proxy_wrong": sum(r["loop_proxy"] and not s for r, s in zip(feats, correct)),
            "loop_proxy_threshold_max4": sum(r["tail_top16_count"] >= 4 and r["tail_repeat16_fraction"] >= .5 for r in feats),
            "loop_proxy_threshold_max12": sum(r["tail_top16_count"] >= 12 and r["tail_repeat16_fraction"] >= .5 for r in feats),
            "think_tag_responses": sum(r["think"] for r in feats)}


def training_metrics(folder):
    all_rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    steps = [r for r in all_rows if "train/global_step" in r]
    assert [r["train/global_step"] for r in steps] == list(range(1, 79))
    keys = ["kl_loss", "kl_loss_per_traj", "student_perplexity", "teacher_perplexity", "response_tokens",
            "grad_norm", "learning_rate", "kl_p50", "kl_p95", "kl_p99", "kl_max"]
    result = {"updates": len(steps), "all_finite": all(math.isfinite(v) for r in steps for v in r.values() if isinstance(v, (int, float))),
              "epoch_metrics": [r for r in all_rows if "epoch/kl_loss" in r],
              "gradient_norm_gt1_updates": sum(r["train/grad_norm"] > 1 for r in steps),
              "learning_rate_max": max(r["train/learning_rate"] for r in steps),
              "learning_rate_last": steps[-1]["train/learning_rate"], "windows": {}}
    for label, subset in [("first10", steps[:10]), ("middle10", steps[34:44]), ("last10", steps[-10:])]:
        result["windows"][label] = {key: desc([r["train/" + key] for r in subset]) for key in keys}
        result["windows"][label]["mean_log_ppl_gap"] = float(np.mean([
            math.log(r["train/student_perplexity"] / r["train/teacher_perplexity"]) for r in subset]))
    result["logger_audit"] = {"rank": 0, "samples_in_logged_microbatches": len(steps) * 2,
        "fraction_training_samples_in_step_loss_records": len(steps) * 2 / 9984,
        "loss_records": "Last outer microbatch of each 32-batch accumulation, not the update average",
        "gradient_records": "Both DP ranks, 32 accumulated outer batches; norm before clipping at 1.0",
        "lr_records": "Scheduler is advanced after optimizer.step; logged LR is for the following update",
        "step_time": "Last outer batch time including optimizer step; not whole update duration",
        "epoch_records": "Arithmetic means of local-rank outer-batch metrics, not global token-weighted validation"}
    return result, steps


def plot(folder, models, feats, scores, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base": "#50667d", "first": "#d28442", "repair": "#176f77"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in ["first", "repair"]:
        rows = steps[name]
        x = np.arange(1, 79)
        for ax, key in [(axes[0, 0], "kl_loss"), (axes[0, 1], "grad_norm"), (axes[1, 0], "student_perplexity")]:
            y = np.asarray([r["train/" + key] for r in rows])
            ax.plot(x, y, color=colors[name], alpha=.2)
            ax.plot(np.arange(5.5, 74), np.convolve(y, np.ones(10) / 10, "valid"), color=colors[name], label=name)
        axes[1, 1].plot(x, [r["train/learning_rate"] for r in rows], label=name, color=colors[name])
    axes[0, 0].set_title("Logged KL (last microbatch; 10-point mean)")
    axes[0, 1].set_title("Accumulated gradient norm before clipping")
    axes[0, 1].axhline(1, ls="--", color="#777", lw=1)
    axes[1, 0].set_title("Logged student PPL (last microbatch)")
    axes[1, 1].set_title("Learning rate logged after scheduler step")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.grid(alpha=.15)
        ax.legend()
    fig.savefig(folder / "training-curves.svg")
    fig.savefig(folder / "training-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    matrix = np.vstack([scores[name].sum(axis=1) for name in models])
    heat = axes[0, 0].imshow(matrix, aspect="auto", vmin=0, vmax=16, cmap="YlGnBu")
    axes[0, 0].set_yticks(range(3), list(models))
    axes[0, 0].set_xticks(range(0, 30, 2), range(1, 31, 2))
    axes[0, 0].set_title("Correct answers out of 16, each AIME25 question")
    fig.colorbar(heat, ax=axes[0, 0])
    for name in models:
        lens = np.sort([r["tokens"] for r in feats[name]])
        axes[0, 1].plot(lens, np.arange(1, len(lens) + 1) / len(lens), label=name, color=colors[name])
    axes[0, 1].set_xlabel("Response tokens")
    axes[0, 1].set_ylabel("Empirical CDF")
    axes[0, 1].set_title("Output length distributions")
    axes[0, 1].legend()
    for i, name in enumerate(models):
        counts = [sum(r["near_cap"] for r in feats[name]), sum(r["loop_proxy"] for r in feats[name]),
                  sum(r["answer"] is None for r in feats[name])]
        axes[1, 0].bar(np.arange(3) + (i - 1) * .25, counts, width=.25, label=name, color=colors[name])
    axes[1, 0].set_xticks(range(3), ["Near output cap", "Tail-loop proxy", "No parsed boxed"])
    axes[1, 0].set_title("Mechanical output flags (480 responses each)")
    axes[1, 0].legend()
    axes[1, 1].scatter(scores["base"].sum(axis=1), scores["repair"].sum(axis=1), alpha=.7, color=colors["repair"])
    axes[1, 1].plot([0, 16], [0, 16], color="#888", ls="--")
    axes[1, 1].set(xlabel="Base correct / 16", ylabel="Repair correct / 16", title="Question-paired accuracy")
    fig.savefig(folder / "output-diagnostics.svg")
    fig.savefig(folder / "output-diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = "distillation-diagnostics-" + stamp
    folder = ROOT / "preparation/logs/distillation-diagnostics" / run_id
    inputs = {"base": BASE, "first": FIRST / "eval", "repair": REPAIR / "eval"}

    def action(run):
        import swanlab
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-4B-Instruct-2507", local_files_only=True)
        result = {"swanlab_url": run.url, "sampling_unit": "question; no per-response cross-model pairing",
                  "bootstrap_repeats": 20000, "bootstrap_seed": 20261116,
                  "evaluation": {}, "comparisons": {}, "difficulty_buckets": {}, "training": {}, "sources": {}}
        models, matrices, feature_sets, rows_by_model = {}, {}, {}, {}
        observations = []
        for label, directory in inputs.items():
            summary = p.read_json(directory / "summary.json")
            assert summary["status"] == "completed"
            file = directory / "generations/aime25_pass16_generation.parquet"
            rows = pq.read_table(file).to_pylist()
            assert len(rows) == 30 and all(len(r["responses"]) == 16 for r in rows)
            texts = [t for row in rows for t in row["responses"]]
            scores = np.asarray([[compute_score(t, row["reward_model"]["ground_truth"]) for t in row["responses"]] for row in rows])
            feats = features(tokenizer, texts)
            for i, feat in enumerate(feats):
                observations.append({"model": label, "row_index": i // 16, "response_index": i % 16,
                                     "correct": int(scores.reshape(-1)[i]), **feat})
            models[label] = summary
            matrices[label], feature_sets[label], rows_by_model[label] = scores, feats, rows
            result["evaluation"][label] = aggregate(scores, feats)
            result["sources"][label] = {"summary": str(directory / "summary.json"), "swanlab_url": summary["swanlab_url"],
                                       "generation_sha256": p.file_hash(file)}
            assert math.isclose(scores.mean(), summary["result"]["metrics"]["eval/aime25/avg@16"], abs_tol=1e-12)
            print(label, json.dumps(result["evaluation"][label]), flush=True)
        for row in range(30):
            assert all(rows_by_model["base"][row]["prompt"] == rows_by_model[name][row]["prompt"] and
                       rows_by_model["base"][row]["reward_model"] == rows_by_model[name][row]["reward_model"] for name in models)
        questions = []
        for row in range(30):
            item = {"row_index": row, "source_index": rows_by_model["base"][row]["extra_info"]["index"]}
            for label in models:
                fs = feature_sets[label][row * 16:(row + 1) * 16]
                answers = Counter(f["answer"] if f["answer"] is not None else "<missing>" for f in fs)
                item[label] = {"correct": int(matrices[label][row].sum()), "mean_tokens": float(np.mean([f["tokens"] for f in fs])),
                               "near_cap": sum(f["near_cap"] for f in fs), "loop_proxy": sum(f["loop_proxy"] for f in fs),
                               "distinct_answers": len(answers), "largest_answer_share": max(answers.values()) / 16,
                               "answer_entropy_nats": -sum(n / 16 * math.log(n / 16) for n in answers.values())}
            questions.append(item)
        for before, after in [("base", "first"), ("base", "repair"), ("first", "repair")]:
            b, a = matrices[before], matrices[after]
            delta = a.mean(axis=1) - b.mean(axis=1)
            bp, ap = b.max(axis=1) > 0, a.max(axis=1) > 0
            gains, losses = int((~bp & ap).sum()), int((bp & ~ap).sum())
            discordant = gains + losses
            result["comparisons"][before + "_to_" + after] = {
                "avg16_delta": float(delta.mean()), "avg16_delta_ci95": ci(delta),
                "questions_avg_gained": int((delta > 0).sum()), "questions_avg_lost": int((delta < 0).sum()),
                "questions_avg_equal": int((delta == 0).sum()), "coverage_gained": gains, "coverage_lost": losses,
                "coverage_gained_rows": np.where(~bp & ap)[0].tolist(), "coverage_lost_rows": np.where(bp & ~ap)[0].tolist(),
                "coverage_mcnemar_exact_two_sided_p": min(1., 2 * sum(math.comb(discordant, j) for j in range(min(gains, losses) + 1)) / 2**discordant)}
        base_counts = matrices["base"].sum(axis=1)
        for name, mask in [("base_0", base_counts == 0), ("base_1_to_8", (base_counts >= 1) & (base_counts <= 8)),
                           ("base_9_to_15", (base_counts >= 9) & (base_counts <= 15)), ("base_16", base_counts == 16)]:
            result["difficulty_buckets"][name] = {"rows": np.where(mask)[0].tolist(), "models": {
                label: {"avg16": float(matrix[mask].mean()), "pass16": float((matrix[mask].max(axis=1) > 0).mean()),
                        "correct_responses": int(matrix[mask].sum())} for label, matrix in matrices.items()}} if mask.any() else {"rows": []}
        train_steps = {}
        for label, directory in [("first", FIRST / "train"), ("repair", REPAIR / "train")]:
            result["training"][label], train_steps[label] = training_metrics(directory)
            print("training", label, json.dumps(result["training"][label]), flush=True)
        result["training_target_audit"] = p.read_json(ROOT / "preparation/logs/aime-regression-analysis/20261116-review/summary.json")["training_targets"]
        result["caveats"] = [
            "Tail-loop flags are mechanical proxies, not verified reasoning-error labels.",
            "Only one generation run per model; repair evaluated AIME25 first, unlike earlier runs after AIME24.",
            "Question bootstrap describes this question sample; no training-seed replication.",
            "No intermediate checkpoint generation accuracy; cannot diagnose overtraining from final accuracy alone.",
            "Logged step losses are last microbatches on rank0, not accumulated-update averages."]
        p.write_json(folder / "analysis.json", result)
        p.write_json(folder / "per-question.json", questions)
        with (folder / "per-response.jsonl").open("w") as stream:
            for row in observations:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        plot(folder, models, feature_sets, matrices, train_steps)
        for name in ["training-curves", "output-diagnostics"]:
            run.log({"figures/" + name: swanlab.Image(str(folder / (name + ".png")))})
        run.log({"analysis/summary": swanlab.Text(json.dumps(result, ensure_ascii=False))})
        print("COMPARISONS", json.dumps(result["comparisons"]), flush=True)
        print("QUESTIONS", json.dumps(questions), flush=True)
        return {"analysis_file": str(folder / "analysis.json"), "per_question_file": str(folder / "per-question.json"),
                "base_avg16": result["evaluation"]["base"]["avg16"], "repair_avg16": result["evaluation"]["repair"]["avg16"],
                "base_pass16": result["evaluation"]["base"]["pass16"], "repair_pass16": result["evaluation"]["repair"]["pass16"]}

    config = {"source_eval_directories": {k: str(v) for k, v in inputs.items()}, "new_model_generation": False,
              "new_training": False, "method": "Saved-output grading/tokenization and training-log analysis"}
    p.experiment("评测", run_id, folder, config, action,
                 name=p.unique_run_name("20261116-评测-蒸馏退化详细诊断", folder.parent))


if __name__ == "__main__":
    main()
