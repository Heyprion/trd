"""Assemble small report artifacts from completed diagnostics; resume their SwanLab run."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import swanlab

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "preparation/logs/distillation-diagnostics/distillation-diagnostics-20261116-103129"
PROBE = ROOT / "preparation/logs/distillation-diagnostics/target-kl-probe-20261116-103527"
REPORT = ROOT / "preparation/training/reports/20261116-distillation-diagnostics.zh-CN.md"
OUT = REPORT.parent / "20261116-distillation-diagnostics"


def read(path):
    return json.loads(path.read_text())


def enumerate_cases():
    divisors = [(a, b) for a in range(5) for b in range(3)]
    has_a4 = sum(1 << i for i, (a, b) in enumerate(divisors) if a == 4)
    has_b2 = sum(1 << i for i, (a, b) in enumerate(divisors) if b == 2)
    favorable = sum(bool(mask & has_a4) and bool(mask & has_b2) for mask in range(1 << 15))
    probability = Fraction(favorable, 1 << 15)
    assert favorable == 27904 and probability == Fraction(109, 128)
    # Six horizontal edges followed by six vertical edges.
    cells = [[2 * y + x, 2 * (y + 1) + x, 6 + 3 * y + x, 6 + 3 * y + x + 1]
             for y in range(2) for x in range(2)]
    valid = sum(all(sum((mask >> edge) & 1 for edge in cell) == 2 for cell in cells)
                for mask in range(1 << 12))
    assert valid == 82
    return {"file_row_22": {"total_subsets": 1 << 15, "favorable": favorable,
                            "probability": str(probability),
                            "answer": probability.numerator + probability.denominator},
            "file_row_18": {"total_colorings": 1 << 12, "valid_colorings": valid}}


def main():
    # Numerical checks and report plots stay in the same documented analysis experiment.
    run = swanlab.init(project="trd-math-opd", id="pxa1iezc", resume="must", mode="online",
                       on_online_error="raise", log_dir=str(SOURCE / "swanlab-report"))
    try:
        OUT.mkdir(exist_ok=True)
        analysis, probe = read(SOURCE / "analysis.json"), read(PROBE / "analysis.json")
        observations = read(PROBE / "per-target.json")
        first_token = {}
        for label in ["base", "first", "repair"]:
            rows = [r for r in observations if r["model"] == label and r["format"] == "chat"]
            assert len(rows) == 32 and all(r["first_token_id"] == 151667 for r in rows)
            logs = np.array([r["first_token_logprob_student"] for r in rows])
            probs = np.exp(logs)
            first_token[label] = {"n": len(rows), "geometric_mean": float(np.exp(logs.mean())),
                                  "arithmetic_mean": float(probs.mean()), "min": float(probs.min()),
                                  "max": float(probs.max()), "temperature": 1.0,
                                  "sampling_filters_applied": False}
        teacher_logs = np.array([r["first_token_logprob_teacher"] for r in observations
                                 if r["model"] == "repair" and r["format"] == "chat"])
        first_token["teacher"] = {"geometric_mean": float(np.exp(teacher_logs.mean())),
                                 "min": float(np.exp(teacher_logs).min()),
                                 "max": float(np.exp(teacher_logs).max())}
        ppl_root = ROOT / "preparation/logs/ppl-kld"
        ppl = {k: read(ppl_root / folder / "summary.json") for k, folder in {
            "base": "baseline-20261115-101823-758028-27412",
            "first": "distilled-20261116-031913-959867-152167",
            "repair": "chat_repaired-20261116-103117-382399-215906"}.items()}
        assert len({r["config"]["protocol_sha256"] for r in ppl.values()}) == 1
        assert all(r["overall"]["rows"] == 512 and r["overall"]["response_tokens"] == 234375
                   for r in ppl.values())
        cases = enumerate_cases()
        responses = [json.loads(line) for line in (SOURCE / "per-response.jsonl").read_text().splitlines()]
        for label, correct, count60671 in [("base", 14, 2), ("first", 9, 6), ("repair", 6, 8)]:
            rows = [r for r in responses if r["model"] == label and r["row_index"] == 21]
            assert len(rows) == 16 and sum(r["correct"] for r in rows) == correct
            assert sum(r["answer"] == "60671" for r in rows) == count60671
            assert not any(r["near_cap"] or r["loop_proxy"] for r in rows)
            cases["file_row_22"][label] = {"correct": correct, "wrong_answer_60671": count60671}
        case18_counts = {}
        for label in ["base", "first", "repair"]:
            case18_counts[label] = sum(r["correct"] for r in responses
                                       if r["model"] == label and r["row_index"] == 17)
        assert list(case18_counts.values()) == [4, 1, 0]
        cases["file_row_18"]["correct_responses"] = case18_counts
        variants = probe["variants"]
        repair_segments = variants["repair_chat"]["segments"]
        kl_share = {key: repair_segments[key]["fkl_sum"] / repair_segments["all"]["fkl_sum"]
                    for key in ["thinking", "final_solution", "eos", "first8"]}
        result = {"date": "2026-11-16", "evaluation": analysis["evaluation"],
                  "comparisons": analysis["comparisons"], "difficulty_buckets": analysis["difficulty_buckets"],
                  "training": analysis["training"], "validation_512": {k: v["overall"] for k, v in ppl.items()},
                  "validation_protocol": ppl["repair"]["config"]["protocol"],
                  "training_target_audit": analysis["training_target_audit"],
                  "target_probe": probe, "first_token_probabilities_chat": first_token,
                  "probe_repair_token_kl_shares": kl_share, "enumerated_cases": cases,
                  "sources": analysis["sources"],
                  "new_training": False, "new_answer_generation": False,
                  "notes": ["Probe targets are balanced 8 per outcome and not representative.",
                            "KL mass is not a parameter-gradient fraction.",
                            "Question-level exploratory bootstrap, no seed replications.",
                            "Validation PPL/KL uses legacy plain teacher-forced references.",
                            "Per-response observations and original answers remain local."]}
        (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        questions = read(SOURCE / "per-question.json")
        columns = ["file_row_1based"] + [name + "_" + key for name in ["base", "first", "repair"]
                   for key in ["correct", "mean_tokens", "near_cap", "loop_proxy"]]
        with (OUT / "per-question.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in questions:
                writer.writerow({"file_row_1based": row["row_index"] + 1,
                                 **{name + "_" + key: row[name][key] for name in ["base", "first", "repair"]
                                    for key in ["correct", "mean_tokens", "near_cap", "loop_proxy"]}})
        for name in ["training-curves", "output-diagnostics"]:
            shutil.copyfile(SOURCE / (name + ".svg"), OUT / (name + ".svg"))
        colors = ["#4263eb", "#e67700", "#0b8a7b"]
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), layout="constrained")
        labels = ["Base", "First", "Repair"]
        axes[0].bar(labels, [first_token[k]["geometric_mean"] for k in ["base", "first", "repair"]], color=colors)
        axes[0].set(yscale="log", ylim=(1e-24, 1e-15), title="First <think> probability (geomean)",
                    ylabel="T=1 raw softmax probability")
        x = np.arange(3)
        for i, (name, color) in enumerate(zip(["base", "first", "repair"], colors)):
            axes[1].bar(x + (i - 1) * .25, [variants[name + "_chat"]["segments"][s]["fkl_mean"]
                        for s in ["all", "thinking", "final_solution"]], width=.25, label=labels[i], color=color)
        axes[1].set_xticks(x, ["All", "Thinking", "Final"])
        axes[1].set(title="Full-vocabulary forward KL", ylabel="nats / target token")
        axes[1].legend()
        for i, name in enumerate(["base", "first", "repair"]):
            axes[2].plot(range(4), [variants[name + "_chat"]["groups"][g]["student_ppl"] for g in [
                "correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong"]],
                marker="o", color=colors[i], label=labels[i])
        axes[2].set_xticks(range(4), ["C -> C", "C -> W", "W -> C", "W -> W"])
        axes[2].set(title="Wrong targets also become easier to fit", ylabel="Student target PPL",
                    xlabel="Initial -> refined answer (8 targets each)")
        axes[2].legend()
        fig.suptitle("32 bounded training targets: balanced diagnostic, not a population estimate", fontsize=12)
        fig.savefig(OUT / "target-probe.svg")
        fig.savefig(SOURCE / "target-probe.png", dpi=160)
        plt.close(fig)
        # Matplotlib emits trailing spaces in SVG path lines; keep report diffs clean.
        for svg in OUT.glob("*.svg"):
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        run.log({"report/markdown": swanlab.Text(REPORT.read_text()),
                 "report/summary": swanlab.Text(json.dumps(result, ensure_ascii=False)),
                 "report/enumerated_cases": swanlab.Text(json.dumps(cases)),
                 "figures/target-probe": swanlab.Image(str(SOURCE / "target-probe.png")),
                 "diagnostics/think_first_token_max_repair": first_token["repair"]["max"],
                 "diagnostics/thought_kl_mass_probe_repair": kl_share["thinking"]})
        print(json.dumps({"report": str(REPORT), "artifacts": str(OUT), "swanlab_url": run.url,
                          "enumerated_cases": cases, "first_token": first_token}, ensure_ascii=False, indent=2))
    except Exception:
        run.finish(state="crashed")
        raise
    run.finish()


if __name__ == "__main__":
    main()
