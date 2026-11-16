"""Count actual changes and correctness in the existing paired trajectory cache."""
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparation.training import pipeline as p
from verl.utils.reward_score.math_reward import compute_score

SOURCE = ROOT / "preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json"
REPORTS = ROOT / "preparation/training/reports"


def compare_subset(rows):
    n = len(rows)
    return {"rows": n, "student_pass": sum(r["student_pass"] for r in rows),
            "teacher_whole_response_pass": sum(r["teacher_pass"] for r in rows),
            "teacher_whole_response_pass_rate": sum(r["teacher_pass"] for r in rows) / n if n else None,
            "outcomes": dict(Counter(r["outcome"] for r in rows))}


def main():
    source = p.read_json(SOURCE)
    teacher_path = Path(source["config"]["training"]["TRAINING_DATA_PATH"])
    student_path = teacher_path.parent / "math_y_o_responses.parquet"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = "rewrite-change-audit-" + stamp
    folder = ROOT / "preparation/logs/distillation-diagnostics" / run_id

    def action(run):
        import swanlab
        students = pq.read_table(student_path).to_pylist()
        teachers = pq.read_table(teacher_path).to_pylist()
        assert len(students) == len(teachers) == 9984
        observations = []
        for index, (s, t) in enumerate(zip(students, teachers, strict=True)):
            before, after = s["responses"][0], t["responses"][0]
            gold = s["reward_model"]["ground_truth"]
            assert gold == t["reward_model"]["ground_truth"]
            assert before.strip() == t["extra_info"]["initial_response"].strip()
            a, b = bool(compute_score(before, gold)), bool(compute_score(after, gold))
            # Whole-response equality alone is confounded by the extra thinking block.
            closed = "</think>" in after
            final = after.split("</think>", 1)[1].strip() if closed else None
            observations.append({"row_index": index, "student_pass": a, "teacher_pass": b,
                "raw_equal": before == after, "raw_strip_equal": before.strip() == after.strip(),
                "has_closed_thought": closed, "has_nonempty_final": bool(final),
                "final_equal_ignoring_edge_whitespace": final == before.strip() if closed else None,
                "final_equal_ignoring_all_whitespace": re.sub(r"\s+", "", final) == re.sub(r"\s+", "", before)
                    if closed else None,
                "teacher_final_pass": bool(compute_score(final, gold)) if closed else None,
                "outcome": ("pass" if a else "fail") + "_to_" + ("pass" if b else "fail")})
        raw_changed = [r for r in observations if not r["raw_equal"]]
        closed_rows = [r for r in observations if r["has_closed_thought"]]
        final_equal = [r for r in closed_rows if r["final_equal_ignoring_edge_whitespace"]]
        final_changed = [r for r in closed_rows if not r["final_equal_ignoring_edge_whitespace"]]
        final_equal_ws = [r for r in closed_rows if r["final_equal_ignoring_all_whitespace"]]
        result = {"swanlab_url": run.url, "total_rows": len(observations),
            "sources": {"student": str(student_path), "teacher": str(teacher_path),
                        "student_sha256": p.file_hash(student_path), "teacher_sha256": p.file_hash(teacher_path)},
            "whole_response": compare_subset(observations),
            "raw_text": {"exact_equal": sum(r["raw_equal"] for r in observations),
                         "equal_after_edge_strip": sum(r["raw_strip_equal"] for r in observations),
                         "changed": compare_subset(raw_changed)},
            "final_solution": {"closed_thought_rows": len(closed_rows),
                "unclosed_thought_rows": len(observations) - len(closed_rows),
                "closed_but_empty_final": sum(not r["has_nonempty_final"] for r in closed_rows),
                "equal_after_edge_strip": compare_subset(final_equal),
                "changed_after_edge_strip": compare_subset(final_changed),
                "equal_ignoring_all_whitespace": compare_subset(final_equal_ws),
                "closed_final_pass": sum(r["teacher_final_pass"] for r in closed_rows),
                "closed_final_pass_rate": sum(r["teacher_final_pass"] for r in closed_rows) / len(closed_rows),
                "verified_final_pass_rate_all_rows": sum(r["teacher_final_pass"] for r in closed_rows) / len(observations),
                "closed_final_outcomes": dict(Counter(
                    ("pass" if r["student_pass"] else "fail") + "_to_" +
                    ("pass" if r["teacher_final_pass"] else "fail") for r in closed_rows)),
                "unclosed_original_student_pass": sum(r["student_pass"] for r in observations
                                                       if not r["has_closed_thought"]),
                "changed_final_pass": sum(r["teacher_final_pass"] for r in final_changed),
                "whole_pass_but_final_not_verified": sum(r["teacher_pass"] and not r["teacher_final_pass"]
                                                         for r in observations)},
            "integer_gold_subset": compare_subset([r for r, s in zip(observations, students, strict=True)
                                                   if s["reward_model"]["ground_truth"].strip().lstrip("+-").isdigit()]),
            "notes": ["Pass means the existing last-boxed answer verifier passes, not proof verification.",
                      "Text difference is not necessarily a mathematical method change.",
                      "Unclosed thinking blocks have no separately identified final solution.",
                      "No training, GPU scoring, or new answer generation."]}
        assert result["whole_response"]["student_pass"] == 7153
        assert result["whole_response"]["teacher_whole_response_pass"] == 7310
        assert result["whole_response"]["outcomes"] == {
            "pass_to_pass": 7104, "fail_to_pass": 206, "pass_to_fail": 49, "fail_to_fail": 2625}
        p.write_json(folder / "analysis.json", result)
        p.write_json(folder / "per-row-flags.json", observations)
        p.write_json(REPORTS / "20261116-rewrite-change-audit.json", result)
        run.log({"audit/summary": swanlab.Text(json.dumps(result, ensure_ascii=False)),
                 "audit/student_pass": 7153, "audit/teacher_pass": 7310,
                 "audit/raw_exact_equal": result["raw_text"]["exact_equal"],
                 "audit/final_exact_equal": len(final_equal)})
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return {"analysis_file": str(folder / "analysis.json"), "total_rows": 9984,
                "student_pass": 7153, "teacher_pass": 7310,
                "raw_equal": result["raw_text"]["exact_equal"],
                "final_equal": len(final_equal), "final_changed": len(final_changed)}

    p.experiment("数据", run_id, folder, {"source_train_summary": str(SOURCE),
        "method": "Paired saved-text comparison, with thinking block removed for final-solution equality",
        "new_training": False, "new_generation": False}, action,
        name=p.unique_run_name("20261116-数据-改写数量与正确性审核", folder.parent))


if __name__ == "__main__":
    main()
