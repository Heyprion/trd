"""CPU regression checks for data decisions and stage boundaries; no model mocks in production."""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import time
import unittest

from core import (answer_key, digest, initial_filter, load_rows, native_index, near_pairs,
                  prepare_rows, problem_key, read_eval, save_table)
from pipeline import clean_stage, export, label_stage


class FixturePool:
    def __init__(self, quality=None, pair="D", label="algebra,easy"):
        self.quality = quality or {}
        self.pair = pair
        self.label = label
        self.calls = []

    def calibrate(self, tasks, deadline):
        pass

    def infer(self, tasks, deadline, label="", retries=1):
        self.calls.extend(tasks)
        result = {}
        for task in tasks:
            if time.monotonic() >= deadline:
                result[task["id"]] = {"id": task["id"], "status": "time_budget", "text": "", "tokens": 0}
                continue
            text = self.pair if task["kind"] == "pair" else self.label if task["kind"] == "label" else self.quality.get(task["id"], "A")
            result[task["id"]] = {"id": task["id"], "status": "ok", "text": text, "tokens": 2}
        return result


def args():
    return Namespace(workers=1, data_version="test", near_threshold=.82, eval_near_threshold=.45,
                     max_near_pairs=20000, seed=42)


def example(problem="Find x if x+1=3.", answer="2", solution=""):
    return {"problem": problem, "answer": answer, "solution": solution}


class DecisionsTest(unittest.TestCase):
    def prepare(self, rows):
        return prepare_rows(rows, "f"*64, "test", 1)

    def test_canonicalization_preserves_semantics(self):
        self.assertNotEqual(problem_key("x<2"), problem_key("x>2"))
        self.assertNotEqual(problem_key("a b"), problem_key("ab"))
        self.assertNotEqual(problem_key("x_1"), problem_key("x1"))
        self.assertNotEqual(problem_key("α+1"), problem_key("β+1"))
        self.assertNotEqual(problem_key("求面积"), problem_key("求体积"))
        self.assertEqual(problem_key("x + 2"), problem_key("x+2"))

    def test_answer_fast_path_is_conservative(self):
        self.assertEqual(answer_key(r"$\dfrac{1}{2}$"), answer_key("0.5"))
        self.assertEqual(answer_key(r"\boxed{035}"), answer_key("35"))
        self.assertNotEqual(answer_key("16:1"), answer_key("16"))
        self.assertNotEqual(answer_key("0"), answer_key("2"))

    def test_empty_solution_and_asy_newlines_survive(self):
        source = "Task\n[asy]\n// comment\ndraw((0,0)--(1,1));\n[/asy]"
        prepared = self.prepare([example(source), example(answer="")])
        self.assertEqual(prepared[0][0]["problem"], source)
        self.assertIsNone(prepared[0][1])
        self.assertEqual(prepared[1][1], "missing_answer")

    def test_duplicate_numeric_answers_merge_but_conflicts_drop_group(self):
        rows = [example(answer="0.5"), example(answer=r"\frac{1}{2}"),
                example("Other task?", "20"), example("Other task?", "25")]
        active, decisions, _ = initial_filter(self.prepare(rows), [], {})
        self.assertEqual(len(active), 1)
        self.assertEqual(CounterReasons(decisions), {"exact_duplicate": 1, "duplicate_answer_conflict": 2})

    def test_native_conflicts_are_excluded_and_consistent_metadata_reused(self):
        row = example()
        key = problem_key(row["problem"])
        first = {**row, "domain": ["Algebra"], "difficulty": 4.0}
        active, _, _ = initial_filter(self.prepare([row]), [], {key: [first]})
        self.assertEqual(active[0]["native_level"], 2)
        self.assertEqual(active[0]["native_domain"], 1)
        active, decisions, _ = initial_filter(self.prepare([row]), [], {key: [first, {**first, "difficulty": 6.0}]})
        self.assertFalse(active)
        self.assertEqual(next(iter(decisions.values()))["reason"], "native_label_conflict")

    def test_eval_exact_overlap_excludes_all_duplicates(self):
        row = example()
        active, decisions, _ = initial_filter(self.prepare([row, row]), [{"problem": row["problem"]}], {})
        self.assertFalse(active)
        self.assertEqual({r["reason"] for r in decisions.values()}, {"eval_exact_overlap"})

    def test_eval_mod_transform_is_retrieved(self):
        p = "The 30 edges of a regular icosahedron are labeled 1 to 30. Count red white blue colorings with two equal colors on each triangular face."
        rows = [{"problem": p}]
        pairs, overflow, _ = near_pairs(rows, [{"problem": p + " Find the answer mod 1000."}], workers=1)
        self.assertFalse(overflow)
        self.assertTrue(any(r["eval"] for r in pairs))

    def test_missing_image_is_removed_before_label_calls(self):
        rows = [example("In the missing figure find angle x.", "90"), example("Compute 1+1.", "2")]
        prepared = self.prepare(rows)
        pool = FixturePool(quality={prepared[0][0]["raw_id"]: "R1"})
        usable, decisions, _ = clean_stage(args(), rows, [], {}, "f"*64, pool, time.monotonic()+10, lambda *x: None)
        labeled, decisions = label_stage(args(), usable, decisions, pool, time.monotonic()+10, lambda *x: None)
        self.assertEqual(len(labeled), 1)
        label_tasks = [r for r in pool.calls if r["kind"] == "label"]
        self.assertEqual(len(label_tasks), 1)
        self.assertNotIn("missing figure", label_tasks[0]["messages"][1]["content"])
        self.assertEqual(set(r["status"] for r in decisions.values()), {"usable", "excluded"})

    def test_deadline_exclusions_remain_explicit(self):
        rows = [example("Compute 1+1.", "2")]
        pool = FixturePool()
        usable, decisions, _ = clean_stage(args(), rows, [], {}, "f"*64, pool, time.monotonic()-1, lambda *x: None)
        self.assertFalse(usable)
        self.assertEqual(next(iter(decisions.values()))["reason"], "quality_time_budget")

    def test_near_pair_timeout_is_a_technical_exclusion(self):
        class PairTimeoutPool(FixturePool):
            def infer(self, tasks, deadline, label="", retries=1):
                if tasks and tasks[0]["kind"] == "pair":
                    return {t["id"]: {"id": t["id"], "status": "time_budget", "text": "", "tokens": 0} for t in tasks}
                return super().infer(tasks, deadline, label, retries)
        row = example("The 30 edges of a regular icosahedron are painted red white or blue. Count the colorings with exactly two equal colors on each triangular face.", "123")
        evaluation = [{"problem": row["problem"] + " Find the answer mod 1000."}]
        usable, decisions, _ = clean_stage(args(), [row], evaluation, {}, "f"*64, PairTimeoutPool(), time.monotonic()+10, lambda *x: None)
        self.assertFalse(usable)
        self.assertEqual(next(iter(decisions.values()))["reason"], "near_pair_time_budget")

    def test_standalone_stage_artifact_round_trip(self):
        rows = [example("Compute 1+1.", "2"), example("Compute 9-2.", "7")]
        pool = FixturePool(label="algebra,medium")
        usable, decisions, _ = clean_stage(args(), rows, [], {}, "f"*64, pool, time.monotonic()+10, lambda *x: None)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = {"input_rows": 2, "policy_hash": "fixed"}
            export(args(), directory/"clean", usable, decisions, manifest)
            loaded = load_rows(directory/"clean/usable.parquet")
            restored = {r["raw_id"]: r for r in load_rows(directory/"clean/decisions.parquet")}
            labeled, restored = label_stage(args(), loaded, restored, pool, time.monotonic()+10, lambda *x: None)
            summary = export(args(), directory/"final", labeled, restored, manifest, final=True)
            self.assertEqual(summary["difficulty_counts"], {"medium": 2})
            from datasets import load_from_disk
            ds = load_from_disk(str(directory/"final/dataset"))
            self.assertEqual(len(ds), 2)
            self.assertEqual(set(ds["difficulty"]), {"medium"})

    def test_export_rejects_incomplete_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                export(args(), Path(tmp), [], {}, {"input_rows": 1})


def CounterReasons(decisions):
    from collections import Counter
    return dict(Counter(r["reason"] for r in decisions.values()))


if __name__ == "__main__":
    unittest.main()
