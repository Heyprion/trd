"""CPU-only scope-boundary checks; no production process is signalled."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from recipe.math_evaluation import runtime_control as c
from preparation.training import finish_reduced_eval as finalizer
from preparation.training.finish_reduced_eval import selected_metrics


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results.json"
        self.control = {"datasets": ["aime24", "aime25"], "stop_after": "aime25",
                        "legacy_suite_pid": 123, "legacy_suite_created_at": 42.0}
        self.entry = {"aime24_pass16_generation_pass_16": .5, "aime25_pass16_generation_pass_16": .6}
        self.output.write_text(json.dumps({"model": self.entry}))
        self.parent = Mock(pid=123)
        self.parent.create_time.return_value = 42.0
        self.parent.cmdline.return_value = ["python", "run_eval_suite.py", "--results_file", str(self.output)]

    def arm(self):
        (self.root / "evaluation-control.json").write_text(json.dumps(self.control))

    def test_absent_control_does_nothing(self):
        with patch.object(os, "kill") as kill:
            self.assertFalse(c.stop_legacy_suite_if_requested(self.output, "aime25_pass16_generation.parquet", "model", Mock()))
        kill.assert_not_called()

    def test_first_benchmark_does_not_stop_current_generation(self):
        self.arm()
        cleanup = Mock()
        with patch.object(os, "kill") as kill:
            self.assertFalse(c.stop_legacy_suite_if_requested(self.output, "aime24_pass16_generation.parquet", "model", cleanup))
        kill.assert_not_called()
        cleanup.assert_not_called()

    def test_only_matching_parent_can_be_stopped(self):
        self.arm()
        with patch.object(os, "getppid", return_value=456), patch.object(os, "kill") as kill:
            self.assertFalse(c.stop_legacy_suite_if_requested(self.output, "aime25_pass16_generation.parquet", "model", Mock()))
        kill.assert_not_called()

    def test_requires_both_persisted_scores(self):
        self.arm()
        self.output.write_text(json.dumps({"model": {"aime24_pass16_generation_pass_16": .5}}))
        with patch.object(os, "getppid", return_value=123), patch("psutil.Process", return_value=self.parent), \
                patch.object(os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "before aime25"):
                c.stop_legacy_suite_if_requested(self.output, "aime25_pass16_generation.parquet", "model", Mock())
        kill.assert_not_called()

    def test_cleanup_and_marker_precede_graceful_termination(self):
        self.arm()
        events = []
        cleanup = Mock(side_effect=lambda: events.append("cleanup"))

        def kill(pid, sig):
            self.assertEqual((pid, sig), (123, signal.SIGTERM))
            self.assertTrue((self.root / "scope-boundary.json").is_file())
            events.append("terminate")

        with patch.object(os, "getppid", return_value=123), patch("psutil.Process", return_value=self.parent), \
                patch.object(os, "kill", side_effect=kill):
            self.assertTrue(c.stop_legacy_suite_if_requested(self.output, "aime25_pass16_generation.parquet", "model", cleanup))
        self.assertEqual(events, ["cleanup", "terminate"])
        self.assertEqual(json.loads((self.root / "scope-boundary.json").read_text())["completed_datasets"], ["aime24", "aime25"])

    def test_reused_pid_is_rejected(self):
        self.arm()
        self.parent.create_time.return_value = 99
        with patch.object(os, "getppid", return_value=123), patch("psutil.Process", return_value=self.parent), \
                patch.object(os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "PID was reused"):
                c.stop_legacy_suite_if_requested(self.output, "aime25_pass16_generation.parquet", "model", Mock())
        kill.assert_not_called()

    def test_reporting_requires_only_selected_benchmarks_and_all_six_metrics(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        gen = self.root / "generations"
        gen.mkdir()
        entry = {}
        for name in self.control["datasets"]:
            pq.write_table(pa.Table.from_pylist([{"responses": ["fixture"] * 16}]), gen / f"{name}_pass16_generation.parquet")
            for key in ("pass16", "pass8", "avg_pass1"):
                entry[f"{name}_{key}_generation_pass_16"] = .5
        self.output.write_text(json.dumps({"model": entry}))
        metrics = selected_metrics(self.root, self.control["datasets"], "model", {"aime24": 1, "aime25": 1})
        self.assertEqual(len(metrics), 6)
        self.assertFalse(any("hmmt25" in key for key in metrics))
        pq.write_table(pa.Table.from_pylist([{"responses": ["fixture"] * 15}]), gen / "aime25_pass16_generation.parquet")
        with self.assertRaisesRegex(ValueError, "Incomplete aime25"):
            selected_metrics(self.root, self.control["datasets"], "model", {"aime24": 1, "aime25": 1})

    def test_finalizer_reuses_existing_record_and_completes_only_requested_scope(self):
        import sys
        import pyarrow as pa
        import pyarrow.parquet as pq

        gen = self.root / "generations"
        gen.mkdir()
        entry = {}
        for name in self.control["datasets"]:
            pq.write_table(pa.Table.from_pylist([{"responses": ["fixture"] * 16}]), gen / f"{name}_pass16_generation.parquet")
            for key in ("pass16", "pass8", "avg_pass1"):
                entry[f"{name}_{key}_generation_pass_16"] = .5
        self.output.write_text(json.dumps({"fixture-distilled": entry}))
        original = {"stage": "eval", "run_id": "fixture", "status": "failed", "error": "expected scope exit",
                    "swanlab_url": "https://swanlab.cn/@fixture/project/runs/abcd1234",
                    "config": {"model_path": "/fixture/model", "generation": {},
                               "datasets": {name: {"rows": 1} for name in ("aime24", "aime25", "hmmt25", "beyondaime", "amobench")}}}
        (self.root / "summary.json").write_text(json.dumps(original))
        fake = Mock()
        with patch.dict(sys.modules, {"swanlab": fake}), patch.object(finalizer, "ROOT", self.root), \
                patch.object(finalizer.subprocess, "run") as commands:
            metrics = finalizer.finalize(self.root, self.control)
        self.assertEqual(fake.init.call_args.kwargs["id"], "abcd1234")
        self.assertEqual(fake.init.call_args.kwargs["resume"], "must")
        self.assertEqual(len(metrics), 6)
        self.assertEqual(commands.call_count, 2)
        self.assertEqual(commands.call_args_list[0].args[0][-1], "aime24,aime25")
        revised = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(revised["status"], "completed")
        self.assertNotIn("error", revised)
        self.assertEqual(list(revised["config"]["datasets"]), ["aime24", "aime25"])
        self.assertEqual(revised["scope_change"]["skipped_datasets"], ["hmmt25", "beyondaime", "amobench"])
        self.assertTrue((self.root / "summary-before-scope-finalization.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
