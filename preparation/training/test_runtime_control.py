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

    def test_run_root_scope_is_read_without_creating_evaluation_folder(self):
        path = self.root / "eval/results.json"
        self.control["datasets"] = ["aime25"]
        self.arm()
        self.assertEqual(c.read_control(path)["datasets"], ["aime25"])
        self.assertFalse(path.parent.exists())

    def test_watcher_waits_for_owner_without_sending_signals(self):
        run = self.root / "run"
        folder = run / "eval"
        folder.mkdir(parents=True)
        (run / "train").mkdir()
        (run / "train/summary.json").write_text('{"status":"completed"}')
        (folder / "summary.json").write_text("{}")
        with patch.object(finalizer, "process_alive", side_effect=[True, False]), \
                patch.object(finalizer.time, "sleep") as sleep, patch.object(os, "kill") as kill, \
                patch.object(finalizer, "finalize", return_value={"eval/aime25/pass@16": .5}) as finish:
            finalizer.finish_after_controller(folder, {"datasets": ["aime25"]}, 123, 42.)
        kill.assert_not_called()
        sleep.assert_called_once_with(2)
        finish.assert_called_once()
        self.assertEqual(json.loads((run / "evaluation-scope-status.json").read_text())["status"], "completed")

    def test_watcher_does_not_finalize_failed_training(self):
        run = self.root / "run"
        (run / "train").mkdir(parents=True)
        (run / "train/summary.json").write_text('{"status":"failed"}')
        with patch.object(finalizer, "process_alive", return_value=False), \
                patch.object(finalizer, "finalize") as finish:
            with self.assertRaisesRegex(RuntimeError, "Training did not complete"):
                finalizer.finish_after_controller(run / "eval", {"datasets": ["aime25"]}, 123, 42.)
        finish.assert_not_called()

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

    def test_finalizer_reports_aime25_without_any_aime24_responses(self):
        import sys
        import pyarrow as pa
        import pyarrow.parquet as pq
        folder = self.root / "eval"
        (folder / "generations").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{"responses": ["fixture"] * 16}]),
                       folder / "generations/aime25_pass16_generation.parquet")
        entry = {f"aime25_{key}_generation_pass_16": .5 for key in ("pass16", "pass8", "avg_pass1")}
        (folder / "results.json").write_text(json.dumps({"fixture-distilled": entry}))
        original = {"stage": "eval", "run_id": "fixture", "status": "failed", "error": "old scope expected AIME24",
                    "swanlab_url": "https://swanlab.cn/@fixture/project/runs/abcd1234",
                    "config": {"model_path": "/fixture/model", "datasets": {"aime24": {"rows": 30}, "aime25": {"rows": 1}}}}
        (folder / "summary.json").write_text(json.dumps(original))
        control = {"datasets": ["aime25"], "reason": "user_requested_aime25_only"}
        with patch.dict(sys.modules, {"swanlab": Mock()}), patch.object(finalizer, "ROOT", self.root), \
                patch.object(finalizer.subprocess, "run"):
            metrics = finalizer.finalize(folder, control)
        self.assertEqual(len(metrics), 3)
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(list(result["config"]["datasets"]), ["aime25"])
        self.assertEqual(result["scope_change"]["skipped_datasets"], ["aime24"])
        self.assertFalse((folder / "generations/aime24_pass16_generation.parquet").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
