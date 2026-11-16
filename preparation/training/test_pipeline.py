"""CPU checks for data isolation, experiment handoff, failures, and real LoRA export."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from preparation.training import pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = {"EPOCH_MODEL_SAVE_DIR": str(self.root / "checkpoint"),
                       "EPOCH_OUTPUT_DIR": str(self.root / "output"), "EVAL_GEN_TP": "2", "MODEL_NAME": "fixture"}

    def model(self):
        folder = self.root / "model"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"fixture")
        return folder

    @contextlib.contextmanager
    def main_mocks(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(p, "ROOT", self.root))
            stack.enter_context(patch.object(p, "build_env", return_value=self.values.copy()))
            stack.enter_context(patch.object(p, "print_config"))
            stack.enter_context(patch.object(p, "validate_train_split", return_value={"rows": 9984}))
            stack.enter_context(patch.object(p, "validate_benchmarks", return_value={"aime24": {"rows": 30}}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def args(self, stage):
        return [stage, "--run-id", "fixture", "--output-root", str(self.root / "logs")]

    def test_dry_run_performs_no_training_or_logging(self):
        with self.main_mocks(), patch.object(p, "train") as train, patch.object(p, "evaluation") as evaluate:
            p.main(self.args("all") + ["--dry-run"])
        train.assert_not_called()
        evaluate.assert_not_called()
        self.assertFalse((self.root / "logs").exists())

    def test_training_failure_never_launches_evaluation(self):
        with self.main_mocks(), patch.object(p, "train", side_effect=RuntimeError("fixture failure")), \
                patch.object(p, "evaluation") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                p.main(self.args("all"))
        evaluate.assert_not_called()

    def test_all_hands_exact_model_and_training_url_to_evaluation(self):
        with self.main_mocks(), patch.object(p, "train", return_value={"swanlab_url": "fixture://train"}), \
                patch.object(p, "evaluation") as evaluate:
            p.main(self.args("all"))
        args = evaluate.call_args.args
        self.assertEqual(args[2], Path(self.values["EPOCH_MODEL_SAVE_DIR"]) / "hf_merged")
        self.assertEqual(args[3]["swanlab_url"], "fixture://train")

    def test_eval_uses_completed_summary_not_a_latest_checkpoint(self):
        model = self.model()
        summary = self.root / "train.json"
        summary.write_text(json.dumps({"stage": "train", "status": "completed", "result": {"model_path": str(model)},
                                       "swanlab_url": "fixture://old-training"}))
        with self.main_mocks(), patch.object(p, "evaluation") as evaluate:
            p.main(self.args("eval") + ["--train-run", str(summary)])
        self.assertEqual(evaluate.call_args.args[2], model)
        self.assertEqual(evaluate.call_args.args[3]["swanlab_url"], "fixture://old-training")

    def test_failed_training_summary_cannot_be_evaluated(self):
        summary = self.root / "failed.json"
        summary.write_text('{"stage":"train", "status":"failed"}')
        with self.main_mocks(), self.assertRaisesRegex(ValueError, "successfully completed"):
            p.main(self.args("eval") + ["--train-run", str(summary)])

    def test_existing_output_is_never_overwritten(self):
        (self.root / "logs/fixture").mkdir(parents=True)
        with self.main_mocks(), self.assertRaisesRegex(ValueError, "already exists"):
            p.main(self.args("all"))

    def test_unmerged_adapter_is_rejected(self):
        model = self.model()
        (model / "lora_adapter").mkdir()
        with self.assertRaisesRegex(ValueError, "not been folded"):
            p.validate_model(model)

    def test_missing_hf_weight_shard_is_rejected(self):
        model = self.model()
        (model / "model.safetensors.index.json").write_text('{"weight_map":{"a":"missing.safetensors"}}')
        with self.assertRaisesRegex(ValueError, "incomplete"):
            p.validate_model(model)

    def test_real_child_stdout_relays_metrics_once(self):
        run = Mock()
        command = [sys.executable, "-c", "from recipe.opd.swanlab_relay import SwanLabRelay; "
                   "print('fixture console'); SwanLabRelay().log({'train/global_step': 1, 'train/kl_loss': 0.5})"]
        p.run_command(command, self.root, run)
        run.log.assert_called_once_with({"train/global_step": 1, "train/kl_loss": 0.5})
        self.assertIn("fixture console", (self.root / "console.log").read_text())
        self.assertEqual(len((self.root / "metrics.jsonl").read_text().splitlines()), 1)

    def test_child_failure_propagates(self):
        with self.assertRaises(subprocess.CalledProcessError):
            p.run_command([sys.executable, "-c", "raise SystemExit(7)"], self.root, Mock())

    def test_experiment_failure_records_crashed_status(self):
        run = Mock(url="fixture://failed")
        fake = Mock()
        fake.init.return_value = run
        with patch.dict(sys.modules, {"swanlab": fake}), patch.object(p, "ROOT", self.root):
            with self.assertRaisesRegex(RuntimeError, "fixture error"):
                p.experiment("train", "fixture", self.root / "train", {}, Mock(side_effect=RuntimeError("fixture error")))
        self.assertEqual(p.read_json(self.root / "train/summary.json")["status"], "failed")
        self.assertEqual(run.finish.call_args.kwargs["state"], "crashed")
        self.assertIn("fixture://failed", (self.root / "HISTORY.md").read_text())

    def test_swanlab_settings_accept_legacy_plain_project_without_creating_run(self):
        from swanlab.sdk.internal.settings import Settings

        run = Mock(url="fixture://not-created")
        fake = Mock()

        def check_settings(**kwargs):
            self.assertNotIn("SWANLAB_PROJECT", os.environ)
            self.assertEqual(kwargs["project"], "trd-math-opd")
            Settings()  # The real SDK parser; no experiment or network request.
            return run

        fake.init.side_effect = check_settings
        with patch.dict(os.environ, {"SWANLAB_PROJECT": "trd-math-opd"}), \
                patch.dict(sys.modules, {"swanlab": fake}), patch.object(p, "ROOT", self.root):
            for stage in ("train", "eval"):
                result = p.experiment(stage, "fixture", self.root / stage, {}, lambda run: {"fixture": True})
                self.assertEqual(result["status"], "completed")
                self.assertEqual(os.environ["SWANLAB_PROJECT"], "trd-math-opd")
        self.assertEqual(fake.init.call_count, 2)

    def eval_fixture(self, incomplete=False, missing_metric=False):
        import pyarrow as pa
        import pyarrow.parquet as pq

        folder = self.root / "eval"
        folder.mkdir()
        model = self.model()
        (folder / "generations").mkdir()
        entry = {}
        for name in p.BENCHMARKS:
            responses = ["2"] * (15 if incomplete and name == "aime24" else 16)
            pq.write_table(pa.Table.from_pylist([{"responses": responses}]),
                           folder / "generations" / f"{name}_pass16_generation.parquet")
            for key in ("pass16", "pass8", "avg_pass1"):
                entry[f"{name}_{key}_generation_pass_16"] = 0.5
        if missing_metric:
            entry.pop("aime24_pass16_generation_pass_16")
        p.write_json(folder / "results.json", {"fixture-distilled": entry})
        run = Mock()
        with patch.object(p, "build_env", return_value=self.values), patch.object(p, "run_command"), \
                patch.object(p, "experiment", side_effect=lambda stage, rid, out, cfg, action: action(run)):
            result = p.evaluation("fixture", folder, model, None, {name: {"rows": 1} for name in p.BENCHMARKS})
        return result, run

    def test_evaluation_uploads_all_selected_benchmark_metrics(self):
        result, run = self.eval_fixture()
        self.assertEqual(len(result["metrics"]), 3 * len(p.BENCHMARKS))
        self.assertEqual(result["metrics"]["eval/aime24/pass@16"], 0.5)
        run.log.assert_called_once_with(result["metrics"])

    def test_incomplete_evaluation_cannot_succeed(self):
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            self.eval_fixture(incomplete=True)

    def test_missing_benchmark_metric_cannot_succeed(self):
        with self.assertRaises(KeyError):
            self.eval_fixture(missing_metric=True)


class FrozenDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import datasets
        import pyarrow as pa
        import pyarrow.parquet as pq

        datasets.disable_progress_bars()
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.rows = {}
        cls.manifest = {"completed": True, "config": {"split_version": "fixture"}, "splits": {}}
        for name, count in (("train_9984", 9984), ("validation", 512)):
            folder = cls.root / name
            folder.mkdir()
            rows = [{"raw_id": f"{name}-{i}", "sample_id": f"sample-{name}-{i}",
                     "duplicate_group_id": f"group-{name}-{i}", "split_version": "fixture",
                     "problem": "1+1", "answer": "2", "solution": "2"} for i in range(count)]
            cls.rows[name] = rows
            datasets.Dataset.from_list(rows).save_to_disk(str(folder / "dataset"))
            pq.write_table(pa.Table.from_pylist(rows), folder / "data.parquet")
            (folder / "ids.json").write_text(json.dumps([r["raw_id"] for r in rows]))
            (folder / "sample_ids.json").write_text(json.dumps([r["sample_id"] for r in rows]))
            cls.manifest["splits"][name] = {"rows": count,
                "parquet_sha256": p.file_hash(folder / "data.parquet"),
                "ids_sha256": p.file_hash(folder / "ids.json"),
                "sample_ids_sha256": p.file_hash(folder / "sample_ids.json")}
        p.write_json(cls.root / "manifest.json", cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.env = patch.dict(os.environ, {"MATH_SPLIT_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.values = {"TRAIN_DATA_PATH": str(self.root / "train_9984/dataset"), "DATA_PATH": "", "MAX_SAMPLES": "",
                       "CORRECTED_RESPONSES_PATH": "", "NGPUS_PER_NODE": "2", "NNODES": "1", "SP_SIZE": "1",
                       "TRAIN_BATCH_SIZE": "2", "GRADIENT_ACCUMULATION_STEPS": "32"}

    def test_frozen_data_hashes_hf_contents_and_batch_count(self):
        result = p.validate_train_split(self.values)
        self.assertEqual(result["rows"], 9984)
        self.assertEqual(result["effective_batch"], 128)
        self.assertEqual(result["nominal_updates_per_epoch"], 78)

    def test_original_full_dataset_is_rejected(self):
        self.values["TRAIN_DATA_PATH"] = str(self.root / "original-full-dataset")
        with self.assertRaisesRegex(ValueError, "frozen 9984 subset"):
            p.validate_train_split(self.values)

    def test_pilot_sample_limit_or_external_parquet_is_rejected(self):
        for key in ("MAX_SAMPLES", "DATA_PATH", "CORRECTED_RESPONSES_PATH"):
            values = dict(self.values, **{key: "fixture"})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                p.validate_train_split(values)

    def test_tampered_frozen_manifest_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["splits"]["train_9984"]["parquet_sha256"] = "bad"
        with patch.object(p, "read_json", return_value=manifest), self.assertRaisesRegex(ValueError, "checksum"):
            p.validate_train_split(self.values)

    def test_generated_data_must_contain_exactly_the_frozen_ids(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        generated = [{"extra_info": r, "responses": ["2"]} for r in self.rows["train_9984"]]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "generated.parquet"
            pq.write_table(pa.Table.from_pylist(generated), path)
            self.assertEqual(p.validate_generated(path, self.values)["rows"], 9984)
            generated[0] = {"extra_info": self.rows["validation"][0], "responses": ["2"]}
            pq.write_table(pa.Table.from_pylist(generated), path)
            with self.assertRaisesRegex(ValueError, "identity/order"):
                p.validate_generated(path, self.values)
            pq.write_table(pa.Table.from_pylist(generated[:8]), path)
            with self.assertRaisesRegex(ValueError, "all 9984"):
                p.validate_generated(path, self.values)


class LoRAExportTests(unittest.TestCase):
    def test_plain_hf_export_includes_nonzero_adapter_and_preserves_alpha(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
        from verl.model_merger.base_model_merger import ModelMergerConfig
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger

        torch.set_num_threads(2)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temp:
                torch.manual_seed(42)
                config = Qwen3Config(vocab_size=23, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                    max_position_embeddings=64, tie_word_embeddings=False)
                config.architectures = ["Qwen3ForCausalLM"]
                config._attn_implementation = "eager"
                base = Qwen3ForCausalLM(config).to(dtype).eval()
                ids = torch.tensor([[1, 3, 4]])
                with torch.inference_mode():
                    original = base(ids).logits.float().clone()
                model = get_peft_model(base, LoraConfig(r=2, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                       task_type="CAUSAL_LM")).eval()
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "lora_" in name:
                            param.fill_(0.15)
                with torch.inference_mode():
                    expected = model(ids).logits.float().clone()
                self.assertGreater((expected - original).abs().max().item(), 1e-4)
                folder = Path(temp)
                model.generation_config.save_pretrained(folder)
                (folder / "lora_train_meta.json").write_text('{"r":2,"lora_alpha":8,"task_type":"CAUSAL_LM"}')
                output = folder / "hf"
                output.mkdir()
                merger = FSDPModelMerger.__new__(FSDPModelMerger)
                merger.config = ModelMergerConfig(operation="merge", backend="fsdp", local_dir=temp,
                                                   target_dir=str(output), merge_lora=True)
                merger.model_config = config
                merger.hf_model_config_path = temp
                with patch.object(merger, "save_hf_processing_assets"):
                    merger.save_hf_model_and_tokenizer(model.state_dict())
                loaded = AutoModelForCausalLM.from_pretrained(output, torch_dtype=dtype, device_map={"": "cpu"}).eval()
                with torch.inference_mode():
                    actual = loaded(ids).logits.float()
                torch.testing.assert_close(actual, expected, rtol=0.03 if dtype == torch.bfloat16 else 2e-5,
                                           atol=0.002 if dtype == torch.bfloat16 else 2e-6)
                self.assertTrue(p.read_json(output / "lora_merge.json")["merged"])
                self.assertEqual(p.read_json(output / "lora_adapter/adapter_config.json")["lora_alpha"], 8)
                p.validate_model(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
