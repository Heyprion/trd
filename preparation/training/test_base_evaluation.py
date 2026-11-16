"""CPU fixtures for a comparable original-student benchmark entrypoint."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from preparation.training import base_evaluation as b
from preparation.training import pipeline as p


class BaseEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "models/Qwen3-4B-Instruct-2507"
        self.distilled = self.root / "distilled"
        for model in (self.base, self.distilled):
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture")
            (model / "generation_config.json").write_text('{"eos_token_id":1,"top_k":20}')
        self.benchmarks = {}
        self.dataset_dir = self.root / "datasets"
        for name in b.DATASETS:
            path = self.dataset_dir / name / f"{name}_test.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{"prompt": [{"role": "user", "content": "1+1"}],
                                                "reward_model": {"ground_truth": "2"}, "data_source": name}] * 30), path)
            self.benchmarks[name] = {"rows": 30, "sha256": p.file_hash(path)}
        self.metrics = {f"eval/{name}/{metric}": .6 for name in b.DATASETS for metric in ("pass@16", "avg@16", "pass@8")}
        self.reference = {"stage": "eval", "status": "completed", "swanlab_url": "fixture://reference",
                          "config": {"datasets": self.benchmarks, "model_path": str(self.distilled),
                                     "pass_k": 16, "temperature": .6, "top_p": .95, "tp": 2,
                                     "generation": dict(zip(b.GENERATION_KEYS, ("4096", "38912", "43008", ".85", "16", "43008")))},
                          "result": {"metrics": self.metrics}}
        self.reference_path = self.root / "reference.json"
        p.write_json(self.reference_path, self.reference)

    def tokenizer(self, tokens=None):
        tok = Mock()
        tok.get_vocab.return_value = {"fixture": 1}
        tok.special_tokens_map = {"eos_token": "fixture"}
        tok.apply_chat_template.return_value = tokens or [1, 2, 3]
        return tok

    def test_reference_pins_limits_and_rejects_different_inputs_or_protocol(self):
        with patch.dict(os.environ, {"EVAL_RESPONSE_LENGTH": "1024", "EVAL_GEN_TP": "1"}):
            b.load_protocol(self.reference_path, self.benchmarks)
            self.assertEqual(os.environ["EVAL_RESPONSE_LENGTH"], "38912")
            self.assertEqual(os.environ["EVAL_GEN_TP"], "2")
        wrong = copy.deepcopy(self.benchmarks)
        wrong["aime24"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checksums"):
            b.load_protocol(self.reference_path, wrong)
        for field, value in (("status", "running"), ("temperature", .8), ("metric", float("nan"))):
            reference = copy.deepcopy(self.reference)
            if field == "status":
                reference[field] = value
            elif field == "metric":
                reference["result"]["metrics"]["eval/aime24/pass@16"] = value
            else:
                reference["config"][field] = value
            self.reference_path.write_text(json.dumps(reference))
            with self.subTest(field=field), self.assertRaises(ValueError):
                b.load_protocol(self.reference_path, self.benchmarks)

    def test_all_sixty_chat_prompts_and_generation_defaults_are_checked(self):
        left, right = self.tokenizer(), self.tokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=[left, right]):
            proof = b.check_tokenizer_protocol(self.base, self.distilled, self.dataset_dir)
        self.assertEqual(proof["compared_prompts"], 60)
        self.assertEqual(left.apply_chat_template.call_count, 60)
        self.assertEqual(right.apply_chat_template.call_count, 60)
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=[left, self.tokenizer([9])]):
            with self.assertRaisesRegex(ValueError, "chat-template token IDs"):
                b.check_tokenizer_protocol(self.base, self.distilled, self.dataset_dir)
        (self.distilled / "generation_config.json").write_text('{"top_k":10}')
        with self.assertRaisesRegex(ValueError, "generation defaults"):
            b.check_tokenizer_protocol(self.base, self.distilled, self.dataset_dir)

    def test_dry_run_does_not_create_run_or_score_models(self):
        output = self.root / "outputs"
        with patch.object(b, "ROOT", self.root), patch.object(p, "BENCHMARKS", b.DATASETS), \
                patch.dict(os.environ, {"EVAL_DATASETS_DIR": str(self.dataset_dir)}), \
                patch("transformers.AutoTokenizer.from_pretrained", side_effect=[self.tokenizer(), self.tokenizer()]), \
                patch.object(p, "evaluation") as evaluation, contextlib.redirect_stdout(io.StringIO()):
            b.main(["--reference-eval", str(self.reference_path), "--output-root", str(output), "--dry-run"])
        evaluation.assert_not_called()
        self.assertFalse(output.exists())

    def test_base_entrypoint_rejects_a_merged_adapter_model(self):
        (self.base / "lora_adapter").mkdir()
        with patch.object(b, "ROOT", self.root), patch.object(p, "BENCHMARKS", b.DATASETS), self.assertRaises(ValueError):
            b.main(["--reference-eval", str(self.reference_path), "--dry-run"])

    def test_chinese_experiment_name_and_same_day_reruns(self):
        output = self.root / "outputs"
        first = b.run_name(output)
        self.assertRegex(first, r"^\d{8}-评测-AIME24和25-pass16-基线$")
        for folder, name in (("one", first), ("two", first + "-r02")):
            path = output / folder / "eval"
            path.mkdir(parents=True)
            p.write_json(path / "summary.json", {"experiment_name": name})
        self.assertEqual(b.run_name(output), first + "-r03")

    def test_base_result_label_is_separate_and_comparison_uses_percentage_points(self):
        folder = self.root / "eval"
        (folder / "generations").mkdir(parents=True)
        entry = {}
        for name in b.DATASETS:
            pq.write_table(pa.Table.from_pylist([{"responses": ["fixture"] * 16}]),
                           folder / "generations" / f"{name}_pass16_generation.parquet")
            for key in ("pass16", "avg_pass1", "pass8"):
                entry[f"{name}_{key}_generation_pass_16"] = .5
        p.write_json(folder / "results.json", {"fixture_base": entry})
        run = Mock()
        values = {"EVAL_GEN_TP": "2", "MODEL_NAME": "fixture-model"}
        reference = {"summary_path": str(self.reference_path), "metrics": self.metrics}
        with patch.object(p, "BENCHMARKS", b.DATASETS), patch.object(p, "build_env", return_value=values), \
                patch.object(p, "run_command") as command, \
                patch.object(p, "experiment", side_effect=lambda stage, rid, out, cfg, action, **kw: action(run)):
            result = p.evaluation("fixture", folder, self.base, None, {name: {"rows": 1} for name in b.DATASETS},
                                  model_label="base", reference=reference, run_name="20261116-评测-AIME24和25-pass16-基线")
        self.assertEqual(command.call_args.args[3]["EVAL_MODEL_NAME"], "fixture_base")
        self.assertEqual(command.call_args.args[3]["DATASETS"], "aime24 aime25")
        self.assertEqual(result["model_label"], "base")
        comparison = p.read_json(folder / "comparison.json")
        self.assertAlmostEqual(comparison["metrics"]["eval/aime24/pass@16"]["distilled_minus_base_pp"], 10)
        self.assertTrue((folder / "comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
