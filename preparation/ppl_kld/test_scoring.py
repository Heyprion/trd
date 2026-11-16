"""CPU regression fixtures only: no production weights, CUDA, or real validation scores."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import evaluate
from scoring import SUM_KEYS, aggregate, encode_row, make_batches, paired_comparison, score_batch

torch.set_num_threads(2)


def tiny_model(seed):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=23, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         attention_dropout=0.0, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def rows_fixture():
    return [{"sample_id": "one", "raw_id": "raw-one", "duplicate_group_id": "group-one", "domain": "algebra",
             "difficulty": "easy", "input_ids": [1, 3, 4, 5, 2], "prompt_tokens": 2, "response_tokens": 3,
             "tokens_sha256": "fixture-one"},
            {"sample_id": "two", "raw_id": "raw-two", "duplicate_group_id": "group-two", "domain": "geometry",
             "difficulty": "hard", "input_ids": [1, 6, 7, 8], "prompt_tokens": 3, "response_tokens": 1,
             "tokens_sha256": "fixture-two"}]


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.student, self.teacher = tiny_model(1), tiny_model(2)
        self.rows = rows_fixture()

    @torch.inference_mode()
    def ordinary_reference(self, student, teacher):
        # Independently enumerated positions include the first response token and final EOS.
        ids = torch.tensor([[1, 3, 4, 5, 2], [1, 6, 7, 8, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        s_logits = student(ids, attention_mask=mask).logits
        t_logits = teacher(ids, attention_mask=mask).logits
        s, t = s_logits.double().log_softmax(-1), t_logits.double().log_softmax(-1)
        first = [(0, 1, 4), (0, 2, 5), (0, 3, 2)]
        second = [(1, 2, 8)]
        expected = []
        for positions in (first, second):
            values = []
            for b, p, target in positions:
                values.append([-s[b, p, target], -t[b, p, target],
                               (t[b, p].exp() * (t[b, p] - s[b, p])).sum(),
                               (s[b, p].exp() * (s[b, p] - t[b, p])).sum()])
            expected.append(torch.tensor(values).sum(0))
        labels = torch.tensor([[-100, -100, 4, 5, 2], [-100, -100, -100, 8, -100]])
        return torch.stack(expected), student(ids, attention_mask=mask, labels=labels).loss

    def test_full_hf_loss_and_fp64_kl_reference(self):
        expected, hf_loss = self.ordinary_reference(self.student, self.teacher)
        got = score_batch(self.student, self.teacher, self.rows, 0, 2)
        for i, row in enumerate(got):
            torch.testing.assert_close(torch.tensor([row[k] for k in SUM_KEYS], dtype=torch.float64),
                                       expected[i].double(), rtol=2e-5, atol=1e-6)
        self.assertAlmostEqual(aggregate(got)["student_nll"], hf_loss.item(), places=6)
        self.assertEqual(aggregate(got)["response_tokens"], 4)

    def test_chunk_and_padding_invariance(self):
        full = score_batch(self.student, self.teacher, self.rows, 0, 100)
        chunked = score_batch(self.student, self.teacher, self.rows, 0, 1)
        single = [score_batch(self.student, self.teacher, [r], 0, 2)[0] for r in self.rows]
        for left, middle, right in zip(full, chunked, single):
            for key in SUM_KEYS:
                self.assertAlmostEqual(left[key], middle[key], places=6)
                self.assertAlmostEqual(left[key], right[key], places=6)

    def test_equal_models_zero_kl(self):
        got = aggregate(score_batch(self.student, self.student, self.rows, 0, 2))
        self.assertEqual(got["forward_kl"], 0)
        self.assertEqual(got["reverse_kl"], 0)

    def test_bf16_models_use_fp32_probability_reduction(self):
        self.student.to(torch.bfloat16)
        self.teacher.to(torch.bfloat16)
        expected, _ = self.ordinary_reference(self.student, self.teacher)
        got = score_batch(self.student, self.teacher, self.rows, 0, 2)
        for i, row in enumerate(got):
            torch.testing.assert_close(torch.tensor([row[k] for k in SUM_KEYS], dtype=torch.float64),
                                       expected[i].double(), rtol=2e-5, atol=1e-6)

    def test_lora_load_and_head_chunking(self):
        from peft import LoraConfig, get_peft_model
        with tempfile.TemporaryDirectory() as temp:
            base, adapter = Path(temp) / "base", Path(temp) / "adapter"
            self.student.save_pretrained(base)
            lora = get_peft_model(copy.deepcopy(self.student), LoraConfig(r=2, lora_alpha=4,
                                  target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            with torch.no_grad():
                for name, parameter in lora.named_parameters():
                    if "lora_B" in name:
                        parameter.fill_(0.05)
            lora.eval().save_pretrained(adapter)
            loaded = evaluate.load_model(adapter, base, "eager", device="cpu", dtype=torch.float32)
            self.assertEqual({p.device.type for p in loaded.parameters()}, {"cpu"})
            expected, _ = self.ordinary_reference(loaded, self.teacher)
            got = score_batch(loaded, self.teacher, self.rows, 0, 2)
            for i, row in enumerate(got):
                torch.testing.assert_close(torch.tensor([row[k] for k in SUM_KEYS], dtype=torch.float64),
                                           expected[i].double(), rtol=2e-5, atol=1e-6)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

    def test_global_token_weighting_not_average_ppl(self):
        rows = [{"response_tokens": n, **{k: value for k in SUM_KEYS}} for n, value in ((1, 1), (3, 6))]
        result = aggregate(rows)
        self.assertEqual(result["student_nll"], 1.75)
        self.assertAlmostEqual(result["student_ppl"], math.exp(1.75))
        self.assertNotAlmostEqual(result["student_ppl"], (math.exp(1) + math.exp(2)) / 2)

    def test_nonfinite_rejected(self):
        with torch.no_grad():
            next(self.student.parameters()).fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            score_batch(self.student, self.teacher, self.rows, 0, 2)

    def test_batch_budget_and_oversized_sample(self):
        self.assertEqual(len(make_batches(self.rows, 8, 8)), 2)
        with self.assertRaises(ValueError):
            make_batches(self.rows, 8, 4)

    def test_empty_reference_and_overflow_fail(self):
        class Tokenizer:
            def encode(self, text, **_):
                return list(range(len(text)))
        row = {**self.rows[0], "problem": "x", "solution": "solution"}
        with self.assertRaises(ValueError):
            encode_row({**row, "solution": " "}, Tokenizer(), 1000)
        with self.assertRaises(ValueError):
            encode_row(row, Tokenizer(), 3)

    def test_pairing_detects_prefix_change_and_constant_delta(self):
        before = score_batch(self.student, self.teacher, self.rows, 0, 2)
        after = copy.deepcopy(before)
        for row in after:
            row["student_nll_sum"] -= 0.1 * row["response_tokens"]
        result = paired_comparison(before, list(reversed(after)), 100)
        self.assertAlmostEqual(result["metrics"]["student_nll"]["delta"], -0.1)
        for bound in result["metrics"]["student_nll"]["delta_ci95"]:
            self.assertAlmostEqual(bound, -0.1)
        after[0]["tokens_sha256"] = "changed"
        with self.assertRaises(ValueError):
            paired_comparison(before, after, 100)

    def test_comparison_rejects_changed_teacher_or_protocol(self):
        current = {"protocol_sha256": "a", "models": {"teacher": {"content_sha256": "teacher"}}}
        baseline = {"config": copy.deepcopy(current)}
        evaluate.check_comparable(current, baseline)
        baseline["config"]["models"]["teacher"]["content_sha256"] = "other"
        with self.assertRaises(ValueError):
            evaluate.check_comparable(current, baseline)
        baseline["config"]["protocol_sha256"] = "changed"
        with self.assertRaises(ValueError):
            evaluate.check_comparable(current, baseline, check_teacher=False)


class FrozenDataTests(unittest.TestCase):
    def create_split(self, root, *, overlap=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import Dataset
        validation = [{"raw_id": "raw-a", "sample_id": "a", "duplicate_group_id": "group-a", "problem": "x",
                       "solution": "s", "domain": "algebra", "difficulty": "easy"}]
        train = [{"raw_id": "raw-b", "sample_id": "b", "duplicate_group_id": "group-a" if overlap else "group-b"}]
        manifest = {"completed": True, "splits": {}}
        for name, rows in (("validation", validation), ("train_pool", train)):
            folder = root / name
            folder.mkdir()
            pq.write_table(pa.Table.from_pylist(rows), folder / "data.parquet")
            (folder / "ids.json").write_text(json.dumps([r["raw_id"] for r in rows]))
            (folder / "sample_ids.json").write_text(json.dumps([r["sample_id"] for r in rows]))
            manifest["splits"][name] = {"rows": len(rows), "parquet_sha256": evaluate.file_hash(folder / "data.parquet"),
                                       "ids_sha256": evaluate.file_hash(folder / "ids.json"),
                                       "sample_ids_sha256": evaluate.file_hash(folder / "sample_ids.json")}
            if name == "validation":
                Dataset.from_list(rows).save_to_disk(str(folder / "dataset"))
        evaluate.write_json(root / "manifest.json", manifest)

    def test_frozen_files_and_hf_directory_checked(self):
        from datasets import Dataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_split(root)
            rows, _ = evaluate.load_validation(root)
            self.assertEqual(len(rows), 1)
            changed = [{**rows[0], "solution": "changed"}]
            import shutil
            shutil.rmtree(root / "validation/dataset")
            Dataset.from_list(changed).save_to_disk(str(root / "validation/dataset"))
            with self.assertRaisesRegex(ValueError, "HF directory"):
                evaluate.load_validation(root)

    def test_overlap_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_split(root, overlap=True)
            with self.assertRaisesRegex(ValueError, "isolated"):
                evaluate.load_validation(root)
            (root / "validation/ids.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluate.load_validation(root)


class NamingTests(unittest.TestCase):
    def test_date_stage_purpose(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(evaluate.swanlab_run_name("distilled", temp, "20261116"),
                             "20261116-评测-PPL和KL-蒸馏后")

    def test_reruns_count_only_same_date_and_purpose(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("distilled-20261116-first", "baseline-20261116-first", "distilled-20261115-first"):
                folder = root / name
                folder.mkdir()
                (folder / "swanlab.json").write_text("{}")
            (root / "distilled-20261116-not-started").mkdir()
            self.assertEqual(evaluate.swanlab_run_name("distilled", root, "20261116"),
                             "20261116-评测-PPL和KL-蒸馏后-r02")


class ParentPipelineTests(unittest.TestCase):
    def config_fixture(self, arguments):
        rows = rows_fixture()
        args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(arguments).items()}
        return {"arguments": args, "data": {"full_validation": True, "frozen_rows": 2,
                                             "response_tokens": 4},
                "models": {}, "protocol_sha256": "synthetic-fixed-protocol",
                "execution": {"pad_token_id": 0}}, rows, [rows]

    def test_artifacts_baseline_pointer_paired_report_and_failure(self):
        import contextlib
        import io
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, teacher = root / "base", root / "teacher"
            student, fixed_teacher = tiny_model(1), tiny_model(2)
            student.save_pretrained(base)
            fixed_teacher.save_pretrained(teacher)
            records = score_batch(student, fixed_teacher, rows_fixture(), 0, 2)
            cli = ["--base-model", str(base), "--student-model", str(base), "--teacher-model", str(teacher),
                   "--output-root", str(root / "results"), "--history-file", str(root / "HISTORY.md")]
            fake_run = SimpleNamespace(url="https://synthetic.example/run", id="fixture")
            with patch.object(evaluate, "preflight", side_effect=self.config_fixture), \
                 patch.object(evaluate, "run_workers", return_value=(records, [])) as workers, \
                 patch("swanlab.init", return_value=fake_run) as init, \
                 patch("swanlab.log"), patch("swanlab.finish") as finish, contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(cli)
                pointer = root / "results/latest-baseline.json"
                old_pointer = pointer.read_text()
                baseline_summary, baseline_records, _ = evaluate.load_baseline(pointer)
                self.assertEqual(baseline_summary["overall"]["response_tokens"], 4)
                self.assertEqual(len(baseline_records), 2)
                self.assertEqual(init.call_args.kwargs["mode"], "online")
                evaluate.main(cli + ["--label", "distilled", "--compare-to", str(pointer)])
                after = next((root / "results").glob("distilled-*/summary.json"))
                self.assertEqual(evaluate.read_json(after)["comparison"]["metrics"]["student_ppl"]["delta"], 0)
                self.assertIn("Delta 95% CI", (after.parent / "REPORT.md").read_text())
                self.assertEqual(pointer.read_text(), old_pointer)
                workers.side_effect = RuntimeError("synthetic worker failure")
                with self.assertRaisesRegex(RuntimeError, "synthetic worker"):
                    evaluate.main(cli)
                self.assertTrue(any((root / "results").glob("baseline-*/failure.json")))
                self.assertEqual(pointer.read_text(), old_pointer)
                self.assertEqual(finish.call_args.kwargs["state"], "crashed")
                self.assertIn("失败/中断", (root / "HISTORY.md").read_text())

    def test_calibration_cannot_replace_complete_baseline(self):
        import contextlib
        import io
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            student = tiny_model(1)
            student.save_pretrained(model)
            records = score_batch(student, student, rows_fixture(), 0, 2)
            def partial_config(a):
                config, rows, batches = self.config_fixture(a)
                config["data"]["full_validation"] = False
                config["data"]["frozen_rows"] = 512
                return config, rows, batches
            with patch.object(evaluate, "preflight", side_effect=partial_config), \
                 patch.object(evaluate, "run_workers", return_value=(records, [])), \
                 patch("swanlab.init", return_value=SimpleNamespace(url="https://synthetic.example/run", id="fixture")), \
                 patch("swanlab.log"), patch("swanlab.finish"), contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(["--base-model", str(model), "--student-model", str(model), "--teacher-model", str(model),
                               "--output-root", str(root / "results"), "--history-file", str(root / "HISTORY.md"),
                               "--max-samples", "2"])
                self.assertFalse((root / "results/latest-baseline.json").exists())

    def test_worker_assignment_is_disjoint_and_honors_gpu_mask(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            a = SimpleNamespace(gpu_ids=["GPU-a", "GPU-b"])
            rows = rows_fixture()
            assignments = []
            def fake_process(command, env, **_):
                assignment = evaluate.read_json(command[5])
                rank = int(command[7])
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], a.gpu_ids[rank])
                assignments.extend(r["sample_id"] for batch in assignment for r in batch)
                (output / f"rank-{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for batch in assignment for row in batch))
                evaluate.write_json(output / f"worker-{rank}.json", {"rows": sum(map(len, assignment))})
                return SimpleNamespace(poll=lambda: 0, wait=lambda **_: 0)
            with patch.object(evaluate.subprocess, "Popen", side_effect=fake_process):
                records, workers = evaluate.run_workers(a, {}, [[r] for r in rows], output, lambda _: None)
            self.assertEqual(sorted(assignments), ["one", "two"])
            self.assertEqual(len(records), 2)
            self.assertEqual(len(workers), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
