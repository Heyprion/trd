"""Regression checks for cached chat conditioning and causal response alignment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from preparation.training import cached_repair as cache
from preparation.training import pipeline
from recipe.opd.dataset.data_utils import KLTrainingDataset

ROOT = Path(__file__).resolve().parents[2]


class ChatRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.student = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-4B-Instruct-2507", local_files_only=True)
        cls.teacher = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-8B", local_files_only=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def dataset(self, *, complete=True, response="<think>\nreason\n</think>\n\n\\boxed{2}", limit=256, teacher=None, fmt="chat"):
        self.sp = "Compute 1+1. Please reason step by step."
        self.tp = "\nRewrite this solution.\n**Your Initial Solution:**\n1+1=2\n**Instructions:**\nKeep correct steps.\n "
        row = {"prompt": [{"role": "user", "content": self.tp}], "responses": [response],
               "extra_info": {"problem": "Compute 1+1.", "initial_response": "1+1=2",
                              "student_prompt": self.sp, "response_complete": complete}}
        path = self.folder / "input.parquet"
        pq.write_table(pa.Table.from_pylist([row]), path)
        return KLTrainingDataset(str(path), self.student, kl_type="forward", max_length=limit,
                                 use_initial_response=True, prompt_truncation=True, distill_mode="opd",
                                 sequence_format=fmt, teacher_tokenizer=teacher or self.teacher)

    @staticmethod
    def next_tokens(item, side):
        indices = item[f"{side}_loss_mask"].bool().nonzero().flatten()
        return item[f"{side}_input_ids"][indices + 1]

    def test_exact_generation_prefix_and_unchanged_target(self):
        ds = self.dataset()
        item = ds[0]
        target = self.student.encode(ds.data[0]["responses"][0], add_special_tokens=False) + [self.student.eos_token_id]
        for side, tok, prompt in [("student", self.student, self.sp), ("teacher", self.teacher, self.tp)]:
            prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
            self.assertEqual(item[f"{side}_input_ids"].tolist(), prefix + target)
            self.assertEqual(self.next_tokens(item, side).tolist(), target)
            self.assertEqual(item[f"{side}_loss_mask"].nonzero()[0].item(), len(prefix) - 1)
            self.assertEqual(item[f"{side}_loss_mask"][-1].item(), 0)

    def test_teacher_uses_own_template_and_response_ids_remain_aligned(self):
        teacher = copy.deepcopy(self.teacher)
        teacher.chat_template = "{{ 'teacher-prefix\\n' }}" + teacher.chat_template
        ds = self.dataset(teacher=teacher)
        item = ds[0]
        self.assertTrue(torch.equal(self.next_tokens(item, "student"), self.next_tokens(item, "teacher")))
        prefix = teacher.apply_chat_template([{"role": "user", "content": self.tp}], tokenize=True, add_generation_prompt=True)
        self.assertEqual(item["teacher_input_ids"][:len(prefix)].tolist(), prefix)

    def test_incomplete_response_is_not_taught_to_stop(self):
        ds = self.dataset(complete=False, response="<think>unfinished")
        item = ds[0]
        self.assertNotEqual(self.next_tokens(item, "student")[-1].item(), self.student.eos_token_id)

    def test_eos_is_not_duplicated(self):
        ds = self.dataset(response="\\boxed{2}" + self.student.eos_token)
        self.assertEqual(self.next_tokens(ds[0], "student").tolist().count(self.student.eos_token_id), 1)

    def test_long_teacher_context_keeps_aligned_target(self):
        ds = self.dataset(limit=90)
        ds.data = [{"prompt": [{"role": "user", "content": self.tp.replace("1+1=2", "long solution " * 200)}],
                    "responses": ["<think>reason</think>\n\\boxed{2}"],
                    "extra_info": {"problem": "Compute 1+1.", "initial_response": "long solution " * 200,
                                   "student_prompt": self.sp, "response_complete": True}}]
        item = ds[0]
        self.assertLessEqual(len(item["teacher_input_ids"]), 90)
        self.assertTrue(torch.equal(self.next_tokens(item, "student"), self.next_tokens(item, "teacher")))
        self.assertEqual(self.next_tokens(item, "teacher")[-1].item(), self.teacher.eos_token_id)

    def test_legacy_plain_encoding_is_unchanged(self):
        ds = self.dataset(fmt="plain")
        ids, _, prompt_len = ds._build_sequence("Problem", "Answer", response_complete=True)
        prefix = self.student.encode("Problem\n", add_special_tokens=False)
        self.assertEqual(prompt_len, len(prefix))
        self.assertEqual(ids.tolist(), prefix + self.student.encode("Answer", add_special_tokens=False))

    def test_vocab_mismatch_fails_before_training(self):
        teacher = copy.deepcopy(self.teacher)
        teacher.add_tokens(["<extra-test-token>"])
        with self.assertRaisesRegex(ValueError, "identical token IDs"):
            self.dataset(teacher=teacher)

    def test_causal_mask_agrees_with_hf_response_loss_including_eos(self):
        item = self.dataset(response="\\boxed{2}")[0]
        torch.manual_seed(3)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(self.student), hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=512, tie_word_embeddings=True))
        model.eval()
        inputs = item["student_input_ids"].unsqueeze(0)
        labels = torch.full_like(inputs, -100)
        indices = item["student_loss_mask"].bool().nonzero().flatten()
        labels[0, indices + 1] = inputs[0, indices + 1]
        with torch.no_grad():
            result = model(inputs, labels=labels)
            selected = result.logits[0, indices].float().log_softmax(-1)
            expected = -selected.gather(1, inputs[0, indices + 1, None]).mean()
        self.assertAlmostEqual(result.loss.item(), expected.item(), places=5)

    def test_derived_cache_preserves_text_and_marks_uncertain_completion(self):
        target = self.folder / "teacher.parquet"
        initial = self.folder / "student.parquet"
        texts = ["<think>reason</think>\n\\boxed{2}", "<think>unfinished", "word " * 600]
        rows = [{"prompt": [{"role": "user", "content": "teacher"}], "responses": [text], "extra_info": {"sample_id": str(i)}} for i, text in enumerate(texts)]
        raw = [{"prompt": [{"role": "user", "content": f"student {i}"}]} for i in range(3)]
        pq.write_table(pa.Table.from_pylist(rows), target)
        pq.write_table(pa.Table.from_pylist(raw), initial)
        provenance = {"teacher_cache": str(target), "teacher_cache_sha256": pipeline.file_hash(target),
                      "student_cache": str(initial), "student_cache_sha256": pipeline.file_hash(initial),
                      "generation_response_limit": 512}
        dest = self.folder / "derived.parquet"
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=self.student):
            manifest = cache.materialize(provenance, dest, "fixture")
        saved = pq.read_table(dest).to_pylist()
        self.assertEqual([r["responses"][0] for r in saved], texts)
        self.assertEqual([r["extra_info"]["response_complete"] for r in saved], [True, False, False])
        self.assertEqual(manifest["complete_rows_with_eos"], 1)
        self.assertEqual(saved[1]["extra_info"]["student_prompt"], "student 1")
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            cache.materialize(provenance, self.folder / "another.parquet", "fixture")

    def test_cached_training_never_calls_generator(self):
        values = {"EPOCH_OUTPUT_DIR": str(self.folder / "outputs"), "EPOCH_MODEL_SAVE_DIR": str(self.folder / "weights"),
                  "MODEL_PATH": "original-student", "TRAINING_DATA_PATH": "old-generated-cache"}
        provenance = {"source_summary": "completed.json"}
        commands = []
        with patch.dict("os.environ", {"KL_FULL_VOCAB_CHUNK_SIZE": "512"}), \
                patch.object(cache, "materialize", return_value={"rows": 9984, "complete_rows_with_eos": 9900}), \
                patch.object(pipeline, "validate_generated", return_value={"rows": 9984, "sha256": "fixture"}), \
                patch.object(pipeline, "validate_model"), \
                patch.object(pipeline, "experiment", side_effect=lambda stage, rid, folder, cfg, action, **kw: action(Mock(url="fixture://repair"))), \
                patch.object(pipeline, "run_command", side_effect=lambda cmd, folder, run, env: commands.append((cmd, env.copy()))):
            result = pipeline.train("fixture", self.folder, values, {"rows": 9984}, cache_source=provenance)
        self.assertEqual(len(commands), 3)
        self.assertTrue(all("prepare_training_data.py" not in " ".join(c) for c, _ in commands))
        self.assertTrue(all(e["DATA_PATH"].endswith("cache/chat-training.parquet") for _, e in commands))
        self.assertEqual(result["cache_reuse"]["rows"], 9984)

    def test_source_checksum_and_model_identity_are_pinned(self):
        model = self.folder / "original-student"
        model.mkdir()
        initial = self.folder / "math_y_o_responses.parquet"
        target = self.folder / "teacher.parquet"
        extra = {"sample_id": "one", "raw_id": "raw", "duplicate_group_id": "group",
                 "split_version": "split", "problem": "1+1?", "initial_response": "2"}
        raw = {"extra_info": extra, "responses": ["2"], "prompt": [{"role": "user", "content": "1+1?"}]}
        row = {"extra_info": extra, "responses": ["\\boxed{2}"],
               "prompt": [{"role": "user", "content": "Problem 1+1? Initial solution: 2"}]}
        raw["reward_model"] = row["reward_model"] = {"ground_truth": "2", "style": "rule"}
        pq.write_table(pa.Table.from_pylist([raw]), initial)
        pq.write_table(pa.Table.from_pylist([row]), target)
        values = {"MODEL_PATH": str(model), "TEACHER_MODEL_PATH": "teacher-model", "DISTILL_MODE": "opd",
                  "Y_MODE": "y_r", "USE_INITIAL_RESPONSE": "true"}
        checked = {"rows": 1, "sha256": pipeline.file_hash(target)}
        source = {"stage": "train", "status": "completed", "swanlab_url": "fixture://source",
                  "config": {"training": {**values, "TRAINING_DATA_PATH": str(target), "MAX_RESPONSE_LENGTH": "16384"},
                             "split": {"manifest_sha256": "frozen"}}, "result": {"training_data": checked}}
        path = self.folder / "summary.json"
        pipeline.write_json(path, source)
        with patch.object(cache, "validate_generated", return_value=checked):
            result = cache.validate_source(path, values, {"manifest_sha256": "frozen"})
            self.assertFalse(result["new_generation"])
            with self.assertRaisesRegex(ValueError, "different MODEL_PATH"):
                cache.validate_source(path, {**values, "MODEL_PATH": "distilled"}, {"manifest_sha256": "frozen"})
            with self.assertRaisesRegex(ValueError, "different frozen split"):
                cache.validate_source(path, values, {"manifest_sha256": "changed"})
        with patch.object(cache, "validate_generated", return_value={**checked, "sha256": "changed"}), \
                self.assertRaisesRegex(ValueError, "pinned checksum"):
            cache.validate_source(path, values, {"manifest_sha256": "frozen"})

    def test_smaller_storage_reserve_requires_the_measured_single_save_profile(self):
        from preparation.stage_outputs import minimum_space
        values = {"MODEL_STORAGE_DTYPE": "bf16", "USE_LORA": "true", "LORA_RANK": "64", "LORA_ALPHA": "128",
                  "TRAIN_EPOCHS_PER_ROUND": "1", "TRAIN_BATCH_SIZE": "2", "GRADIENT_ACCUMULATION_STEPS": "32",
                  "NGPUS_PER_NODE": "2", "NNODES": "1", "SP_SIZE": "1", "SAVE_STEPS": "100",
                  "MODEL_NAME": "Qwen3-4B-Instruct-2507"}
        with patch.dict("os.environ", {"OUTPUT_SPACE_PROFILE": ""}):
            self.assertEqual(minimum_space(values), (20, 18))
        with patch.dict("os.environ", {"OUTPUT_SPACE_PROFILE": "balanced-single-save"}):
            self.assertEqual(minimum_space(values), (12, 11))
            with self.assertRaises(ValueError):
                minimum_space({**values, "SAVE_STEPS": "20"})
            with self.assertRaises(ValueError):
                minimum_space({**values, "LORA_RANK": "128"})


if __name__ == "__main__":
    unittest.main()
