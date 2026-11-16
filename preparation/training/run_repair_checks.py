"""Track CPU regression tests and a real cached-target token audit online."""
import datetime as dt
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparation.training import pipeline as p
from preparation.training.cached_repair import materialize, validate_source


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from recipe.opd.dataset.data_utils import KLTrainingDataset

    values = p.build_env()
    split = p.validate_train_split(values)
    source = ROOT / "preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json"
    provenance = validate_source(source, values, split)
    run_id = "chat-repair-cpu-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = ROOT / "preparation/logs/training-development" / run_id

    def action(run):
        p.run_command([sys.executable, "-m", "unittest", "-v", "preparation.training.test_chat_repair",
                       "preparation.training.test_pipeline", "preparation.training.test_base_evaluation",
                       "preparation.training.test_runtime_control"], folder, run)
        manifest = materialize(provenance, folder / "cache/chat-training.parquet", values["MODEL_PATH"])
        student = AutoTokenizer.from_pretrained(values["MODEL_PATH"], local_files_only=True)
        teacher = AutoTokenizer.from_pretrained(values["TEACHER_MODEL_PATH"], local_files_only=True)
        ds = KLTrainingDataset(manifest["derived_cache"], student, teacher_tokenizer=teacher,
            sequence_format="chat", kl_type="forward", max_length=int(values["MAX_LENGTH"]),
            use_initial_response=True, prompt_truncation=True, distill_mode="opd")
        indices = np.random.default_rng(20261116).choice(len(ds), 512, replace=False)
        eos_rows, truncated = 0, 0
        for i in indices:
            item = ds[int(i)]
            next_ids = []
            for side in ("student", "teacher"):
                selected = item[f"{side}_loss_mask"].bool().nonzero().flatten()
                assert selected.numel() > 0
                assert selected[-1] + 1 < item[f"{side}_input_ids"].numel()
                next_ids.append(item[f"{side}_input_ids"][selected + 1])
            assert torch.equal(*next_ids)
            eos_rows += next_ids[0][-1].item() == student.eos_token_id
            row = ds.data[int(i)]
            expected = len(student.encode(row["responses"][0], add_special_tokens=False)) + int(row["extra_info"]["response_complete"])
            truncated += next_ids[0].numel() < expected
        report = {"sample_rows": len(indices), "seed": 20261116, "aligned_causal_targets": True,
                  "eos_rows_in_audit": eos_rows, "target_truncated_rows_in_audit": truncated,
                  "cache": manifest, "new_generation": False, "new_model_inference": False}
        p.write_json(folder / "validation.json", report)
        run.log({"audit/aligned_rows": len(indices), "audit/eos_rows": eos_rows,
                 "audit/target_truncated_rows": truncated, "audit/summary": __import__("swanlab").Text(json.dumps(report))})
        return report

    p.experiment("数值校验", run_id, folder, {"training": values, "cache_source": provenance,
                 "real_production_weight_inference": False}, action,
                 name=p.unique_run_name(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d") +
                                        "-数值校验-聊天格式和缓存复用", folder.parent))


if __name__ == "__main__":
    main()
