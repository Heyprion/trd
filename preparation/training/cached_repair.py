"""Validate and derive chat-training input from a completed, frozen TRD run."""
from pathlib import Path

from preparation.training.pipeline import file_hash, read_json, validate_generated, write_json


def validate_source(summary_path, values, split):
    import pyarrow.parquet as pq

    summary_path = Path(summary_path).expanduser().resolve()
    source = read_json(summary_path)
    if source.get("stage") != "train" or source.get("status") != "completed":
        raise ValueError("Cache source must be a completed training summary")
    config = source["config"]["training"]
    for key in ("MODEL_PATH", "TEACHER_MODEL_PATH"):
        if Path(config[key]).resolve() != Path(values[key]).resolve():
            raise ValueError(f"Cached trajectories were produced by a different {key}")
    if values["DISTILL_MODE"] != "opd" or values["Y_MODE"] != "y_r" or values["USE_INITIAL_RESPONSE"] != "true":
        raise ValueError("Cached chat repair requires refine OPD on y_r")
    if source["config"]["split"]["manifest_sha256"] != split["manifest_sha256"]:
        raise ValueError("Cached run uses a different frozen split")
    if (Path(values["MODEL_PATH"]) / "adapter_config.json").exists() or (
        Path(values["MODEL_PATH"]) / "lora_merge.json").exists():
        raise ValueError("Repair must initialize from the original student, not a distilled model")
    target = Path(config["TRAINING_DATA_PATH"]).resolve()
    checked = validate_generated(target, values)
    if checked != source["result"]["training_data"]:
        raise ValueError("Teacher cache differs from the completed run's pinned checksum")
    initial = target.with_name("math_y_o_responses.parquet")
    raw_rows = pq.read_table(initial).to_pylist()
    refined = pq.read_table(target).to_pylist()
    if len(raw_rows) != len(refined):
        raise ValueError("Student and teacher cache row counts differ")
    for raw, row in zip(raw_rows, refined, strict=True):
        for key in ("sample_id", "raw_id", "duplicate_group_id", "split_version", "problem"):
            if raw["extra_info"][key] != row["extra_info"][key]:
                raise ValueError(f"Student/teacher cache identity differs: {key}")
        if row["extra_info"]["initial_response"] != raw["responses"][0]:
            raise ValueError("Teacher initial response differs from its cached student rollout")
        if row["reward_model"] != raw["reward_model"]:
            raise ValueError("Student/teacher gold answers differ")
        for messages in (raw["prompt"], row["prompt"]):
            if len(messages) != 1 or messages[0]["role"] != "user" or not messages[0]["content"]:
                raise ValueError("Cached generation requires exactly one original user message")
        # The teacher's prompt must actually contain this row's initial response
        # and problem, not just metadata which happens to have matching IDs.
        if raw["responses"][0] not in row["prompt"][0]["content"] or (
            row["extra_info"]["problem"] not in row["prompt"][0]["content"]):
            raise ValueError("Teacher generation prompt does not contain its problem and initial response")
    return {"source_summary": str(summary_path), "source_summary_sha256": file_hash(summary_path),
            "source_swanlab": source["swanlab_url"], "teacher_cache": str(target),
            "teacher_cache_sha256": checked["sha256"], "student_cache": str(initial),
            "student_cache_sha256": file_hash(initial), "rows": checked["rows"],
            "generation_response_limit": int(config["MAX_RESPONSE_LENGTH"]),
            "new_generation": False}


def materialize(provenance, destination, model_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from verl.utils.reward_score.math_reward import last_boxed_only_string

    # Recheck at consumption time; never silently regenerate a missing cache.
    for key in ("student_cache", "teacher_cache"):
        if file_hash(provenance[key]) != provenance[key + "_sha256"]:
            raise ValueError(f"Cached input changed after validation: {key}")
    rows = pq.read_table(provenance["teacher_cache"]).to_pylist()
    initial = pq.read_table(provenance["student_cache"]).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    eos_count = 0
    reasons = {}
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        lengths = tokenizer([r["responses"][0] for r in batch], add_special_tokens=False,
                            return_length=True, return_attention_mask=False)["length"]
        for offset, (row, length) in enumerate(zip(batch, lengths, strict=True)):
            text = row["responses"][0]
            # The old generator saved text, not finish_reason. Infer completion
            # conservatively; never teach EOS to an open thought or capped output.
            reason = "inferred_complete"
            if length >= provenance["generation_response_limit"] - 2:
                reason = "near_generation_limit"
            elif text.count("<think>") != text.count("</think>"):
                reason = "unclosed_thought"
            elif last_boxed_only_string(text) is None:
                reason = "no_parseable_box"
            complete = reason == "inferred_complete"
            row["extra_info"].update(student_prompt=initial[start + offset]["prompt"][0]["content"],
                                     response_complete=complete, completion_basis=reason)
            eos_count += complete
            reasons[reason] = reasons.get(reason, 0) + 1
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Derived cache already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), destination)
    manifest = {**provenance, "derived_cache": str(destination), "derived_cache_sha256": file_hash(destination),
                "sequence_format": "chat", "target_text_changed": False, "filtered_rows": 0,
                "complete_rows_with_eos": eos_count, "completion_basis_counts": reasons,
                "completion_caveat": "Inferred from saved text; original finish_reason was not stored."}
    write_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest
