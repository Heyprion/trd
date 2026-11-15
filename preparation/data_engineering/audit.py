#!/usr/bin/env python3
"""Read-only CPU audit for the data-engineering proposal; never filters inputs.

All row-level artifacts stay in preparation/logs (ignored by Git). Flags and
lexical matches are candidates for review, not labels of defective examples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import swanlab
from transformers import AutoTokenizer
from verl.utils.reward_score.math_reward import is_equiv, last_boxed_only_string, remove_boxed

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def canonical(text: str) -> str:
    # Preserve case, numbers, operators and every LaTeX command.
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def trigrams(text: str) -> set[tuple[str, ...]]:
    # Lowercase only in candidate retrieval; never an automatic deletion key.
    words = re.findall(r"\\[a-zA-Z]+|\d+|[a-zA-Z]+|[^\w\s]", text.lower())
    return set(zip(words, words[1:], words[2:]))


def describe(values: list[int]) -> dict:
    a = np.asarray(values)
    return {"count": len(values), "total": int(a.sum()), "mean": float(a.mean()),
            **{f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 95, 99)},
            "max": int(a.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "preparation/logs/data-engineering/audit-v1")
    parser.add_argument("--omni-jsonl", type=Path)
    parser.add_argument("--omni-revision", default="")
    parser.add_argument("--name", default="data-engineering-readonly-audit-v1")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "data/raw/agentica-org/DeepScaleR-Preview-Dataset/deepscaler.json"
    raw_bytes = raw_path.read_bytes()
    cfg = {"dataset": "agentica-org/DeepScaleR-Preview-Dataset",
           "revision": "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29",
           "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
           "seed": 20261115, "device": "cpu", "mutates_training_data": False,
           "tokenizer": "Qwen3-4B-Instruct-2507", "sample_count": 64,
           "canonicalization": "NFC; remove whitespace; preserve case/operators/numbers/LaTeX",
           "near_eval": "token-trigram Jaccard candidate retrieval; threshold 0.5; no auto decisions",
           "omni_revision": args.omni_revision}
    run = swanlab.init(project="trd-math-opd", name=args.name,
                       job_type="data-audit", mode="online", on_online_error="raise",
                       log_dir=str(out / "swanlab"), config=cfg)
    (out / "swanlab.json").write_text(json.dumps({"url": run.url}, indent=2) + "\n")
    started = time.perf_counter()
    try:
        rows = json.loads(raw_bytes)
        counts = Counter()
        flags = defaultdict(list)
        raw_groups = defaultdict(list)
        groups = defaultdict(list)
        answer_types = Counter()
        box_check = Counter()
        box_candidates = []
        patterns = {
            "asymptote_marker": r"\[asy\]|\\begin\{asy\}",
            "external_image_marker": r"!\[[^\]]*\]\(|<img\b|\\includegraphics|https?://\S+\.(?:png|jpg|jpeg|gif|svg)",
            "diagram_word": r"\b(?:diagram|figure|pictured|shown below)\b",
            "proof_word": r"\b(?:prove|show that)\b",
            "multiple_choice_marker": r"\(A\)|\\text\{\s*\(A\)\s*\}",
            "replacement_character": "\ufffd",
        }
        compiled = {k: re.compile(v, re.I) for k, v in patterns.items()}
        for i, row in enumerate(rows):
            for key in ("problem", "answer", "solution"):
                value = row.get(key)
                if not isinstance(value, str):
                    counts[f"{key}_nonstring"] += 1
                elif not value.strip():
                    counts[f"{key}_empty"] += 1
            p, a, s = (row.get(k) or "" for k in ("problem", "answer", "solution"))
            raw_groups[p].append(i)
            groups[canonical(p)].append(i)
            for key, pattern in compiled.items():
                if pattern.search(p):
                    flags[key].append(i)
            a = a.strip()
            if re.fullmatch(r"[+-]?\d+", a):
                kind = "integer"
            elif re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", a):
                kind = "decimal"
            elif re.fullmatch(r"(?:\\text\{|\\mathrm\{)?\(?[A-E]\)?\}?", a):
                kind = "choice_letter_candidate"
            else:
                kind = "expression_or_other"
            answer_types[kind] += 1
            if s.strip():
                box = last_boxed_only_string(s)
                if box is None:
                    box_check["nonempty_solution_without_extractable_last_box"] += 1
                else:
                    try:
                        extracted = remove_boxed(box)
                        if is_equiv(extracted, a):
                            box_check["last_box_equivalent_by_existing_grader"] += 1
                        else:
                            box_check["last_box_not_proven_equivalent"] += 1
                            box_candidates.append({"index": i, "answer": a, "last_box": extracted})
                    except (AssertionError, ValueError, IndexError):
                        box_check["last_box_extraction_error"] += 1

        duplicates = [v for v in groups.values() if len(v) > 1]
        conflicts = [v for v in duplicates if len({canonical(rows[i]["answer"]) for i in v}) > 1]
        print("Loaded rows and lexical flags", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-4B-Instruct-2507", local_files_only=True)
        lengths = {}
        for key in ("problem", "answer", "solution"):
            values = [r[key] for r in rows]
            lens = []
            for start in range(0, len(values), 512):
                lens.extend(map(len, tokenizer(values[start:start+512], add_special_tokens=False)["input_ids"]))
            lengths[key] = lens
        prompt_lens = []
        for start in range(0, len(rows), 256):
            batch = [[{"role": "user", "content": r["problem"] + " " + INSTRUCTION}]
                     for r in rows[start:start+256]]
            prompt_lens.extend(map(len, tokenizer.apply_chat_template(batch, tokenize=True, add_generation_prompt=True)))
        print("Tokenized all fields and rendered student prompts", flush=True)

        eval_rows = []
        for path in sorted((ROOT / "data/eval").rglob("*.parquet")):
            for j, row in enumerate(pq.read_table(path).to_pylist()):
                extra = row.get("extra_info") or {}
                question = extra.get("problem") or extra.get("question")
                if not question:
                    question = row["prompt"][0]["content"].removesuffix(INSTRUCTION).strip()
                eval_rows.append({"dataset": path.parent.name, "index": j, "problem": question})
        exact_eval = []
        for er in eval_rows:
            for i in groups.get(canonical(er["problem"]), []):
                exact_eval.append({"dataset": er["dataset"], "eval_index": er["index"], "train_index": i})

        eval_grams = [trigrams(er["problem"]) for er in eval_rows]
        inverse = defaultdict(list)
        for j, grams in enumerate(eval_grams):
            for gram in grams:
                inverse[gram].append(j)
        near_eval = []
        for i, row in enumerate(rows):
            grams = trigrams(row["problem"])
            overlaps = Counter(j for g in grams for j in inverse.get(g, ()))
            for j, overlap in overlaps.items():
                score = overlap / (len(grams) + len(eval_grams[j]) - overlap)
                if score >= 0.5:
                    near_eval.append({"train_index": i, "dataset": eval_rows[j]["dataset"],
                                      "eval_index": eval_rows[j]["index"], "jaccard": score})
        print("Completed exact and lexical candidate benchmark overlap scan", flush=True)

        omni_summary = None
        if args.omni_jsonl:
            omni_bytes = args.omni_jsonl.read_bytes()
            omni = [json.loads(line) for line in omni_bytes.decode().splitlines() if line.strip()]
            joined = []
            for j, row in enumerate(omni):
                for i in groups.get(canonical(row["problem"]), []):
                    joined.append({"train_index": i, "omni_index": j,
                                   "difficulty": row.get("difficulty"), "domain": row.get("domain"),
                                   "source": row.get("source"),
                                   "answer_equiv_existing_grader": is_equiv(rows[i]["answer"], row.get("answer", ""))})
            (out / "omni-matches.json").write_text(json.dumps(joined, ensure_ascii=False, indent=2) + "\n")
            by_train = defaultdict(list)
            for match in joined:
                by_train[match["train_index"]].append(match)
            omni_summary = {"rows": len(omni), "sha256": hashlib.sha256(omni_bytes).hexdigest(),
                            "revision": args.omni_revision, "matched_train_rows": len({r["train_index"] for r in joined}),
                            "matched_omni_rows": len({r["omni_index"] for r in joined}),
                            "matched_pairs": len(joined),
                            "answer_equivalent_pairs_existing_grader": sum(r["answer_equiv_existing_grader"] for r in joined),
                            "unique_match_train_rows": sum(len(v)==1 for v in by_train.values()),
                            "unique_match_answer_equivalent_train_rows": sum(len(v)==1 and v[0]["answer_equiv_existing_grader"] for v in by_train.values()),
                            "multiple_match_train_rows": sum(len(v)>1 for v in by_train.values()),
                            "conflicting_difficulty_train_rows": sum(len({r["difficulty"] for r in v})>1 for v in by_train.values()),
                            "difficulty_counts_unambiguous_labels": dict(Counter(str(v[0]["difficulty"]) for v in by_train.values() if len({r["difficulty"] for r in v})==1))}

        sample_ids = random.Random(cfg["seed"]).sample(range(len(rows)), cfg["sample_count"])
        samples = {"random": [{"index": i, **rows[i]} for i in sample_ids],
                   "flags": {k: [{"index": i, **rows[i]} for i in v[:5]] for k, v in flags.items()},
                   "last_box_candidates": box_candidates[:30], "duplicate_groups": duplicates,
                   "conflicting_answer_groups": conflicts, "exact_eval_matches": exact_eval,
                   "near_eval_candidates": sorted(near_eval, key=lambda r: -r["jaccard"])}
        (out / "samples-and-candidates.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
        pq.write_table(pa.Table.from_pylist([
            {"raw_index": i, **{f"{key}_tokens": val[i] for key, val in lengths.items()},
             "student_chat_prompt_tokens": prompt_lens[i]} for i in range(len(rows))]), out / "lengths.parquet")
        summary = {"config": cfg, "swanlab_url": run.url, "rows": len(rows),
                   "schema_keys": sorted(rows[0]), "missing_or_empty": dict(counts),
                   "nonempty_solutions": sum(bool(r["solution"].strip()) for r in rows),
                   "raw_exact_duplicate_excess": sum(len(v)-1 for v in raw_groups.values()),
                   "normalized_duplicate_groups": len(duplicates),
                   "normalized_duplicate_excess": sum(len(v)-1 for v in duplicates),
                   "normalized_answer_conflict_groups_string_only": len(conflicts),
                   "answer_types_heuristic": dict(answer_types), "problem_flags_heuristic": {k:len(v) for k,v in flags.items()},
                   "reference_box_check_existing_grader": dict(box_check),
                   "tokens": {key: describe(vals) for key, vals in lengths.items()},
                   "nonempty_solution_tokens": describe([lengths["solution"][i] for i,r in enumerate(rows) if r["solution"].strip()]),
                   "student_chat_prompt_tokens": describe(prompt_lens),
                   "student_chat_prompt_over_2048": sum(v>2048 for v in prompt_lens),
                   "eval_counts": dict(Counter(r["dataset"] for r in eval_rows)),
                   "exact_eval_match_pairs": len(exact_eval),
                   "near_eval_candidate_pairs_ge_0_5": len(near_eval),
                   "omni_metadata_recovery": omni_summary, "wall_seconds": time.perf_counter()-started,
                   "limitations": ["lexical candidates are not confirmed duplicates", "no full semantic decontamination",
                                   "existing answer grader is incomplete; disagreement is not a proven wrong label",
                                   "no model generation or difficulty annotation performed"]}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        swanlab.log({"audit/rows":len(rows), "audit/nonempty_solutions": summary["nonempty_solutions"],
                     "audit/duplicate_excess": summary["normalized_duplicate_excess"],
                     "audit/exact_eval_matches":len(exact_eval), "audit/wall_seconds":summary["wall_seconds"],
                     "artifacts/summary":swanlab.Text(json.dumps(summary, ensure_ascii=False)),
                     "artifacts/log":swanlab.Text("Read-only CPU audit complete. Per-row samples, candidates and lengths kept locally: " + str(out))})
        swanlab.finish()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        swanlab.log({"artifacts/error":swanlab.Text(traceback.format_exc())})
        swanlab.finish(state="crashed")
        raise


if __name__ == "__main__":
    main()
