#!/usr/bin/env python3
"""Download only the two models and six source datasets needed for math OPD.

Record immutable Hugging Face revisions and sizes. Files live directly in local
directories (no second copy of model weights in the HF blob cache).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from parallel_download import download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B"]
DATASETS = [
    "agentica-org/DeepScaleR-Preview-Dataset", "math-ai/aime24", "math-ai/aime25",
    "PraMamba/HMMT-202502", "ByteDance-Seed/BeyondAIME", "meituan-longcat/AMO-Bench",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["models", "datasets", "all"], default="all")
    parser.add_argument("--parallel-http", action="store_true", help="Use verified parallel HTTP ranges for model weights")
    parser.add_argument("--manifest", type=Path, default=ROOT / "preparation/assets.json")
    parser.add_argument("--weight-endpoint", help="Optional mirror for verified weight ranges; metadata still uses HF_ENDPOINT")
    parser.add_argument("--weight-url-template", help="Optional alternate weight URL with {repo_id}, {revision}, {filename}; HF SHA256 is still enforced")
    args = parser.parse_args()
    manifest_path = args.manifest
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    api = HfApi()
    for kind, names in [("model", MODELS), ("dataset", DATASETS)]:
        if args.only != "all" and args.only != kind + "s":
            continue
        for name in names:
            key = f"{kind}:{name}"
            revision = manifest.get(key, {}).get("revision") or api.repo_info(name, repo_type=kind).sha
            destination = ROOT / (f"models/{name.split('/')[-1]}" if kind == "model" else f"data/raw/{name}")
            manifest[key] = {"repo_id": name, "repo_type": kind, "revision": revision,
                             "path": str(destination), "complete": False}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Downloading {name}@{revision} -> {destination}", flush=True)
            snapshot_download(name, repo_type=kind, revision=revision, local_dir=destination,
                              allow_patterns=["*.safetensors", "*.json", "*.jsonl", "*.parquet", "*.csv",
                                              "*.txt", "*.model", "*.tiktoken", "*.jinja", "README.md"],
                              ignore_patterns=["*.safetensors"] if kind == "model" and args.parallel_http else None,
                              max_workers=4)
            if kind == "model" and args.parallel_http:
                info = api.model_info(name, revision=revision, files_metadata=True)
                manifest[key]["weight_sha256"] = {item.rfilename: item.lfs.sha256 for item in info.siblings
                                                   if item.rfilename.endswith(".safetensors")}
                if args.weight_url_template:
                    manifest[key]["weight_url_template"] = args.weight_url_template
                for item in info.siblings:
                    if item.rfilename.endswith(".safetensors"):
                        url = f"{(args.weight_endpoint or api.endpoint).rstrip('/')}/{name}/resolve/{revision}/{item.rfilename}"
                        if args.weight_url_template:
                            url = args.weight_url_template.format(repo_id=name, revision=revision, filename=item.rfilename)
                        download(url, destination / item.rfilename, item.size, item.lfs.sha256)
            files = {str(p.relative_to(destination)): p.stat().st_size for p in destination.rglob("*")
                     if p.is_file() and ".cache" not in p.relative_to(destination).parts}
            manifest[key].update(complete=True, files=files)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Ready: {name} ({sum(files.values()) / 1024**3:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
