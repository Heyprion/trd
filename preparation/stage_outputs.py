#!/usr/bin/env python3
"""Place full FSDP state and HF export on separate available filesystems."""
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipe/opd/run/utils"))
from run_config import build_env


def minimum_space(values):
    if os.environ.get("OUTPUT_SPACE_PROFILE") != "balanced-single-save":
        return 20, 18
    # A second retained experiment fits on the current disks. This narrower
    # reserve is based on measured 8.971/8.478 GiB checkpoint/export sizes;
    # it applies only to the same single-save BF16 LoRA run, never generically.
    required = {"MODEL_STORAGE_DTYPE": "bf16", "USE_LORA": "true", "LORA_RANK": "64",
                "LORA_ALPHA": "128", "TRAIN_EPOCHS_PER_ROUND": "1", "TRAIN_BATCH_SIZE": "2",
                "GRADIENT_ACCUMULATION_STEPS": "32", "NGPUS_PER_NODE": "2", "NNODES": "1", "SP_SIZE": "1"}
    if any(values.get(key) != value for key, value in required.items()) or int(values["SAVE_STEPS"]) < 78:
        raise ValueError("balanced-single-save storage requires the measured BF16/r64/9984/78-update profile")
    if values["MODEL_NAME"] != "Qwen3-4B-Instruct-2507":
        raise ValueError("balanced-single-save storage is sized only for the original 4B student")
    return 12, 11


def main():
    values = build_env()
    checkpoint_root = Path(values["MODEL_SAVE_DIR"]).resolve()
    export_root = Path(os.environ["HF_EXPORT_ROOT"]).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    ckpt_free = shutil.disk_usage(checkpoint_root).free / 1024**3
    export_free = shutil.disk_usage(export_root).free / 1024**3
    same_disk = checkpoint_root.stat().st_dev == export_root.stat().st_dev
    ckpt_need, export_need = minimum_space(values)
    if (same_disk and ckpt_free < ckpt_need + export_need) or (
        not same_disk and (ckpt_free < ckpt_need or export_free < export_need)
    ):
        raise SystemExit(
            f"Insufficient output space: checkpoint disk {ckpt_free:.1f} GiB, export disk {export_free:.1f} GiB. "
            f"Reserve at least {ckpt_need} GiB for FSDP state and {export_need} GiB for HF export "
            f"({ckpt_need + export_need} GiB if sharing a disk). "
            "Set MODEL_SAVE_DIR and HF_EXPORT_ROOT to larger storage before training."
        )
    epoch_dir = Path(values["EPOCH_MODEL_SAVE_DIR"]).resolve()
    relative = epoch_dir.relative_to(checkpoint_root)
    export_dir = export_root / relative / "hf_merged"
    export_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    link = epoch_dir / "hf_merged"
    if link.is_symlink():
        if link.resolve() != export_dir:
            raise SystemExit(f"Existing export symlink points elsewhere: {link}")
    elif link.exists():
        raise SystemExit(f"Existing export directory requires its original storage configuration: {link}")
    else:
        link.symlink_to(export_dir, target_is_directory=True)
    print(f"FSDP checkpoints: {epoch_dir} ({ckpt_free:.1f} GiB free)")
    print(f"Merged HF model:  {link} -> {export_dir} ({export_free:.1f} GiB free)")


if __name__ == "__main__":
    main()
