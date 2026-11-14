#!/usr/bin/env python3
"""Place full FSDP state and HF export on separate available filesystems."""
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipe/opd/run/utils"))
from run_config import build_env


def main():
    values = build_env()
    checkpoint_root = Path(values["MODEL_SAVE_DIR"]).resolve()
    export_root = Path(os.environ["HF_EXPORT_ROOT"]).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    ckpt_free = shutil.disk_usage(checkpoint_root).free / 1024**3
    export_free = shutil.disk_usage(export_root).free / 1024**3
    same_disk = checkpoint_root.stat().st_dev == export_root.stat().st_dev
    if (same_disk and ckpt_free < 38) or (not same_disk and (ckpt_free < 20 or export_free < 18)):
        raise SystemExit(
            f"Insufficient output space: checkpoint disk {ckpt_free:.1f} GiB, export disk {export_free:.1f} GiB. "
            "Reserve at least 20 GiB for FSDP state and 18 GiB for HF export (38 GiB if sharing a disk). "
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
