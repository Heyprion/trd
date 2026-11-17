"""Pilot-only seeded entrypoint; accepts the existing KL training arguments."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from transformers import set_seed

    seed = int(os.environ["THINKING_ABLATION_SEED"])
    set_seed(seed)
    print(f"[thinking-ablation] Python/NumPy/Torch seed={seed}; deterministic kernels not enforced", flush=True)
    from recipe.opd.run_training import main as train

    train()


if __name__ == "__main__":
    main()
