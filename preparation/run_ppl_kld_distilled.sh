#!/usr/bin/env bash
# Pinned to the completed 9,984-question run and its existing 512-question baseline.
set -euo pipefail
PPL_KLD_DISTILLED_ENTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPL_KLD_TRAIN_SUMMARY="$PPL_KLD_DISTILLED_ENTRY/logs/math-runs/20261115-185850-279780-53607/train/summary.json"
PPL_KLD_BASELINE_SUMMARY="$PPL_KLD_DISTILLED_ENTRY/logs/ppl-kld/baseline-20261115-101823-758028-27412/summary.json"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$PPL_KLD_DISTILLED_ENTRY/../.venv/bin/python" ]]; then
        PYTHON_BIN="$PPL_KLD_DISTILLED_ENTRY/../.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi
PPL_KLD_DISTILLED_MODEL="$("$PYTHON_BIN" - "$PPL_KLD_TRAIN_SUMMARY" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
if summary.get("stage") != "train" or summary.get("status") != "completed":
    raise SystemExit("The pinned training run is not completed")
model = Path(summary["result"]["model_path"]).resolve()
marker = model / "lora_merge.json"
if not marker.is_file() or json.loads(marker.read_text()).get("merged") is not True:
    raise SystemExit("The pinned model has no completed LoRA merge")
print(model)
PY
)"
exec bash "$PPL_KLD_DISTILLED_ENTRY/run_ppl_kld.sh" \
    --label distilled \
    --student-model "$PPL_KLD_DISTILLED_MODEL" \
    --compare-to "$PPL_KLD_BASELINE_SUMMARY" \
    "$@"
