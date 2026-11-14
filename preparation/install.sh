#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -c 'import torch; assert torch.__version__.startswith("2.8.0"), torch.__version__; assert torch.version.cuda == "12.8"; assert torch._C._GLIBCXX_USE_CXX11_ABI'
if [[ ! -x .venv/bin/python ]]; then python -m venv --system-site-packages .venv; fi
.venv/bin/python -m pip install --index-url "${OPD_PIP_INDEX_URL:-https://pypi.org/simple}" --no-cache-dir 'pip==25.2'
shopt -s nullglob
LOCAL_WHEELS=(preparation/wheels/{vllm-,xformers-,torchaudio-,ray-,cupy_cuda12x-,llvmlite-,pyarrow-,scipy-,opencv_python_headless-}*.whl)
if (( ${#LOCAL_WHEELS[@]} )); then
    .venv/bin/python -m pip install --no-deps "${LOCAL_WHEELS[@]}"
fi
.venv/bin/python -m pip install --index-url "${OPD_PIP_INDEX_URL:-https://pypi.org/simple}" --no-cache-dir -r preparation/requirements-math-opd.txt -c preparation/requirements-resolved.txt
mkdir -p preparation/wheels
WHEEL='flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
if [[ ! -f "preparation/wheels/$WHEEL" ]]; then
    curl --fail --location --retry 3 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL/+/%2B}" -o "preparation/wheels/$WHEEL"
fi
.venv/bin/python -m pip install --no-cache-dir --no-deps "preparation/wheels/$WHEEL"
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
# torch is inherited, so its host torchrun entrypoint would select host Python.
cat > .venv/bin/torchrun <<'SH'
#!/usr/bin/env bash
exec "$(dirname "$(readlink -f "$0")")/python" -m torch.distributed.run "$@"
SH
chmod +x .venv/bin/torchrun
.venv/bin/python -m pip freeze > preparation/installed-freeze.txt
