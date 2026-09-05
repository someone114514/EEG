#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BUNDLE_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv .venv
"$BUNDLE_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$BUNDLE_ROOT/.venv/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128
"$BUNDLE_ROOT/.venv/bin/python" -m pip install -r environment/requirements_meta_ttt.txt

"$BUNDLE_ROOT/.venv/bin/python" - <<'PY'
import json
import sys
import torch

payload = {
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; install a compatible NVIDIA driver before running MetaTTT")
PY
