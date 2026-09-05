#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$BUNDLE_ROOT/environment/activate.sh" >/dev/null

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
"$META_TTT_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import torch

paths = {
    "bundle": Path(os.environ["BFA_ROOT"]),
    "windows": Path(os.environ["BFA_WINDOWS"]),
    "fold_root": Path(os.environ["BFA_FOLD_ROOT"]),
    "recordings": Path(os.environ["BFA_RECORDINGS"]),
    "seizures": Path(os.environ["BFA_SEIZURES"]),
    "cache_root": Path(os.environ["BFA_CACHE_ROOT"]),
    "pretrained": Path(os.environ["NEUROTTT_CODE_ROOT"]) / "pretrained_weights" / "pretrained_weights.pth",
}
print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cuda": torch.cuda.is_available(),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "paths": {key: {"path": str(value), "exists": value.exists()} for key, value in paths.items()}}, indent=2))
missing = [str(value) for value in paths.values() if not value.exists()]
if missing:
    raise SystemExit("missing required path(s): " + "; ".join(missing))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY
echo "preflight passed"
