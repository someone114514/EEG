#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$BUNDLE_ROOT/environment/activate.sh" >/dev/null

sample="$BFA_CACHE_ROOT/chb01/chb01_01.npy"
[[ -f "$sample" ]] || { echo "missing cache sample: $sample" >&2; exit 2; }
"$META_TTT_PYTHON" - "$sample" <<'PY'
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
view = np.load(path, mmap_mode="r", allow_pickle=False)
print({"path": str(path), "dtype": str(view.dtype), "shape": list(view.shape),
       "finite_probe": bool(np.isfinite(view[:, : min(view.shape[-1], 2000)]).all())})
if view.ndim != 2 or view.shape[0] != 16:
    raise SystemExit(f"unexpected cache shape: {view.shape}")
PY
echo "external cache probe passed"
