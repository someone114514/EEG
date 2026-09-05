#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BFA_ROOT:-}" ]]; then
  BFA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  export BFA_ROOT
fi

export NEUROTTT_CODE_ROOT="${NEUROTTT_CODE_ROOT:-$BFA_ROOT/external/NeuroTTT_CBraMod}"
export BFA_MANIFEST_ROOT="${BFA_MANIFEST_ROOT:-$BFA_ROOT/manifests}"
export BFA_WINDOWS="${BFA_WINDOWS:-$BFA_MANIFEST_ROOT/windows.parquet}"
export BFA_FOLD_ROOT="${BFA_FOLD_ROOT:-$BFA_MANIFEST_ROOT/groupkfold_cv_v1}"
export BFA_RECORDINGS="${BFA_RECORDINGS:-$BFA_MANIFEST_ROOT/recordings.parquet}"
export BFA_SEIZURES="${BFA_SEIZURES:-$BFA_MANIFEST_ROOT/seizures.parquet}"
export BFA_CACHE_ROOT="${BFA_CACHE_ROOT:-$BFA_ROOT/data/bfa_cache_v3_official_noclip/cbramod}"
export BFA_RAW_ROOT="${BFA_RAW_ROOT:-$BFA_ROOT/data/chbmit-1.0.0}"
export META_TTT_PYTHON="${META_TTT_PYTHON:-$BFA_ROOT/.venv/bin/python}"

export PYTHONPATH="$BFA_ROOT/project/src:$NEUROTTT_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="false"

echo "BFA_ROOT=$BFA_ROOT"
echo "BFA_CACHE_ROOT=$BFA_CACHE_ROOT"
echo "NEUROTTT_CODE_ROOT=$NEUROTTT_CODE_ROOT"
echo "META_TTT_PYTHON=$META_TTT_PYTHON"
