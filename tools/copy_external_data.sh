#!/usr/bin/env bash
set -euo pipefail

# Copy data into a target machine. This script is intentionally explicit:
# it never deletes the destination and it does not infer paths from filenames.
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$BUNDLE_ROOT/environment/activate.sh" >/dev/null

SOURCE_CACHE="${1:-}"
SOURCE_RAW="${2:-}"
if [[ -z "$SOURCE_CACHE" ]]; then
  echo "usage: bash tools/copy_external_data.sh /source/cache/cbramod [/source/chbmit-1.0.0]" >&2
  exit 2
fi
[[ -d "$SOURCE_CACHE" ]] || { echo "source cache not found: $SOURCE_CACHE" >&2; exit 2; }
mkdir -p "$BFA_CACHE_ROOT"
rsync -a --info=progress2 "$SOURCE_CACHE/" "$BFA_CACHE_ROOT/"

if [[ -n "$SOURCE_RAW" ]]; then
  [[ -d "$SOURCE_RAW" ]] || { echo "source raw data not found: $SOURCE_RAW" >&2; exit 2; }
  mkdir -p "$BFA_RAW_ROOT"
  rsync -a --info=progress2 "$SOURCE_RAW/" "$BFA_RAW_ROOT/"
fi

bash "$BUNDLE_ROOT/tools/verify_external_data.sh"
