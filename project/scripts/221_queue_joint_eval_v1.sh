#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/b_false_alarm_atlas"
cd "$ROOT"
NAMESPACE="${JOINT_TTT_NAMESPACE:-cbramod-joint-ttt-v1-formal}"
export JOINT_TTT_NAMESPACE="$NAMESPACE"
OUT="$ROOT/outputs/reports/$NAMESPACE/evaluation"
mkdir -p "$OUT"
PROGRESS="$OUT/queue_progress.json"
LOG="$OUT/queue.log"
if [[ -e "$OUT/queue.lock" ]]; then echo "evaluation lock exists; refusing duplicate" >&2; exit 2; fi
printf '{"pid":%s,"namespace":"%s","started_utc":"%s"}
' "$$" "$NAMESPACE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/queue.lock"
trap 'mv "$OUT/queue.lock" "$OUT/queue.lock.released"' EXIT
python_bin="$ROOT/.venv/bin/python"
echo "[$(date -u +%FT%TZ)] start namespace=$NAMESPACE" >> "$LOG"
JOINT_TTT_NAMESPACE="$NAMESPACE" PYTHONPATH=src PYTHONUNBUFFERED=1 \
  "$python_bin" scripts/233_parallel_joint_ttt_evaluation.py
