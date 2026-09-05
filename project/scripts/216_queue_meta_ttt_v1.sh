#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/b_false_alarm_atlas"
cd "$ROOT"
NAMESPACE="${META_TTT_NAMESPACE:-cbramod-meta-ttt-v1-formal}"
export META_TTT_NAMESPACE="$NAMESPACE"
OUT="$ROOT/outputs/reports/$NAMESPACE"
mkdir -p "$OUT"
QUEUE_LOG="$OUT/queue.log"
PROGRESS="$OUT/queue_progress.json"
if [[ -e "$OUT/queue.lock" ]]; then
  echo "queue lock exists; refusing duplicate launch: $OUT/queue.lock" >&2
  exit 2
fi
printf '{"pid":%s,"namespace":"%s","started_utc":"%s"}
' "$$" "$NAMESPACE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/queue.lock"
trap 'mv "$OUT/queue.lock" "$OUT/queue.lock.released"' EXIT

units=()
for fold in 0 1 2 3 4; do
  for seed in 17 42 3407; do units+=("$fold:$seed"); done
done
python_bin="$ROOT/.venv/bin/python"
atomic_progress() {
  local status="$1" completed="$2" current="$3"
  "$python_bin" - "$PROGRESS" "$status" "$completed" "$current" "${#units[@]}" <<'PY'
import json, os, sys
path, status, completed, current, total = sys.argv[1:]
payload = {
    "status": status,
    "completed": int(completed),
    "current": None if current == "none" else current,
    "total": int(total),
}
tmp = path + ".tmp"
with open(tmp, "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(tmp, path)
PY
}
atomic_progress running 0 none
echo "[$(date -u +%FT%TZ)] start namespace=$NAMESPACE" >> "$QUEUE_LOG"
completed=0
for unit in "${units[@]}"; do
  fold="${unit%%:*}"; seed="${unit##*:}"
  run_dir="$OUT/runs/fold${fold}_seed${seed}"
  if [[ -e "$run_dir/manifest.json" ]]; then
    echo "[$(date -u +%FT%TZ)] refusing existing completed run $run_dir" | tee -a "$QUEUE_LOG" >&2
    exit 3
  fi
  atomic_progress running "$completed" "fold${fold}_seed${seed}"
  log="$OUT/formal_fold${fold}_seed${seed}.log"
  echo "[$(date -u +%FT%TZ)] begin fold=$fold seed=$seed" | tee -a "$QUEUE_LOG"
  PYTHONPATH=src PYTHONUNBUFFERED=1 "$python_bin" scripts/212_meta_ttt_train.py train     --fold "$fold" --seed "$seed" --updates "${META_TTT_UPDATES:-5000}"     --batch-size "${META_TTT_BATCH_SIZE:-4}" --micro-contexts "${META_TTT_MICRO_CONTEXTS:-1}"     --inner-lr "${META_TTT_INNER_LR:-1e-5}" --outer-lr "${META_TTT_OUTER_LR:-1e-4}"     --device "${META_TTT_DEVICE:-cuda}" 2>&1 | tee "$log"
  completed=$((completed + 1))
  atomic_progress running "$completed" none
  echo "[$(date -u +%FT%TZ)] complete fold=$fold seed=$seed" | tee -a "$QUEUE_LOG"
done
atomic_progress complete "$completed" none
echo "[$(date -u +%FT%TZ)] queue complete" | tee -a "$QUEUE_LOG"
