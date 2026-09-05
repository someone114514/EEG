#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/b_false_alarm_atlas"
cd "$ROOT"
NAMESPACE="${META_TTT_NAMESPACE:-cbramod-meta-ttt-v1-formal}"
export META_TTT_NAMESPACE="$NAMESPACE"
OUT="$ROOT/outputs/reports/$NAMESPACE/evaluation"
mkdir -p "$OUT"
PROGRESS="$OUT/queue_progress.json"
LOG="$OUT/queue.log"
if [[ -e "$OUT/queue.lock" ]]; then echo "evaluation lock exists; refusing duplicate" >&2; exit 2; fi
printf '{"pid":%s,"namespace":"%s","started_utc":"%s"}
' "$$" "$NAMESPACE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/queue.lock"
trap 'mv "$OUT/queue.lock" "$OUT/queue.lock.released"' EXIT
units=()
for fold in 0 1 2 3 4; do for seed in 17 42 3407; do units+=("$fold:$seed"); done; done
python_bin="$ROOT/.venv/bin/python"
progress() {
  "$python_bin" - "$PROGRESS" "$1" "$2" "$3" "${#units[@]}" <<'PY'
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
progress running 0 none
echo "[$(date -u +%FT%TZ)] start namespace=$NAMESPACE" >> "$LOG"
completed=0
for unit in "${units[@]}"; do
  fold="${unit%%:*}"; seed="${unit##*:}"
  checkpoint="$ROOT/outputs/reports/$NAMESPACE/runs/fold${fold}_seed${seed}/checkpoint.pt"
  run_dir="$OUT/fold${fold}_seed${seed}"
  if [[ ! -f "$checkpoint" ]]; then echo "missing checkpoint $checkpoint" | tee -a "$LOG" >&2; exit 4; fi
  if [[ -e "$run_dir/manifest.json" ]]; then echo "existing evaluation; refusing duplicate $run_dir" | tee -a "$LOG" >&2; exit 3; fi
  progress running "$completed" "fold${fold}_seed${seed}"
  echo "[$(date -u +%FT%TZ)] begin fold=$fold seed=$seed" | tee -a "$LOG"
  TTT_METHOD=meta JOINT_TTT_NAMESPACE="$NAMESPACE" PYTHONPATH=src PYTHONUNBUFFERED=1 "$python_bin" scripts/214_evaluate_joint_ttt.py --method meta --fold "$fold" --seed "$seed" --device "${META_TTT_EVAL_DEVICE:-cuda}" --update-after-score 2>&1 | tee "$OUT/fold${fold}_seed${seed}.log"
  completed=$((completed+1)); progress running "$completed" none
  echo "[$(date -u +%FT%TZ)] complete fold=$fold seed=$seed" | tee -a "$LOG"
done
progress complete "$completed" none
echo "[$(date -u +%FT%TZ)] queue complete" | tee -a "$LOG"
