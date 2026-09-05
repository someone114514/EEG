#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/b_false_alarm_atlas"
cd "$ROOT"
NAMESPACE="${TUSZ_JOINT_TTT_NAMESPACE:-cbramod-joint-ttt-tusz-one-round-v1}"
OUT="$ROOT/outputs/reports/$NAMESPACE"
JOINT_EVAL="$ROOT/outputs/reports/cbramod-joint-ttt-v1-formal/evaluation/queue_progress.json"
META_QUEUE="$ROOT/outputs/reports/cbramod-meta-ttt-v1-formal/queue_progress.json"
META_EVAL="$ROOT/outputs/reports/cbramod-meta-ttt-v1-formal/evaluation/queue_progress.json"
LABEL_EVAL="$ROOT/outputs/reports/cbramod-label-prior-tta-v1-formal/evaluation/queue_progress.json"
PYTHON="$ROOT/.venv/bin/python"

status_of() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("status", "missing"))
except Exception:
    print("missing")
PY
}

while true; do
  status="$(status_of "$JOINT_EVAL")"
  case "$status" in
    complete) break ;;
    failed|error)
      echo "CHB Joint evaluation status=$status; refusing TUSZ launch" >&2
      exit 4
      ;;
    *) sleep 60 ;;
  esac
done

# The existing CHB continuation wrapper starts Meta-TTT immediately after
# Joint evaluation. Wait for that GPU work before launching the TUSZ run.
while pgrep -f 'scripts/(216_queue_meta_ttt_v1|223_wait_then_eval_meta_v1|225_wait_then_summarize_ttt_v1)' >/dev/null 2>&1; do
  for path in "$META_QUEUE" "$META_EVAL" "$LABEL_EVAL"; do
    if [[ -f "$path" ]]; then
      status="$(status_of "$path")"
      case "$status" in
        failed|error)
          echo "existing CHB TTT stage failed: $path status=$status" >&2
          exit 5
          ;;
      esac
    fi
  done
  sleep 60
done

PYTHONPATH=src "$PYTHON" scripts/110_preflight_no_duplicate_overlap.py --namespace "$NAMESPACE" --write-report
if [[ -e "$OUT/manifest.json" || -e "$OUT/queue.lock" ]]; then
  echo "existing TUSZ one-round output or lock; refusing duplicate: $OUT" >&2
  exit 3
fi
mkdir -p "$OUT"
printf '{"pid":%s,"namespace":"%s","started_utc":"%s"}\n' "$$" "$NAMESPACE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/queue.lock"
trap 'mv "$OUT/queue.lock" "$OUT/queue.lock.released"' EXIT
PYTHONPATH=src PYTHONUNBUFFERED=1 TUSZ_JOINT_TTT_NAMESPACE="$NAMESPACE" "$PYTHON" scripts/230_run_tusz_joint_ttt_one_round.py --device "${TUSZ_JOINT_TTT_DEVICE:-cuda}" 2>&1 | tee "$OUT/queue.log"

