#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/b_false_alarm_atlas"
cd "$ROOT"
JOINT_OUT="$ROOT/outputs/reports/cbramod-joint-ttt-v1-formal"
META_OUT="$ROOT/outputs/reports/cbramod-meta-ttt-v1-formal"
while true; do
  status="$("$ROOT/.venv/bin/python" - "$JOINT_OUT/queue_progress.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("status","missing"))
except Exception: print("missing")
PY
)"
  case "$status" in
    complete) break ;;
    failed|error) echo "joint queue status=$status; refusing meta launch" >&2; exit 4 ;;
    *) sleep 60 ;;
  esac
done
PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/110_preflight_no_duplicate_overlap.py --namespace cbramod-meta-ttt-v1-formal --write-report
echo "joint queue complete; auditing and running one-pass joint evaluation before meta training"
PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/110_preflight_no_duplicate_overlap.py --namespace cbramod-joint-ttt-v1-formal-evaluation --write-report
JOINT_TTT_NAMESPACE=cbramod-joint-ttt-v1-formal JOINT_TTT_EVAL_DEVICE=cuda bash scripts/221_queue_joint_eval_v1.sh
exec env META_TTT_NAMESPACE=cbramod-meta-ttt-v1-formal META_TTT_DEVICE=cuda META_TTT_UPDATES=5000 bash scripts/216_queue_meta_ttt_v1.sh
