#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/b_false_alarm_atlas"
cd "$ROOT"
META_OUT="$ROOT/outputs/reports/cbramod-meta-ttt-v1-formal"
while true; do
  status="$("$ROOT/.venv/bin/python" - "$META_OUT/queue_progress.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("status","missing"))
except Exception: print("missing")
PY
)"
  case "$status" in
    complete) break ;;
    failed|error) echo "meta queue status=$status; refusing evaluation launch" >&2; exit 4 ;;
    *) sleep 60 ;;
  esac
done
PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/110_preflight_no_duplicate_overlap.py --namespace cbramod-meta-ttt-v1-formal-evaluation --write-report
exec env META_TTT_NAMESPACE=cbramod-meta-ttt-v1-formal META_TTT_EVAL_DEVICE=cuda bash scripts/222_queue_meta_eval_v1.sh
