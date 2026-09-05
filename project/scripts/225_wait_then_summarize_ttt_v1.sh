#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/b_false_alarm_atlas"
cd "$ROOT"
paths=(
  "$ROOT/outputs/reports/cbramod-joint-ttt-v1-formal/evaluation/queue_progress.json"
  "$ROOT/outputs/reports/cbramod-meta-ttt-v1-formal/evaluation/queue_progress.json"
  "$ROOT/outputs/reports/cbramod-label-prior-tta-v1-formal/evaluation/queue_progress.json"
)
while true; do
  all_complete=1
  for path in "${paths[@]}"; do
    status="$("$ROOT/.venv/bin/python" - "$path" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("status","missing"))
except Exception: print("missing")
PY
)"
    if [[ "$status" == failed || "$status" == error ]]; then echo "evaluation status=$status for $path" >&2; exit 4; fi
    if [[ "$status" != complete ]]; then all_complete=0; fi
  done
  if [[ "$all_complete" == 1 ]]; then break; fi
  sleep 60
done
PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/226_audit_ttt_results.py
PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/224_summarize_ttt_results.py
PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/227_analyze_ttt_results.py
PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/228_write_ttt_report.py
