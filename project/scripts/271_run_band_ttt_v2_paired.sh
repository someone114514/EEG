#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
exec /root/b_false_alarm_atlas/.venv/bin/python \
  /root/b_false_alarm_atlas/scripts/270_queue_band_ttt_v2_paired.py \
  --parallel-workers 4 \
  --gpu-slots 4 \
  --window-batch-size 128 \
  --stream-batch-size 128 \
  --workers 8 \
  >> /root/b_false_alarm_atlas/outputs/reports/band-ttt-v2-fold01/paired_queue.log 2>&1
