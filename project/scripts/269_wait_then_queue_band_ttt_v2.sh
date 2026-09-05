#!/usr/bin/env bash
set -euo pipefail
# Compatibility entrypoint: the protocol was amended to paired val->test.
exec /usr/bin/bash /root/b_false_alarm_atlas/scripts/271_run_band_ttt_v2_paired.sh
