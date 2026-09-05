"""Summarize completed Band-TTT v2 fold-0/1 jobs against frozen baselines."""
from __future__ import annotations
import os

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT = ROOT / "outputs" / "reports" / "band-ttt-v2-fold01"
SOURCE = ROOT / "outputs" / "reports" / "meta-ttt-chbmit-5fold-v1"


def pooled(metrics: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(int(x["true_positive_events"]) for x in metrics)
    fa = sum(int(x["false_alarm_events"]) for x in metrics)
    truth = sum(int(x["truth_events"]) for x in metrics)
    alarm_s = sum(float(x["false_alarm_time_seconds"]) for x in metrics)
    hours = sum(float(x["total_monitoring_hours"]) for x in metrics)
    nonseizure = sum(float(x["nonseizure_hours"]) for x in metrics)
    delay_count = sum(int(x.get("detection_delay_count", 0)) for x in metrics)
    delay_sum = sum(float(x.get("detection_delay_sum_s", 0.0)) for x in metrics)
    return {
        "true_positive_events": tp, "false_alarm_events": fa, "truth_events": truth,
        "false_alarm_time_seconds": alarm_s, "total_monitoring_hours": hours, "nonseizure_hours": nonseizure,
        "event_sensitivity": tp / truth if truth else float("nan"),
        "fa_per_24h": fa * 24.0 / nonseizure if nonseizure else float("nan"),
        "false_alarm_time_min_per_24h": alarm_s / 60.0 * 24.0 / hours if hours else float("nan"),
        "detection_delay_mean_s": delay_sum / delay_count if delay_count else float("nan"),
        "detection_delay_count": delay_count,
    }


def main() -> None:
    manifest = json.loads((OUT / "frozen_manifest.json").read_text())
    frozen_fold = []
    frozen_elapsed = 0.0
    for fold in (0, 1):
        path = SOURCE / "evaluation" / "meta_band_frozen" / f"fold{fold}_seed3407" / "test_completed.json"
        item = json.loads(path.read_text())
        frozen_fold.append(item["selected_event_operating_point"])
        frozen_elapsed += float(item["elapsed_s"])
    frozen = pooled(frozen_fold)
    rows = [{"condition": "meta_band_frozen", "comparison_mode": "existing_validation_locked", **frozen, "elapsed_s": frozen_elapsed, "gpu_peak_mib": 0.0}]
    patients = []
    missing = []
    for config in manifest["configurations"]:
        own = []
        matched = []
        elapsed = peak = 0.0
        for fold in (0, 1):
            result_dir = OUT / "evaluation" / config["config_id"] / f"fold{fold}_seed3407"
            completed = result_dir / "test_completed.json"
            if not completed.is_file():
                missing.append(str(completed))
                continue
            item = json.loads(completed.read_text())
            own.append(item["selected_event_operating_point"])
            matched.append(item["matched_frozen_threshold_metrics"])
            elapsed += float(item["elapsed_s"])
            peak = max(peak, float(item["gpu_peak_mib"]))
            for mode, filename in (("own_validation_locked", "test_patient_waterfall.csv"), ("matched_frozen_threshold", "test_patient_waterfall_matched_frozen_threshold.csv")):
                frame = pd.read_csv(result_dir / filename)
                frame.insert(0, "fold", fold); frame.insert(0, "comparison_mode", mode); frame.insert(0, "condition", config["config_id"])
                patients.append(frame)
        if len(own) != 2:
            continue
        for mode, metric in (("own_validation_locked", pooled(own)), ("matched_frozen_threshold", pooled(matched))):
            rows.append({
                "condition": config["config_id"], "comparison_mode": mode, **metric,
                "delta_event_sensitivity_vs_frozen": metric["event_sensitivity"] - frozen["event_sensitivity"],
                "delta_fa_per_24h_vs_frozen": metric["fa_per_24h"] - frozen["fa_per_24h"],
                "delta_false_alarm_time_min_per_24h_vs_frozen": metric["false_alarm_time_min_per_24h"] - frozen["false_alarm_time_min_per_24h"],
                "delta_detection_delay_mean_s_vs_frozen": metric["detection_delay_mean_s"] - frozen["detection_delay_mean_s"],
                "elapsed_s": elapsed, "runtime_overhead_s_vs_frozen": elapsed - frozen_elapsed, "gpu_peak_mib": peak,
            })
    summary_dir = OUT / "summary"; summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_dir / "condition_metrics.csv", index=False)
    if patients:
        pd.concat(patients, ignore_index=True).to_csv(summary_dir / "patient_metrics.csv", index=False)
    status = {"status": "complete" if not missing else "partial", "completed_configurations": sum(row["comparison_mode"] == "own_validation_locked" for row in rows), "expected_configurations": 16, "missing_test_outputs": missing}
    (summary_dir / "manifest.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
