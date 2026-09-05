"""Audit and threshold-sweep the completed same-patient CBraMod experiment."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = os.environ.get("CBRAMOD_AUDIT_NAMESPACE", "cbramod-chb-same-patient-online-adaptation-formal-v1")
OUT = ROOT / "outputs/reports" / NAMESPACE
SCRIPT = ROOT / "scripts/202_cbramod_same_patient_adaptation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cbramod_adaptation", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_module()
    windows, recordings, seizures = module.load_tables()
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for manifest_path in sorted((OUT / "runs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        patient = str(manifest["patient_id"])
        method = str(manifest["method"])
        probabilities = pd.read_parquet(OUT / "runs" / manifest_path.parent.name / "probabilities.parquet")
        baseline = pd.read_parquet(OUT / "runs" / f"frozen__{patient}__seed17" / "probabilities.parquet")
        if len(probabilities) != len(baseline):
            raise RuntimeError(f"row count mismatch for {patient}/{method}")
        if not probabilities[["patient", "recording", "start_s", "end_s"]].reset_index(drop=True).equals(
            baseline[["patient", "recording", "start_s", "end_s"]].reset_index(drop=True)
        ):
            raise RuntimeError(f"time grid mismatch for {patient}/{method}")
        diff = probabilities.probability.to_numpy(float) - baseline.probability.to_numpy(float)
        rows.append(
            {
                "patient_id": patient,
                "method": method,
                "scored_windows": len(probabilities),
                "mean_abs_probability_change": float(np.mean(np.abs(diff))),
                "max_abs_probability_change": float(np.max(np.abs(diff))),
                "fraction_probability_changed": float(np.mean(np.abs(diff) > 1e-6)),
                "head_unchanged": bool(manifest["head_unchanged"]),
                "backbone_changed": bool(manifest["initial_backbone_sha256"] != manifest["final_backbone_sha256"]),
                "projection_changed": bool(manifest["initial_projection_sha256"] != manifest["final_projection_sha256"]),
                "scored_before_update": all(bool(item["scored_before_update"]) for item in json.loads((manifest_path.parent / "history.json").read_text())),
                "updates": int(manifest["updates"]),
                "event_sensitivity_at_0.01": float(manifest["metrics"]["event_sensitivity"]),
                "fa_per_24h_at_0.01": float(manifest["metrics"]["fa_per_24h"]),
            }
        )
        for threshold in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            module.THRESHOLD = threshold
            metrics = module.score_events(probabilities, recordings, seizures, patient)
            threshold_rows.append({"patient_id": patient, "method": method, "threshold": threshold, **metrics})
    pd.DataFrame(rows).sort_values(["patient_id", "method"]).to_csv(OUT / "adaptation_audit.csv", index=False)
    sweep = pd.DataFrame(threshold_rows).sort_values(["patient_id", "threshold", "method"])
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
    # Restore protocol value in the imported module (the saved runs remain at
    # the frozen 0.01 operating point).
    module.THRESHOLD = 0.01
    summary = {
        "status": "PASS",
        "namespace": NAMESPACE,
        "run_count": len(rows),
        "all_heads_unchanged": all(bool(row["head_unchanged"]) for row in rows),
        "all_scored_before_update": all(bool(row["scored_before_update"]) for row in rows),
        "thresholds_are_descriptive_only": True,
        "frozen_operating_threshold": 0.01,
        "outputs": ["adaptation_audit.csv", "threshold_sweep.csv"],
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
