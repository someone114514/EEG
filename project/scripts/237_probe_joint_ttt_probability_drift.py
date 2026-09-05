"""Read-only probe of block-wise probability drift in completed test runs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/root/b_false_alarm_atlas")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    evaluator = load(ROOT / "scripts/214_evaluate_joint_ttt.py", "joint_eval_probe")
    module = evaluator.load_adaptation_module()
    windows, recordings, seizures = module.load_tables()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probes = [(0, 17), (3, 3407)]
    report = []
    for fold, seed in probes:
        patients = evaluator.fold_patients(fold, "test")
        patient = patients[0]
        _, _, frozen_adapter, frozen_head = evaluator.load_ttt_checkpoint(module, fold, seed)
        frozen = evaluator.stream_patient(
            module, frozen_adapter, frozen_head, patient, windows, recordings, device,
            adapt=False, threshold=0.001, seed=seed, update_after_score=True, method="joint",
        )
        _, _, adapted_adapter, adapted_head = evaluator.load_ttt_checkpoint(module, fold, seed)
        adapted = evaluator.stream_patient(
            module, adapted_adapter, adapted_head, patient, windows, recordings, device,
            adapt=True, threshold=0.001, seed=seed, update_after_score=True, method="joint",
        )
        keys = ["recording", "start_s"]
        merged = frozen[keys + ["probability", "block"]].merge(
            adapted[keys + ["probability", "block"]], on=keys, suffixes=("_frozen", "_adapted"), validate="one_to_one"
        )
        diff = merged["probability_adapted"] - merged["probability_frozen"]
        changed = np.abs(diff.to_numpy()) > 1e-8
        threshold = 0.001
        crosses = ((merged.probability_frozen < threshold) & (merged.probability_adapted >= threshold)) | ((merged.probability_frozen >= threshold) & (merged.probability_adapted < threshold))
        report.append({
            "fold": fold,
            "seed": seed,
            "patient": patient,
            "rows": int(len(merged)),
            "blocks": int(merged.block_frozen.nunique()),
            "changed_probability_rows": int(changed.sum()),
            "changed_probability_fraction": float(changed.mean()) if len(changed) else 0.0,
            "max_abs_probability_change": float(np.abs(diff.to_numpy()).max(initial=0.0)),
            "mean_abs_probability_change": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
            "threshold_crossings": int(crosses.sum()),
            "first_block_changed_rows": int(changed[merged.block_frozen == merged.block_frozen.min()].sum()) if len(merged) else 0,
            "later_block_changed_rows": int(changed[merged.block_frozen > merged.block_frozen.min()].sum()) if len(merged) else 0,
        })
    out = ROOT / "outputs/reports/cbramod-joint-ttt-v1-formal/audit_detailed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "probability_drift_probe.json").write_text(json.dumps({"status": "complete", "probes": report}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
