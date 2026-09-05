from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import (
    CONDITIONS,
    SEIZURE_OFFSET_POST_S,
    SEIZURE_ONSET_PRE_S,
    _scoring_truths,
    _union_length,
    eventize,
    match_events,
)


COMPARISONS = (
    ("band_joint_frozen", "supervised_frozen"),
    ("mask_joint_frozen", "supervised_frozen"),
    ("band_joint_band_ttt", "band_joint_frozen"),
    ("mask_joint_mask_ttt", "mask_joint_frozen"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def bh_fdr(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, float(values[original_index] * len(values) / rank))
        adjusted[original_index] = running
    return adjusted.tolist()


@dataclass
class Components:
    true_positive_events: int
    false_alarm_events: int
    false_alarm_time_seconds: float
    truth_events: int
    nonseizure_hours: float
    delays: list[float]


def patient_components(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, threshold: float) -> dict[str, Components]:
    output: dict[str, Components] = {}
    for patient, patient_table in table.groupby("patient", sort=True):
        true_positives = false_alarms = truth_events = 0
        false_alarm_time_seconds = 0.0
        nonseizure_seconds = 0.0
        delays: list[float] = []
        for recording_id, group in patient_table.groupby("recording", sort=False):
            ordered = group.sort_values("end", kind="stable")
            evaluation_start = float(ordered.end.iloc[0])
            recording = recordings[recordings.recording_id == recording_id]
            if recording.empty:
                raise ValueError(f"missing recording metadata: {recording_id}")
            duration = float(recording.duration_s.iloc[0])
            scoring_truths, raw_truths = _scoring_truths(seizures, recording_id, evaluation_start, duration)
            predictions = eventize(
                ordered.end.to_numpy(dtype=float),
                ordered.probability.to_numpy(dtype=float),
                threshold=threshold,
            )
            matched = match_events(predictions, scoring_truths)
            true_positives += len(matched.pairs)
            false_alarms += len(matched.unmatched_predictions)
            truth_events += len(scoring_truths)
            for prediction in predictions:
                overlap = _union_length([
                    (max(prediction.start_s, truth_start), min(prediction.end_s, truth_end))
                    for truth_start, truth_end in raw_truths
                    if min(prediction.end_s, truth_end) > max(prediction.start_s, truth_start)
                ])
                false_alarm_time_seconds += max(0.0, prediction.end_s - prediction.start_s - overlap)
            delays.extend(
                max(0.0, predictions[pair.prediction_index].start_s - raw_truths[pair.truth_index][0])
                for pair in matched.pairs
            )
            raw_seizure_seconds = _union_length(raw_truths)
            nonseizure_seconds += max(0.0, duration - evaluation_start - raw_seizure_seconds)
        output[str(patient)] = Components(true_positives, false_alarms, false_alarm_time_seconds, truth_events, nonseizure_seconds / 3600.0, delays)
    return output


def aggregate(components: dict[str, Components], patients: list[str] | None = None) -> dict[str, float | int]:
    selected = list(components) if patients is None else patients
    values = [components[patient] for patient in selected]
    tp = sum(value.true_positive_events for value in values)
    fp = sum(value.false_alarm_events for value in values)
    false_alarm_time_seconds = sum(value.false_alarm_time_seconds for value in values)
    truth = sum(value.truth_events for value in values)
    hours = sum(value.nonseizure_hours for value in values)
    delays = [delay for value in values for delay in value.delays]
    return {
        "patients": len(selected),
        "true_positive_events": tp,
        "false_alarm_events": fp,
        "false_alarm_time_seconds": false_alarm_time_seconds,
        "truth_events": truth,
        "nonseizure_hours": hours,
        "event_sensitivity": tp / truth if truth else float("nan"),
        "fa_per_24h": fp * 24.0 / hours if hours else float("nan"),
        "false_alarm_time_s_per_24h": false_alarm_time_seconds * 24.0 / hours if hours else float("nan"),
        "false_alarm_time_min_per_24h": false_alarm_time_seconds / 60.0 * 24.0 / hours if hours else float("nan"),
        "detection_delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
        "detection_delay_median_s": float(np.median(delays)) if delays else float("nan"),
    }


def load_condition(root: Path, condition: str, seizures: pd.DataFrame, recordings: pd.DataFrame) -> tuple[dict[str, Components], dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    locks: list[dict[str, Any]] = []
    all_components: dict[str, Components] = {}
    for fold in range(5):
        run = root / "evaluation" / condition / f"fold{fold}_seed3407"
        completion_path = run / "test_completed.json"
        validation_path = run / "validation_metrics.json"
        probability_path = run / "test_probabilities.parquet"
        if not completion_path.is_file() or not validation_path.is_file() or not probability_path.is_file():
            raise FileNotFoundError(f"incomplete evaluation: {run}")
        completion = json.loads(completion_path.read_text())
        validation = json.loads(validation_path.read_text())
        if completion.get("test_evaluation_count") != 1 or completion.get("threshold_source") != "validation_only":
            raise ValueError(f"invalid test discipline: {completion_path}")
        if completion["checkpoint_sha256"] != validation["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch: {run}")
        table = pd.read_parquet(probability_path)
        threshold = float(validation["selected_event_operating_point"]["threshold"])
        fold_components = patient_components(table, seizures, recordings, threshold)
        overlap = set(all_components) & set(fold_components)
        if overlap:
            raise ValueError(f"patients evaluated in multiple outer folds: {sorted(overlap)}")
        all_components.update(fold_components)
        parts.append(table.assign(fold=fold, condition=condition, threshold=threshold))
        locks.append({
            "fold": fold,
            "threshold": threshold,
            "checkpoint_sha256": completion["checkpoint_sha256"],
            "probability_sha256": sha256(probability_path),
        })
    oof = pd.concat(parts, ignore_index=True)
    return all_components, {"locks": locks, "oof": oof}


def run(root: Path, recordings_path: Path, seizures_path: Path, bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    recordings = pd.read_parquet(recordings_path)
    seizures = pd.read_parquet(seizures_path)
    components: dict[str, dict[str, Components]] = {}
    condition_metrics: list[dict[str, Any]] = []
    locks: dict[str, Any] = {}
    for condition in CONDITIONS:
        current, detail = load_condition(root, condition, seizures, recordings)
        components[condition] = current
        metric = {"condition": condition, **aggregate(current)}
        condition_metrics.append(metric)
        oof_path = output / f"{condition}_oof_probabilities.parquet"
        detail["oof"].to_parquet(oof_path, index=False)
        locks[condition] = {"folds": detail["locks"], "oof_sha256": sha256(oof_path)}
    patient_sets = [set(value) for value in components.values()]
    if any(value != patient_sets[0] for value in patient_sets[1:]) or len(patient_sets[0]) != 22:
        raise ValueError("conditions do not cover the same 22 outer-test patients")
    patients = sorted(patient_sets[0])
    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, Any]] = []
    for replicate in range(bootstrap_replicates):
        sample = rng.choice(patients, size=len(patients), replace=True).tolist()
        for condition in CONDITIONS:
            bootstrap_rows.append({"replicate": replicate, "condition": condition, **aggregate(components[condition], sample)})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap_path = output / "patient_bootstrap_2000.parquet"
    bootstrap.to_parquet(bootstrap_path, index=False)
    comparison_rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    point = {row["condition"]: row for row in condition_metrics}
    for treatment, reference in COMPARISONS:
        left = bootstrap[bootstrap.condition == treatment].set_index("replicate")
        right = bootstrap[bootstrap.condition == reference].set_index("replicate")
        delta_fa = (left.fa_per_24h - right.fa_per_24h).to_numpy()
        delta_fa_time = (left.false_alarm_time_s_per_24h - right.false_alarm_time_s_per_24h).to_numpy()
        delta_sensitivity = (left.event_sensitivity - right.event_sensitivity).to_numpy()
        p_value = min(1.0, 2.0 * min(float(np.mean(delta_fa_time <= 0)), float(np.mean(delta_fa_time >= 0))))
        p_values.append(p_value)
        sensitivity_point = float(point[treatment]["event_sensitivity"] - point[reference]["event_sensitivity"])
        fa_point = float(point[treatment]["fa_per_24h"] - point[reference]["fa_per_24h"])
        comparison_rows.append({
            "treatment": treatment,
            "reference": reference,
            "delta_sensitivity": sensitivity_point,
            "delta_sensitivity_ci_low": float(np.quantile(delta_sensitivity, 0.025)),
            "delta_sensitivity_ci_high": float(np.quantile(delta_sensitivity, 0.975)),
            "delta_fa_per_24h": fa_point,
            "delta_fa_ci_low": float(np.quantile(delta_fa, 0.025)),
            "delta_fa_ci_high": float(np.quantile(delta_fa, 0.975)),
            "delta_false_alarm_time_s_per_24h": float(point[treatment]["false_alarm_time_s_per_24h"] - point[reference]["false_alarm_time_s_per_24h"]),
            "delta_false_alarm_time_ci_low": float(np.quantile(delta_fa_time, 0.025)),
            "delta_false_alarm_time_ci_high": float(np.quantile(delta_fa_time, 0.975)),
            "p_value_two_sided_delta_false_alarm_time": p_value,
            "ttt_positive_gate": bool(sensitivity_point >= -0.03 and float(np.quantile(delta_fa_time, 0.975)) < 0) if "ttt" in treatment else None,
        })
    q_values = bh_fdr(p_values)
    for row, q_value in zip(comparison_rows, q_values, strict=True):
        row["q_value_bh_fdr"] = q_value
    metrics_path = output / "condition_metrics.csv"
    comparisons_path = output / "prespecified_comparisons.csv"
    pd.DataFrame(condition_metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparisons_path, index=False)
    manifest = {
        "release_id": "neurottt-chbmit-5fold-v1",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "conditions": list(CONDITIONS),
        "comparisons": [list(value) for value in COMPARISONS],
        "patients": patients,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "threshold_source": "validation_only_per_fold_condition",
        "seizure_scoring_collar": {
            "onset_pre_s": SEIZURE_ONSET_PRE_S,
            "offset_post_s": SEIZURE_OFFSET_POST_S,
            "clip_to_recording": True,
        },
        "seizure_scoring_truth_definition": "raw interval expanded for event matching/sensitivity only",
        "alarm_time_truth_definition": "original unexpanded seizure interval; collar excluded from alarm-time subtraction and denominator",
        "detection_delay_reference": "original annotated onset, clipped to evaluable EDF",
        "test_evaluation_count_per_fold_condition": 1,
        "locks": locks,
        "artifacts": {
            "condition_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
            "comparisons": {"path": str(comparisons_path), "sha256": sha256(comparisons_path)},
            "bootstrap": {"path": str(bootstrap_path), "sha256": sha256(bootstrap_path)},
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/neurottt-chbmit-5fold-v1')
    parser.add_argument("--recordings", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'manifests/recordings.parquet')
    parser.add_argument("--seizures", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'manifests/seizures.parquet')
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.output_root, args.recordings, args.seizures, args.bootstrap_replicates, args.seed), indent=2), flush=True)


if __name__ == "__main__":
    main()
