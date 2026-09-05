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

from .meta_evaluate import CONDITIONS, SEIZURE_OFFSET_POST_S, SEIZURE_ONSET_PRE_S, SEIZURE_SCORING_REVISION, score_probabilities


COMPARISONS = (
    ("meta_band_ttt", "meta_band_frozen"),
    ("meta_temporal_ttt", "meta_temporal_frozen"),
    ("meta_band_frozen", "supervised_frozen"),
    ("meta_temporal_frozen", "supervised_frozen"),
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


def bh_fdr(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for index in range(len(p) - 1, -1, -1):
        rank = index + 1
        running = min(running, float(p[order[index]] * len(p) / rank))
        q[order[index]] = running
    return q.tolist()


@dataclass
class Components:
    true_positive_events: int
    false_alarm_events: int
    truth_events: int
    false_alarm_time_seconds: float
    total_monitoring_hours: float
    nonseizure_hours: float
    delay_sum_s: float
    delay_count: int


def patient_components(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, threshold: float) -> dict[str, Components]:
    output: dict[str, Components] = {}
    for patient, group in table.groupby("patient", sort=True):
        metric = score_probabilities(group, seizures, recordings, threshold)
        output[str(patient)] = Components(
            int(metric["true_positive_events"]), int(metric["false_alarm_events"]), int(metric["truth_events"]),
            float(metric["false_alarm_time_seconds"]), float(metric["total_monitoring_hours"]),
            float(metric["nonseizure_hours"]), float(metric.get("detection_delay_sum_s", 0.0)), int(metric.get("detection_delay_count", 0)),
        )
    return output


def aggregate(components: dict[str, Components], patients: list[str] | None = None) -> dict[str, float | int]:
    selected = list(components) if patients is None else patients
    values = [components[patient] for patient in selected]
    tp = sum(value.true_positive_events for value in values)
    fp = sum(value.false_alarm_events for value in values)
    truth = sum(value.truth_events for value in values)
    false_alarm_time = sum(value.false_alarm_time_seconds for value in values)
    total_hours = sum(value.total_monitoring_hours for value in values)
    nonseizure_hours = sum(value.nonseizure_hours for value in values)
    delay_sum = sum(value.delay_sum_s for value in values)
    delay_count = sum(value.delay_count for value in values)
    return {
        "patients": len(selected), "true_positive_events": tp, "false_alarm_events": fp, "truth_events": truth,
        "false_alarm_time_seconds": false_alarm_time, "total_monitoring_hours": total_hours,
        "nonseizure_hours": nonseizure_hours,
        "event_sensitivity": tp / truth if truth else float("nan"),
        "fa_per_24h": fp * 24.0 / nonseizure_hours if nonseizure_hours else float("nan"),
        "false_alarm_time_min_per_24h": false_alarm_time / 60.0 * 24.0 / total_hours if total_hours else float("nan"),
        "false_alarm_time_s_per_24h": false_alarm_time * 24.0 / total_hours if total_hours else float("nan"),
        "detection_delay_mean_s": delay_sum / delay_count if delay_count else float("nan"),
    }


def load_condition(root: Path, condition: str, seizures: pd.DataFrame, recordings: pd.DataFrame) -> tuple[dict[str, Components], pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    components: dict[str, Components] = {}
    locks: list[dict[str, Any]] = []
    for fold in range(5):
        run = root / "evaluation" / condition / f"fold{fold}_seed3407"
        completion_path = run / "test_completed.json"
        validation_path = run / "validation_metrics.json"
        probability_path = run / "test_probabilities.parquet"
        if not all(path.is_file() for path in (completion_path, validation_path, probability_path)):
            raise FileNotFoundError(run)
        completion = json.loads(completion_path.read_text())
        validation = json.loads(validation_path.read_text())
        if completion.get("test_evaluation_count") != 1 or completion.get("threshold_source") != "validation_only":
            raise ValueError(f"test discipline failed: {run}")
        if completion.get("checkpoint_sha256") != validation.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint changed after validation: {run}")
        table = pd.read_parquet(probability_path)
        threshold = float(validation["selected_event_operating_point"]["threshold"])
        current = patient_components(table, seizures, recordings, threshold)
        overlap = set(components) & set(current)
        if overlap:
            raise ValueError(f"patient appears in multiple outer folds: {sorted(overlap)}")
        components.update(current)
        parts.append(table.assign(fold=fold, condition=condition, threshold=threshold))
        locks.append({"fold": fold, "threshold": threshold, "checkpoint_sha256": completion["checkpoint_sha256"], "probability_sha256": sha256(probability_path)})
    return components, pd.concat(parts, ignore_index=True), locks


def run(root: Path, recordings_path: Path, seizures_path: Path, bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    recordings = pd.read_parquet(recordings_path)
    seizures = pd.read_parquet(seizures_path)
    all_components: dict[str, dict[str, Components]] = {}
    metrics: list[dict[str, Any]] = []
    locks: dict[str, Any] = {}
    for condition in CONDITIONS:
        current, oof, condition_locks = load_condition(root, condition, seizures, recordings)
        all_components[condition] = current
        metrics.append({"condition": condition, **aggregate(current)})
        oof_path = output / f"{condition}_oof_probabilities.parquet"
        oof.to_parquet(oof_path, index=False)
        locks[condition] = {"folds": condition_locks, "oof_sha256": sha256(oof_path)}
    patient_sets = [set(value) for value in all_components.values()]
    if any(value != patient_sets[0] for value in patient_sets[1:]) or len(patient_sets[0]) != 22:
        raise ValueError("conditions do not cover the same 22 outer-test patients")
    patients = sorted(patient_sets[0])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(bootstrap_replicates):
        sample = rng.choice(patients, size=len(patients), replace=True).tolist()
        for condition in CONDITIONS:
            rows.append({"replicate": replicate, "condition": condition, **aggregate(all_components[condition], sample)})
    bootstrap = pd.DataFrame(rows)
    bootstrap_path = output / f"patient_bootstrap_{bootstrap_replicates}.parquet"
    bootstrap.to_parquet(bootstrap_path, index=False)
    point = {row["condition"]: row for row in metrics}
    comparison_rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for treatment, reference in COMPARISONS:
        left = bootstrap[bootstrap.condition == treatment].set_index("replicate")
        right = bootstrap[bootstrap.condition == reference].set_index("replicate")
        delta_time = (left.false_alarm_time_min_per_24h - right.false_alarm_time_min_per_24h).to_numpy()
        delta_fa = (left.fa_per_24h - right.fa_per_24h).to_numpy()
        delta_sens = (left.event_sensitivity - right.event_sensitivity).to_numpy()
        p = min(1.0, 2 * min(float(np.mean(delta_time <= 0)), float(np.mean(delta_time >= 0))))
        p_values.append(p)
        comparison_rows.append({
            "treatment": treatment, "reference": reference,
            "delta_sensitivity": float(point[treatment]["event_sensitivity"] - point[reference]["event_sensitivity"]),
            "delta_sensitivity_ci_low": float(np.quantile(delta_sens, .025)), "delta_sensitivity_ci_high": float(np.quantile(delta_sens, .975)),
            "delta_fa_per_24h": float(point[treatment]["fa_per_24h"] - point[reference]["fa_per_24h"]),
            "delta_fa_ci_low": float(np.quantile(delta_fa, .025)), "delta_fa_ci_high": float(np.quantile(delta_fa, .975)),
            "delta_false_alarm_time_min_per_24h": float(point[treatment]["false_alarm_time_min_per_24h"] - point[reference]["false_alarm_time_min_per_24h"]),
            "delta_false_alarm_time_ci_low": float(np.quantile(delta_time, .025)), "delta_false_alarm_time_ci_high": float(np.quantile(delta_time, .975)),
            "p_value_two_sided_delta_false_alarm_time": p,
            "ttt_positive_gate": bool(float(point[treatment]["event_sensitivity"] - point[reference]["event_sensitivity"]) >= -.03 and float(np.quantile(delta_time, .975)) < 0) if "ttt" in treatment else None,
        })
    for row, q in zip(comparison_rows, bh_fdr(p_values), strict=True):
        row["q_value_bh_fdr"] = q
    metrics_path = output / "condition_metrics.csv"; comparisons_path = output / "prespecified_comparisons.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False); pd.DataFrame(comparison_rows).to_csv(comparisons_path, index=False)
    manifest = {
        "release_id": "meta-ttt-chbmit-5fold-v1", "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "seizure_scoring_revision": SEIZURE_SCORING_REVISION,
        "conditions": list(CONDITIONS), "comparisons": [list(value) for value in COMPARISONS], "patients": patients,
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed,
        "threshold_source": "validation_only_per_fold_condition", "test_evaluation_count_per_fold_condition": 1,
        "false_alarm_time_definition": "predicted_alarm_union minus raw seizure_union; total_monitoring_hours denominator",
        "alarm_time_truth_definition": "raw seizure interval; onset/offset collar is excluded from alarm-time subtraction",
        "false_alarm_event_rate_definition": "event matching uses collar-expanded truths; denominator uses raw non-seizure monitoring time",
        "seizure_scoring_truth_definition": "raw interval expanded to onset-30s and offset+60s for event matching, clipped to evaluable EDF",
        "detection_delay_reference": "raw onset, clipped to evaluation start",
        "seizure_scoring_collar_onset_pre_s": SEIZURE_ONSET_PRE_S,
        "seizure_scoring_collar_offset_post_s": SEIZURE_OFFSET_POST_S,
        "locks": locks,
        "artifacts": {"condition_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)}, "comparisons": {"path": str(comparisons_path), "sha256": sha256(comparisons_path)}, "bootstrap": {"path": str(bootstrap_path), "sha256": sha256(bootstrap_path)}},
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/meta-ttt-chbmit-5fold-v1')
    parser.add_argument("--recordings", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'manifests/recordings.parquet')
    parser.add_argument("--seizures", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'manifests/seizures.parquet')
    parser.add_argument("--bootstrap-replicates", type=int, default=2000); parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(); print(json.dumps(run(args.output_root, args.recordings, args.seizures, args.bootstrap_replicates, args.seed), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
