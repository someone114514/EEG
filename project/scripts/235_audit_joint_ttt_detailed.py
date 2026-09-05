"""Read-only audit of Joint-TTT training, checkpoint changes, and evaluation.

This script never retrains, changes thresholds, or rewrites the official
15-run outputs.  It compares the frozen source checkpoints with the saved
Joint-TTT checkpoints and reconciles training histories with paired frozen /
adapted test metrics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "cbramod-joint-ttt-v1-formal"
RUN_ROOT = ROOT / "outputs/reports" / NAMESPACE / "runs"
EVAL_ROOT = ROOT / "outputs/reports" / NAMESPACE / "evaluation"
SOURCE_ROOT = ROOT / "runs/v3-groupkfold-confirmatory-v1/cbramod"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"
OUT = ROOT / "outputs/reports" / NAMESPACE / "audit_detailed"
UNITS = [(fold, seed) for fold in range(5) for seed in (17, 42, 3407)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(fold: int, seed: int) -> Path:
    return SOURCE_ROOT / f"split{fold}_seed{seed}_main/checkpoints/step_05000.pt"


def load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def delta_stats(source: dict[str, Any], final: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Compare a nested module or state-dict key prefix."""
    if prefix in source and isinstance(source[prefix], dict):
        left = source[prefix]
        right = final.get(prefix, {})
        common = sorted(set(left).intersection(right))
    else:
        left = {key: value for key, value in source.items() if key == prefix or key.startswith(prefix + ".")}
        right = {key: value for key, value in final.items() if key == prefix or key.startswith(prefix + ".")}
        common = sorted(set(left).intersection(right))
    changed_tensors = 0
    changed_elements = 0
    total_elements = 0
    max_abs = 0.0
    sum_sq = 0.0
    source_sq = 0.0
    for key in common:
        a = left[key].detach().cpu().float()
        b = right[key].detach().cpu().float()
        if a.shape != b.shape:
            continue
        d = (b - a).numpy()
        changed = int(np.count_nonzero(d))
        changed_elements += changed
        total_elements += int(d.size)
        if changed:
            changed_tensors += 1
        max_abs = max(max_abs, float(np.max(np.abs(d), initial=0.0)))
        sum_sq += float(np.sum(d * d, dtype=np.float64))
        source_sq += float(np.sum(a.numpy() * a.numpy(), dtype=np.float64))
    l2 = float(np.sqrt(sum_sq))
    source_l2 = float(np.sqrt(source_sq))
    return {
        "prefix": prefix,
        "common_tensor_count": len(common),
        "source_key_examples": sorted(list(source.keys()))[:8],
        "final_key_examples": sorted(list(final.keys()))[:8],
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "total_element_count": total_elements,
        "changed_fraction": changed_elements / total_elements if total_elements else 0.0,
        "max_abs_change": max_abs,
        "l2_change": l2,
        "relative_l2_change": l2 / source_l2 if source_l2 else None,
    }


def metric_summary(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    numeric = {}
    for column in (
        "event_sensitivity",
        "fa_per_24h",
        "detection_delay_mean_s",
        "true_positive_events",
        "false_alarm_events",
        "truth_events",
    ):
        if column in table:
            numeric[column] = float(table[column].mean())
    return {"rows": int(len(table)), "macro_mean": numeric}


def audit_unit(fold: int, seed: int) -> dict[str, Any]:
    name = f"fold{fold}_seed{seed}"
    run_dir = RUN_ROOT / name
    eval_dir = EVAL_ROOT / name
    source_file = source_path(fold, seed)
    final_file = run_dir / "checkpoint.pt"
    source = load(source_file)
    final = load(final_file)
    pretrained_raw = load(PRETRAINED)
    # The original v3 detector checkpoint keeps the CBraMod backbone frozen
    # and stores only the learned projection under encoder.*.  Joint-TTT
    # source training initializes that backbone from the released pretrained
    # weights, so the correct pre-training comparison is pretrained -> final,
    # not source-checkpoint -> final for the backbone.
    pretrained_backbone = {f"backbone.{key}": value for key, value in pretrained_raw.items()}
    history = json.loads((run_dir / "history.json").read_text())
    run_manifest = json.loads((run_dir / "manifest.json").read_text())
    eval_manifest = json.loads((eval_dir / "manifest.json").read_text())
    updates = [int(row["update"]) for row in history]
    detection = np.asarray([float(row["detection_loss"]) for row in history], dtype=float)
    reconstruction = np.asarray([float(row["reconstruction_loss"]) for row in history], dtype=float)
    total = np.asarray([float(row["total_loss"]) for row in history], dtype=float)
    grad = np.asarray([float(row["grad_norm_pre_clip"]) for row in history], dtype=float)
    validation_records = [row for row in history if "validation_detection_loss" in row]
    balance_pairs = sorted({(int(row.get("positive_n", -1)), int(row.get("negative_n", -1))) for row in history})
    frozen = pd.read_csv(eval_dir / "test_frozen_metrics.csv")
    adapted = pd.read_csv(eval_dir / "test_metrics.csv")
    paired = frozen.merge(adapted, on="patient", suffixes=("_frozen", "_adapted"), validate="one_to_one")
    fp_delta = paired["false_alarm_events_adapted"] - paired["false_alarm_events_frozen"]
    tp_delta = paired["true_positive_events_adapted"] - paired["true_positive_events_frozen"]
    sens_delta = paired["event_sensitivity_adapted"] - paired["event_sensitivity_frozen"]
    adapted_probability_table = pd.read_parquet(eval_dir / "test_adapted_probabilities.parquet", columns=["probability", "block", "adapted_after_score"])
    probability = adapted_probability_table["probability"].to_numpy(dtype=float)
    probability_stats = {
        "rows": int(len(probability)),
        "min": float(np.min(probability)) if len(probability) else None,
        "q01": float(np.quantile(probability, 0.01)) if len(probability) else None,
        "q05": float(np.quantile(probability, 0.05)) if len(probability) else None,
        "median": float(np.median(probability)) if len(probability) else None,
        "q95": float(np.quantile(probability, 0.95)) if len(probability) else None,
        "max": float(np.max(probability)) if len(probability) else None,
        "fraction_below_0_001": float(np.mean(probability < 0.001)) if len(probability) else None,
        "fraction_below_0_01": float(np.mean(probability < 0.01)) if len(probability) else None,
    }
    validation_probability = pd.read_parquet(eval_dir / "validation_adapted_probabilities.parquet", columns=["probability", "label"])
    validation_by_label = {}
    for label in (0.0, 1.0):
        values = validation_probability.loc[validation_probability["label"] == label, "probability"].to_numpy(dtype=float)
        validation_by_label[str(int(label))] = {
            "rows": int(len(values)),
            "min": float(np.min(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "q95": float(np.quantile(values, 0.95)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
            "fraction_ge_0_001": float(np.mean(values >= 0.001)) if len(values) else None,
            "fraction_ge_0_01": float(np.mean(values >= 0.01)) if len(values) else None,
        }
    return {
        "unit": name,
        "fold": fold,
        "seed": seed,
        "source_checkpoint_sha256": sha256(source_file),
        "final_checkpoint_sha256": sha256(final_file),
        "source_update_metadata": source.get("update"),
        "final_update_metadata": final.get("update"),
        "history_length": len(history),
        "history_updates_contiguous_1_to_5000": updates == list(range(1, 5001)),
        "history_finite": bool(np.isfinite(detection).all() and np.isfinite(reconstruction).all() and np.isfinite(total).all() and np.isfinite(grad).all()),
        "history_label_balance_pairs": balance_pairs,
        "history_all_batches_balanced_2_plus_2": balance_pairs == [(2, 2)],
        "training": {
            "best_validation_update": run_manifest.get("best_validation_update"),
            "best_validation_loss": run_manifest.get("best_validation_loss"),
            "validation_checks": int(len(validation_records)),
            "last_validation_detection_loss": float(validation_records[-1]["validation_detection_loss"]) if validation_records else None,
            "validation_loss_at_best": float(min(float(row["validation_detection_loss"]) for row in validation_records)) if validation_records else None,
            "first_total_loss": float(total[0]),
            "last_total_loss": float(total[-1]),
            "first_detection_loss": float(detection[0]),
            "last_detection_loss": float(detection[-1]),
            "median_grad_norm": float(np.median(grad)),
            "max_grad_norm": float(np.max(grad)),
            "mean_grad_norm": float(np.mean(grad)),
            "mean_total_loss_first_100": float(np.mean(total[:100])),
            "mean_total_loss_last_100": float(np.mean(total[-100:])),
            "mean_detection_loss_last_100": float(np.mean(detection[-100:])),
            "mean_reconstruction_loss_last_100": float(np.mean(reconstruction[-100:])),
        },
        "checkpoint_deltas": [
            delta_stats(pretrained_backbone, final["encoder"], "backbone"),
            delta_stats(source["encoder"], final["encoder"], "projection"),
            delta_stats(source, final, "head"),
        ],
        "evaluation": {
            "threshold": eval_manifest.get("threshold"),
            "threshold_source": eval_manifest.get("threshold_source"),
            "update_after_score": eval_manifest.get("update_after_score"),
            "test_labels_used_for_adaptation": eval_manifest.get("test_labels_used_for_adaptation"),
            "frozen": metric_summary(eval_dir / "test_frozen_metrics.csv"),
            "adapted": metric_summary(eval_dir / "test_metrics.csv"),
            "paired_patients": int(len(paired)),
            "patients_with_changed_fp_count": int(np.count_nonzero(fp_delta)),
            "patients_with_changed_tp_count": int(np.count_nonzero(tp_delta)),
            "patients_with_changed_sensitivity_count": int(np.count_nonzero(sens_delta)),
            "sum_fp_delta": int(fp_delta.sum()),
            "sum_tp_delta": int(tp_delta.sum()),
            "mean_sensitivity_delta": float(sens_delta.mean()),
            "mean_fa24_delta": float(paired["fa_per_24h_adapted"].mean() - paired["fa_per_24h_frozen"].mean()),
            "adapted_probability_stats": probability_stats,
            "test_block_count": int(adapted_probability_table["block"].nunique()),
            "test_adapted_after_score_all_true": bool(adapted_probability_table["adapted_after_score"].astype(bool).all()),
            "validation_probability_by_label": validation_by_label,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    units = [audit_unit(fold, seed) for fold, seed in UNITS]
    backbones = [unit["checkpoint_deltas"][0] for unit in units]
    projections = [unit["checkpoint_deltas"][1] for unit in units]
    heads = [unit["checkpoint_deltas"][2] for unit in units]
    training = [unit["training"] for unit in units]
    evaluations = [unit["evaluation"] for unit in units]
    report = {
        "namespace": NAMESPACE,
        "status": "read_only_audit_complete",
        "units": units,
        "checks": {
            "unit_count": len(units),
            "all_histories_5000_contiguous": all(x["history_updates_contiguous_1_to_5000"] for x in units),
            "all_histories_finite": all(x["history_finite"] for x in units),
            "all_training_batches_balanced_2_plus_2": all(x["history_all_batches_balanced_2_plus_2"] for x in units),
            "all_backbones_changed": all(x["checkpoint_deltas"][0]["changed_tensor_count"] > 0 for x in units),
            "all_joint_training_heads_changed": all(x["checkpoint_deltas"][2]["changed_tensor_count"] > 0 for x in units),
            "all_eval_thresholds_validation_only": all(x["evaluation"]["threshold_source"] == "validation-only median of patient validation selections" for x in units),
            "all_eval_test_labels_unused": all(x["evaluation"]["test_labels_used_for_adaptation"] is False for x in units),
            "all_eval_adapted_after_score_flag_true": all(x["evaluation"]["test_adapted_after_score_all_true"] for x in units),
            "all_eval_test_streams_have_multiple_blocks": all(x["evaluation"]["test_block_count"] > 1 for x in units),
        },
        "aggregate": {
            "backbone_l2_change_min": float(min(x["l2_change"] for x in backbones)),
            "backbone_l2_change_median": float(np.median([x["l2_change"] for x in backbones])),
            "backbone_l2_change_max": float(max(x["l2_change"] for x in backbones)),
            "backbone_relative_l2_change_min": float(min(x["relative_l2_change"] for x in backbones)),
            "backbone_relative_l2_change_median": float(np.median([x["relative_l2_change"] for x in backbones])),
            "backbone_relative_l2_change_max": float(max(x["relative_l2_change"] for x in backbones)),
            "backbone_max_abs_change_min": float(min(x["max_abs_change"] for x in backbones)),
            "backbone_max_abs_change_median": float(np.median([x["max_abs_change"] for x in backbones])),
            "backbone_max_abs_change_max": float(max(x["max_abs_change"] for x in backbones)),
            "projection_relative_l2_change_min": float(min(x["relative_l2_change"] for x in projections)),
            "projection_relative_l2_change_median": float(np.median([x["relative_l2_change"] for x in projections])),
            "projection_relative_l2_change_max": float(max(x["relative_l2_change"] for x in projections)),
            "head_relative_l2_change_min": float(min(x["relative_l2_change"] for x in heads)),
            "head_relative_l2_change_median": float(np.median([x["relative_l2_change"] for x in heads])),
            "head_relative_l2_change_max": float(max(x["relative_l2_change"] for x in heads)),
            "best_validation_updates": [x["best_validation_update"] for x in training],
            "joint_eval_changed_patient_units": int(sum(x["patients_with_changed_fp_count"] > 0 or x["patients_with_changed_tp_count"] > 0 for x in evaluations)),
            "joint_eval_sum_fp_delta": int(sum(x["sum_fp_delta"] for x in evaluations)),
            "joint_eval_sum_tp_delta": int(sum(x["sum_tp_delta"] for x in evaluations)),
        },
    }
    (OUT / "joint_ttt_detailed_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n")
    rows = []
    for unit in units:
        back = unit["checkpoint_deltas"][0]
        train = unit["training"]
        ev = unit["evaluation"]
        rows.append({
            "unit": unit["unit"],
            "history_length": unit["history_length"],
            "best_validation_update": train["best_validation_update"],
            "validation_loss_at_best": train["validation_loss_at_best"],
            "last_validation_detection_loss": train["last_validation_detection_loss"],
            "final_total_loss": train["last_total_loss"],
            "median_grad_norm": train["median_grad_norm"],
            "backbone_changed_tensors": back["changed_tensor_count"],
            "backbone_max_abs_change": back["max_abs_change"],
            "backbone_l2_change": back["l2_change"],
            "patients_with_changed_fp_count": ev["patients_with_changed_fp_count"],
            "sum_fp_delta": ev["sum_fp_delta"],
            "mean_sensitivity_delta": ev["mean_sensitivity_delta"],
            "mean_fa24_delta": ev["mean_fa24_delta"],
            "threshold": ev["threshold"],
            "adapted_probability_min": ev["adapted_probability_stats"]["min"],
            "adapted_probability_q01": ev["adapted_probability_stats"]["q01"],
            "adapted_probability_median": ev["adapted_probability_stats"]["median"],
            "adapted_probability_fraction_below_0_001": ev["adapted_probability_stats"]["fraction_below_0_001"],
            "validation_negative_median": ev["validation_probability_by_label"]["0"]["median"],
            "validation_positive_median": ev["validation_probability_by_label"]["1"]["median"],
            "validation_negative_fraction_ge_0_001": ev["validation_probability_by_label"]["0"]["fraction_ge_0_001"],
            "validation_positive_fraction_ge_0_001": ev["validation_probability_by_label"]["1"]["fraction_ge_0_001"],
        })
    pd.DataFrame(rows).to_csv(OUT / "joint_ttt_detailed_audit.csv", index=False)
    print(json.dumps({"checks": report["checks"], "aggregate": report["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
