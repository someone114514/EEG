"""Read-only, fold-aware analysis of the three CBraMod TTT experiments.

This script consumes frozen per-run manifests and the already published source
test metrics.  It does not select thresholds, rerun scoring, or read any new
test predictions.  Because the same outer-test patients are repeated across
seeds, the primary summary is a macro-average over five folds after averaging
the three seeds within each fold; seed variability is reported separately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/cbramod-ttt-method-comparison-v1"
FOLDS = range(5)
SEEDS = (17, 42, 3407)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def metric_row(rows: list[dict[str, Any]], *, method: str, condition: str, fold: int, seed: int) -> dict[str, Any]:
    tp = float(sum(float(x.get("true_positive_events", 0.0)) for x in rows))
    fa = float(sum(float(x.get("false_alarm_events", 0.0)) for x in rows))
    truth = float(sum(float(x.get("truth_events", 0.0)) for x in rows))
    hours = float(sum(float(x.get("nonseizure_hours", 0.0)) for x in rows))
    delay_values = [float(x["detection_delay_mean_s"]) for x in rows if x.get("detection_delay_mean_s") is not None and math.isfinite(float(x["detection_delay_mean_s"]))]
    return {
        "method": method,
        "condition": condition,
        "fold": fold,
        "seed": seed,
        "true_positive_events": tp,
        "false_alarm_events": fa,
        "truth_events": truth,
        "nonseizure_hours": hours,
        "event_sensitivity": tp / truth if truth else float("nan"),
        "fa_per_24h": fa * 24.0 / hours if hours else float("nan"),
        "detection_delay_mean_s": float(np.mean(delay_values)) if delay_values else float("nan"),
    }


def source_metric(fold: int, seed: int) -> dict[str, Any]:
    path = ROOT / "runs/v3-groupkfold-confirmatory-v1/cbramod" / f"split{fold}_seed{seed}_main/test/metrics.json"
    payload = read_json(path)
    if payload is None or not isinstance(payload.get("score"), dict):
        return {"method": "source_detector", "condition": "frozen", "fold": fold, "seed": seed, "status": "missing", "source_metrics": str(path.relative_to(ROOT))}
    score = payload["score"]
    return {
        "method": "source_detector",
        "condition": "frozen",
        "fold": fold,
        "seed": seed,
        "true_positive_events": float(score.get("true_positive_events", 0.0)),
        "false_alarm_events": float(score.get("false_alarm_events", 0.0)),
        "truth_events": float(score.get("truth_events", 0.0)),
        "nonseizure_hours": float(score.get("nonseizure_hours", 0.0)),
        "event_sensitivity": float(score.get("event_sensitivity", float("nan"))),
        "fa_per_24h": float(score.get("fa_per_24h", float("nan"))),
        "detection_delay_mean_s": float("nan"),
        "source_metrics": str(path.relative_to(ROOT)),
    }


def collect() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in FOLDS:
        for seed in SEEDS:
            rows.append(source_metric(fold, seed))
            joint = read_json(OUT.parent / "cbramod-joint-ttt-v1-formal" / "evaluation" / f"fold{fold}_seed{seed}" / "manifest.json")
            if joint is None:
                missing.append(f"joint eval fold{fold} seed{seed}")
            else:
                rows.append(metric_row(joint.get("test_frozen_metrics", []), method="joint_ttt", condition="frozen", fold=fold, seed=seed))
                rows.append(metric_row(joint.get("test_adapted_metrics", []), method="joint_ttt", condition="adapted", fold=fold, seed=seed))
            meta = read_json(OUT.parent / "cbramod-meta-ttt-v1-formal" / "evaluation" / f"fold{fold}_seed{seed}" / "manifest.json")
            if meta is None:
                missing.append(f"meta eval fold{fold} seed{seed}")
            else:
                rows.append(metric_row(meta.get("test_frozen_metrics", []), method="meta_ttt", condition="frozen", fold=fold, seed=seed))
                rows.append(metric_row(meta.get("test_adapted_metrics", []), method="meta_ttt", condition="adapted", fold=fold, seed=seed))
            prior = read_json(OUT.parent / "cbramod-label-prior-tta-v1-formal" / "evaluation" / f"fold{fold}_seed{seed}" / "manifest.json")
            if prior is None:
                missing.append(f"label prior eval fold{fold} seed{seed}")
            else:
                rows.append(metric_row(prior.get("test_metrics", []), method="label_prior", condition="gated_postprocess", fold=fold, seed=seed))
    return pd.DataFrame(rows), missing


def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = ["event_sensitivity", "fa_per_24h", "detection_delay_mean_s"]
    valid = table[table.get("status", pd.Series(index=table.index, dtype=object)).fillna("complete") != "missing"].copy()
    seed_rows = []
    for (method, condition, fold), group in valid.groupby(["method", "condition", "fold"], dropna=False):
        row = {"method": method, "condition": condition, "fold": int(fold), "n_seeds": int(len(group))}
        for col in numeric:
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(float)
            row[f"{col}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"{col}_sd"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
        seed_rows.append(row)
    seed_summary = pd.DataFrame(seed_rows)
    fold_rows = []
    for (method, condition), group in seed_summary.groupby(["method", "condition"], dropna=False):
        row = {"method": method, "condition": condition, "n_folds": int(group.fold.nunique())}
        for col in numeric:
            values = pd.to_numeric(group[f"{col}_mean"], errors="coerce").to_numpy(float)
            row[f"{col}_fold_macro_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"{col}_fold_macro_sd"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
        fold_rows.append(row)
    fold_summary = pd.DataFrame(fold_rows)
    source = valid[valid.method == "source_detector"].set_index(["fold", "seed"])
    deltas = []
    for _, row in valid[valid.method != "source_detector"].iterrows():
        key = (int(row.fold), int(row.seed))
        if key not in source.index:
            continue
        base = source.loc[key]
        deltas.append({"method": row.method, "condition": row.condition, "fold": row.fold, "seed": row.seed,
                       "delta_event_sensitivity_vs_source": row.event_sensitivity - base.event_sensitivity,
                       "delta_fa_per_24h_vs_source": row.fa_per_24h - base.fa_per_24h,
                       "delta_detection_delay_mean_s_vs_source": row.detection_delay_mean_s - base.detection_delay_mean_s})
    paired = []
    for method in ("joint_ttt", "meta_ttt"):
        subset = valid[valid.method == method]
        frozen = subset[subset.condition == "frozen"].set_index(["fold", "seed"])
        adapted = subset[subset.condition == "adapted"].set_index(["fold", "seed"])
        for key in sorted(set(frozen.index).intersection(adapted.index)):
            f = frozen.loc[key]; a = adapted.loc[key]
            paired.append({"method": method, "fold": key[0], "seed": key[1],
                           "frozen_event_sensitivity": f.event_sensitivity,
                           "adapted_event_sensitivity": a.event_sensitivity,
                           "delta_event_sensitivity_adapted_minus_frozen": a.event_sensitivity - f.event_sensitivity,
                           "frozen_fa_per_24h": f.fa_per_24h,
                           "adapted_fa_per_24h": a.fa_per_24h,
                           "delta_fa_per_24h_adapted_minus_frozen": a.fa_per_24h - f.fa_per_24h,
                           "frozen_detection_delay_mean_s": f.detection_delay_mean_s,
                           "adapted_detection_delay_mean_s": a.detection_delay_mean_s,
                           "delta_detection_delay_mean_s_adapted_minus_frozen": a.detection_delay_mean_s - f.detection_delay_mean_s})
    return seed_summary, fold_summary, pd.DataFrame(deltas), pd.DataFrame(paired)


def main() -> None:
    table, missing = collect()
    seed_summary, fold_summary, deltas, paired = summarize(table)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "per_run_metrics.csv", index=False)
    seed_summary.to_csv(OUT / "seed_summary.csv", index=False)
    fold_summary.to_csv(OUT / "method_fold_macro_summary.csv", index=False)
    deltas.to_csv(OUT / "paired_deltas_vs_source.csv", index=False)
    paired.to_csv(OUT / "paired_deltas_adapted_minus_frozen.csv", index=False)
    manifest = {
        "release_id": "cbramod-ttt-method-comparison-v1-analysis",
        "status": "complete" if not missing else "INCOMPLETE",
        "missing_runs": missing,
        "primary_aggregation": "mean of three seeds within each outer fold, then macro-average over five folds; repeated seeds are not treated as independent patients",
        "source_metrics": "existing v3-groupkfold-confirmatory-v1 test metrics.json; no new scoring",
        "outputs": ["per_run_metrics.csv", "seed_summary.csv", "method_fold_macro_summary.csv", "paired_deltas_vs_source.csv", "paired_deltas_adapted_minus_frozen.csv"],
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True))
    if not table.empty:
        print(f"rows={len(table)} methods={sorted(table.method.dropna().unique().tolist())}")


if __name__ == "__main__":
    main()
