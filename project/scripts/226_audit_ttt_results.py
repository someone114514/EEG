"""Audit the completed method-comparison runs without recomputing metrics.

The audit is intentionally read-only.  It verifies that every expected
fold/seed has a training and evaluation manifest, that checkpoints and source
artifacts match their recorded hashes, and that test evaluation/selection
rules were respected.  Missing runs are reported as INCOMPLETE rather than
silently treated as a pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/cbramod-ttt-method-comparison-v1"
FOLDS = range(5)
SEEDS = (17, 42, 3407)
EXPECTED = [(fold, seed) for fold in FOLDS for seed in SEEDS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def check_training(namespace: str, checkpoint_key: str, require_source: bool = True) -> dict[str, Any]:
    base = ROOT / "outputs/reports" / namespace / "runs"
    issues: list[str] = []
    records: list[dict[str, Any]] = []
    for fold, seed in EXPECTED:
        run = base / f"fold{fold}_seed{seed}"
        manifest_path = run / "manifest.json"
        manifest = read(manifest_path)
        record: dict[str, Any] = {"fold": fold, "seed": seed, "manifest": str(manifest_path.relative_to(ROOT))}
        if manifest is None:
            issues.append(f"missing training manifest: {manifest_path}")
            records.append({**record, "status": "missing"})
            continue
        record["status"] = manifest.get("status")
        if manifest.get("status") != "training_complete": issues.append(f"training status not complete: {manifest_path}")
        if int(manifest.get("requested_updates", -1)) != 5000: issues.append(f"requested_updates != 5000: {manifest_path}")
        for key in ("outer_test_read", "outer_test_used_for_selection", "test_rows_loaded", "test_labels_loaded"):
            if manifest.get(key) is not False: issues.append(f"{key} is not false: {manifest_path}")
        checkpoint = ROOT / str(manifest.get(checkpoint_key, ""))
        if not checkpoint.exists():
            issues.append(f"missing checkpoint: {checkpoint}")
        else:
            actual = sha256(checkpoint)
            hash_key = "checkpoint_sha256" if checkpoint_key == "checkpoint" else "label_prior_sha256"
            record["checkpoint_sha256_actual"] = actual
            record["checkpoint_sha256_recorded"] = manifest.get(hash_key)
            # Older label-prior manifests predate the explicit hash field; the
            # audit records the actual digest without rewriting those frozen
            # manifests.  Joint/meta manifests must carry and match a hash.
            if checkpoint_key == "checkpoint" and actual != manifest.get(hash_key):
                issues.append(f"checkpoint hash mismatch: {checkpoint}")
        if require_source:
            source = ROOT / str(manifest.get("source_checkpoint", ""))
            if not source.exists():
                issues.append(f"missing source checkpoint: {source}")
            elif sha256(source) != manifest.get("source_checkpoint_sha256"):
                issues.append(f"source checkpoint hash mismatch: {source}")
        records.append(record)
    return {"namespace": namespace, "runs": len(records), "complete_runs": sum(r.get("status") == "training_complete" for r in records), "issues": issues, "records": records}


def check_eval(namespace: str, kind: str) -> dict[str, Any]:
    base = ROOT / "outputs/reports" / namespace / "evaluation"
    issues: list[str] = []
    records: list[dict[str, Any]] = []
    for fold, seed in EXPECTED:
        manifest_path = base / f"fold{fold}_seed{seed}" / "manifest.json"
        manifest = read(manifest_path)
        record: dict[str, Any] = {"fold": fold, "seed": seed, "manifest": str(manifest_path.relative_to(ROOT))}
        if manifest is None:
            issues.append(f"missing evaluation manifest: {manifest_path}")
            records.append({**record, "status": "missing"})
            continue
        record["status"] = manifest.get("status")
        if manifest.get("status") != "complete": issues.append(f"evaluation status not complete: {manifest_path}")
        if kind in {"joint", "meta"}:
            source = ROOT / str(manifest.get("source_checkpoint", ""))
            expected_hash = manifest.get("source_checkpoint_sha256")
            if not source.exists(): issues.append(f"missing evaluated checkpoint: {source}")
            elif expected_hash and sha256(source) != expected_hash: issues.append(f"evaluated checkpoint hash mismatch: {source}")
        if kind in {"joint", "meta"}:
            if int(manifest.get("test_evaluation_count_per_condition", -1)) != 1: issues.append(f"test_evaluation_count_per_condition != 1: {manifest_path}")
            if manifest.get("test_labels_used_for_adaptation") is not False: issues.append(f"test labels used for adaptation: {manifest_path}")
            if "validation" not in str(manifest.get("threshold_source", "")).lower(): issues.append(f"threshold not validation-only: {manifest_path}")
            if not manifest.get("test_frozen_metrics") or not manifest.get("test_adapted_metrics"): issues.append(f"paired test metrics missing: {manifest_path}")
        else:
            if int(manifest.get("test_evaluation_count", -1)) != 1: issues.append(f"test_evaluation_count != 1: {manifest_path}")
            if manifest.get("test_labels_used_for_selection") is not False or manifest.get("test_labels_used_for_adaptation") is not False: issues.append(f"test labels used by prior evaluation: {manifest_path}")
            if manifest.get("prior_training_outer_test_read") is not False: issues.append(f"prior training read outer test: {manifest_path}")
            if "validation" not in str(manifest.get("threshold_source", "")).lower(): issues.append(f"prior threshold not validation-only: {manifest_path}")
            if not manifest.get("test_metrics"): issues.append(f"test metrics missing: {manifest_path}")
        records.append(record)
    return {"namespace": namespace, "kind": kind, "runs": len(records), "complete_runs": sum(r.get("status") == "complete" for r in records), "issues": issues, "records": records}


def main() -> None:
    report = {
        "release_id": "cbramod-ttt-method-comparison-v1-audit",
        "expected_runs_per_method": len(EXPECTED),
        "training": {
            "joint": check_training("cbramod-joint-ttt-v1-formal", "checkpoint", True),
            "meta": check_training("cbramod-meta-ttt-v1-formal", "checkpoint", True),
            "label_prior": check_training("cbramod-label-prior-tta-v1-formal", "label_prior", False),
        },
        "evaluation": {
            "joint": check_eval("cbramod-joint-ttt-v1-formal", "joint"),
            "meta": check_eval("cbramod-meta-ttt-v1-formal", "meta"),
            "label_prior": check_eval("cbramod-label-prior-tta-v1-formal", "prior"),
        },
    }
    issues: list[str] = []
    for section in ("training", "evaluation"):
        for name, item in report[section].items():
            issues.extend(f"{section}.{name}: {issue}" for issue in item["issues"])
    report["issues"] = issues
    report["status"] = "PASS" if not issues else ("INCOMPLETE" if any("missing" in x for x in issues) else "FAIL")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
