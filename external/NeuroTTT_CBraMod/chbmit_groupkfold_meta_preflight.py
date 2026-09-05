from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from chbmit_groupkfold.data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, fold_manifest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(tmp, path)


def active_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,cmd="], check=True, capture_output=True, text=True)
    lines = []
    ignored = {os.getpid(), os.getppid()}
    ignored.update(int(value) for value in os.environ.get("META_PREFLIGHT_IGNORE_PIDS", "").split(",") if value.strip().isdigit())
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        pid = int(parts[0]) if parts and parts[0].isdigit() else -1
        if pid not in ignored and "chbmit_groupkfold" in line and "meta_preflight" not in line and "grep" not in line:
            lines.append(line.strip())
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    issues: list[str] = []
    active = active_processes()
    if active:
        issues.append("active_chbmit_groupkfold_process")
    existing = [path for path in root.iterdir() if path.name not in {"preflight.json", "freeze"}] if root.exists() else []
    if existing and not args.allow_existing:
        # A pre-existing namespace is never reused.  This catches accidental
        # restarts and checkpoint reuse before any model is instantiated.
        issues.append("target_namespace_exists_or_nonempty")
    if existing and args.allow_existing:
        allowed = {
            "logs",
            "runs",
            "queue_status.json",
            "evaluation",
            "evaluation_logs",
            "evaluation_queue_status.json",
        }
        unexpected = sorted(path.name for path in existing if path.name not in allowed)
        if unexpected:
            issues.append("resume_namespace_has_unexpected_entries")

    folds: dict[str, Any] = {}
    all_patients: list[str] = []
    test_patients: list[str] = []
    for fold in range(5):
        payload = fold_manifest(fold, args.fold_root)
        train, validation, test = (set(map(str, payload[key])) for key in ("train", "validation", "test"))
        if train & validation or train & test or validation & test:
            issues.append(f"patient_overlap_fold_{fold}")
        all_patients.extend(sorted(train | validation | test))
        test_patients.extend(sorted(test))
        folds[str(fold)] = {key: sorted(map(str, payload[key])) for key in ("train", "validation", "test")}
    counts: dict[str, int] = {}
    for patient in all_patients:
        counts[patient] = counts.get(patient, 0) + 1
    if any(count != 5 for count in counts.values()):
        issues.append("fold_manifest_patient_coverage_not_exactly_once_per_fold")
    test_counts: dict[str, int] = {}
    for patient in test_patients:
        test_counts[patient] = test_counts.get(patient, 0) + 1
    if any(count != 1 for count in test_counts.values()):
        issues.append("outer_test_patient_appears_in_multiple_folds")

    windows_hash = sha256(args.windows)
    cv_hash = sha256(args.fold_root / "cv_manifest.json")
    pretrained_hash = sha256(args.pretrained)
    # Verify an actual cache view without reading the outer-test rows.
    cache_probe: dict[str, Any] = {"status": "not_checked"}
    try:
        import pandas as pd
        frame = pd.read_parquet(args.windows, columns=["relative_path"])
        if frame.empty:
            raise RuntimeError("windows manifest is empty")
        path = args.cache_root / Path(str(frame.iloc[0, 0])).with_suffix(".npy")
        view = np.load(path, mmap_mode="r", allow_pickle=False)
        if view.ndim != 2 or view.shape[0] != 16:
            raise ValueError(f"unexpected cache shape {view.shape}")
        cache_probe = {"status": "passed", "path": str(path), "shape": list(view.shape), "dtype": str(view.dtype)}
    except Exception as exc:
        cache_probe = {"status": "failed", "error": repr(exc)}
        issues.append("cache_probe_failed")

    payload = {
        "release_id": "meta-ttt-chbmit-5fold-v1",
        "status": "passed" if not issues else "blocked",
        "issues": issues,
        "active_processes": active,
        "target_namespace": str(root),
        "old_namespaces_preserved": [
            "/root/b_false_alarm_atlas/outputs/reports/neurottt-chbmit-5fold-v1",
            "/root/b_false_alarm_atlas/outputs/reports/neurottt-chbmit-5fold-v1-bandfix-v3",
        ],
        "folds": folds,
        "patient_count": len(counts),
        "outer_test_patient_count": len(test_counts),
        "source_hashes": {"windows": windows_hash, "cv_manifest": cv_hash, "pretrained": pretrained_hash},
        "cache_probe": cache_probe,
        "seizure_scoring": {
            "revision": "event_collar_30s_60s_raw_alarm_time_v1",
            "truth_definition": "raw interval expanded to onset-30s and offset+60s for event matching, clipped to evaluable EDF",
            "onset_pre_s": 30.0,
            "offset_post_s": 60.0,
            "alarm_time_truth_definition": "raw seizure interval; collar excluded from alarm-time subtraction",
            "detection_delay_reference": "raw onset, clipped to evaluation start",
            "training_labels_modified": False,
        },
        "test_partition_read": False,
        "test_evaluation_count": 0,
        "created_at": utc_now(),
    }
    atomic_json(root / "preflight.json", payload)
    if issues:
        raise SystemExit(3)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/meta-ttt-chbmit-5fold-v1')
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parent / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
