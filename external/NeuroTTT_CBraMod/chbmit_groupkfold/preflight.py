from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, fold_manifest

EXPECTED_COMMIT = "798d27d6eaf39adf2dd544a88697d64e065da165"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def active_experiments() -> list[dict[str, str]]:
    matches = []
    current = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == current:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "chbmit_groupkfold_train.py" in command or "chbmit_groupkfold_evaluate.py" in command:
            matches.append({"pid": entry.name, "command": command})
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/neurottt-chbmit-5fold-v1')
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--smoke-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/neurottt-chbmit-5fold-v1-smoke-v1')
    parser.add_argument("--mask-smoke-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/neurottt-chbmit-5fold-v1-smoke-v4')
    args = parser.parse_args()
    if (args.output_root / "runs").exists() and any((args.output_root / "runs").iterdir()):
        raise FileExistsError(f"formal run directory is not empty: {args.output_root / 'runs'}")
    active = active_experiments()
    if active:
        raise RuntimeError(f"duplicate active experiment: {active}")
    git = ["git", "-c", "core.filemode=false", "-c", "core.autocrlf=true", "-C", str(args.repo_root)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"unexpected upstream commit: {commit}")
    git_status = subprocess.check_output(git + ["status", "--porcelain"], text=True).splitlines()
    folds = [fold_manifest(fold, args.fold_root) for fold in range(5)]
    outer_tests = [patient for payload in folds for patient in payload["test"]]
    if len(outer_tests) != 22 or len(set(outer_tests)) != 22:
        raise RuntimeError("outer test folds do not cover each canonical patient exactly once")
    smoke_files = {
        "detection_only": args.smoke_root / "runs/detection_only_fold0_seed3407/completed.json",
        "band_joint": args.smoke_root / "runs/band_joint_fold0_seed3407/completed.json",
        "mask_joint": args.mask_smoke_root / "runs/mask_joint_fold0_seed3407/completed.json",
    }
    smoke = {}
    for condition, path in smoke_files.items():
        payload = json.loads(path.read_text())
        if not payload.get("gradient_gate_passed") or payload.get("test_evaluation_count") != 0:
            raise RuntimeError(f"failed smoke gate: {condition}: {payload}")
        smoke[condition] = {"path": str(path), "sha256": sha256(path), "best_validation_auprc": payload["best_validation_auprc"], "gradient_ratio_median": payload["gradient_ratio_median"]}
    free_root = shutil.disk_usage("/").free
    if free_root < 100 * 2**30:
        raise RuntimeError(f"less than 100 GiB free on WSL root: {free_root}")
    cache_examples = []
    for patient in sorted({patient for payload in folds for key in ("train", "validation", "test") for patient in payload[key]})[:3]:
        case = "chb01" if patient == "chb01_21" else patient
        files = sorted((args.cache_root / case).glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"cache missing for {patient}")
        cache_examples.append(str(files[0]))
    code_root = Path(__file__).resolve().parent
    code_hashes = {path.name: sha256(path) for path in sorted(code_root.glob("*.py"))}
    manifest = {
        "release_id": "neurottt-chbmit-5fold-v1",
        "status": "preflight_passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": commit,
        "git_status_at_freeze": git_status,
        "formal_conditions": ["supervised_frozen", "band_joint_frozen", "band_joint_band_ttt", "mask_joint_frozen", "mask_joint_mask_ttt"],
        "source_training_conditions": ["detection_only", "band_joint", "mask_joint"],
        "folds": 5,
        "seed": 3407,
        "source_runs": 15,
        "test_evaluation_count_per_condition_fold": 1,
        "test_used_for_training_or_selection": False,
        "patient_grouping": "canonical patient; outer GroupKFold(5); inner patient GroupShuffleSplit",
        "source_hashes": {"windows": sha256(args.windows), "cv_manifest": sha256(args.fold_root / "cv_manifest.json")},
        "code_hashes": code_hashes,
        "smoke_audit": smoke,
        "cache_unit": "existing CBraMod cache is microvolts/100; no second scaling",
        "cache_examples": cache_examples,
        "free_root_bytes": free_root,
        "active_duplicate_processes": active,
        "performance_policy": {"physical_batch": 128, "effective_batch": 128, "source_parallelism": 3, "ttt_engine": "fast scalar exact; vmap rejected by parity/compatibility audit"},
        "gradient_gate_definition": "median weighted EMA auxiliary/detection gradient ratio in [0.5,2.0]; instantaneous batch ratio retained for diagnosis",
        "mask_gradient_protocol_revision": "cap raised from 10 to 100 before formal validation/test because train-only smoke showed 25-50x smaller reconstruction gradients and 64% cap saturation",
    }
    atomic_json(args.output_root / "protocol_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
