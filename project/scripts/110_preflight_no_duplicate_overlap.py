"""Preflight guard against duplicate or silently overlapping experiments.

The guard distinguishes an exact duplicate (same model, seed and patient split)
from an intentional protocol comparison (same CHB patients but a different
split/protocol).  Exact duplicates and checkpoint reuse are hard failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
RUNS = ROOT / "runs"
BASELINES = RUNS / "baselines"
MODELS = ("eegnet", "psd_catboost", "deepconvnet", "shallowconvnet")
FOLDS = range(5)
SEEDS = (17, 42, 3407)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def split_signature(path: Path) -> str:
    data = json.loads(path.read_text())
    canonical = json.dumps({k: sorted(data.get(k, [])) for k in ("train", "validation", "test")}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_run_records() -> list[dict[str, object]]:
    records = []
    for manifest in BASELINES.glob("*/**/run_manifest.json"):
        try:
            obj = json.loads(manifest.read_text())
        except Exception:
            continue
        split = Path(str(obj.get("split_manifest", "")))
        if not split.is_absolute(): split = ROOT / split
        if not split.exists(): continue
        records.append({
            "namespace": manifest.parts[-4] if len(manifest.parts) >= 4 else "unknown",
            "path": str(manifest.relative_to(ROOT)).replace("\\", "/"),
            "model": obj.get("model"), "seed": int(obj.get("model_seed", obj.get("seed", -1))),
            "split_manifest": str(split.relative_to(ROOT)).replace("\\", "/"),
            "split_sha256": sha256(split), "split_signature": split_signature(split),
            "completed": (manifest.parent / "completed.json").exists(),
            "checkpoint_hashes": [sha256(p) for p in (manifest.parent / "checkpoints").glob("*.pt")],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="v3-groupkfold-baselines-v1")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs" / "reports" / args.namespace
    target_records = load_run_records()
    duplicates, overlaps, reused = [], [], []
    target = [r for r in target_records if r["namespace"] == args.namespace]
    old = [r for r in target_records if r["namespace"] != args.namespace]
    for cur in target:
        for prior in old:
            if (cur["model"], cur["seed"], cur["split_signature"]) == (prior["model"], prior["seed"], prior["split_signature"]):
                if str(prior["namespace"]).endswith("-smoke"):
                    overlaps.append({"current": cur, "prior": prior, "kind": "smoke_same_split_not_formal_result"})
                else:
                    duplicates.append({"current": cur, "prior": prior})
            elif cur["model"] == prior["model"] and cur["seed"] == prior["seed"]:
                overlaps.append({"current": cur, "prior": prior, "kind": "same_model_seed_different_patient_split"})
            if set(cur["checkpoint_hashes"]) & set(prior["checkpoint_hashes"]):
                reused.append({"current": cur, "prior": prior, "kind": "checkpoint_hash_reuse"})

    # The current queue may have one in-progress record; multiple identical
    # command lines are a duplicate execution even if no completed file exists.
    active_count = 0
    active_jobs = []
    try:
        import subprocess
        lines = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True).splitlines()
        commands = {}
        for line in lines:
            bits = line.strip().split(maxsplit=2)
            if len(bits) == 3: commands[int(bits[0])] = (int(bits[1]), bits[2])
        for pid, (ppid, command) in commands.items():
            parent_command = commands.get(ppid, (0, ""))[1]
            if "run_standard_groupkfold_baselines_v1.sh" in parent_command and ("61_run_baseline.py" in command or "64_run_conv_baseline.py" in command):
                active_jobs.append({"pid": pid, "ppid": ppid, "command": command})
        active_count = len(active_jobs)
    except Exception:
        active_count = -1
    report = {
        "analysis_id": "no-duplicate-overlap-preflight-v1",
        "namespace": args.namespace,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_records": len(target), "historical_records_scanned": len(old),
        "exact_duplicates": duplicates, "protocol_overlaps": overlaps,
        "checkpoint_reuse": reused, "active_target_python_processes": active_count,
        "active_jobs": active_jobs,
        "status": "FAIL" if duplicates or reused or active_count > 1 else "PASS",
        "interpretation": "same CHB patients with different split/protocol is recorded as overlap, not merged evidence",
    }
    out.mkdir(parents=True, exist_ok=True)
    if args.write_report:
        (out / "no_duplicate_overlap_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "FAIL": raise SystemExit(2)


if __name__ == "__main__": main()
