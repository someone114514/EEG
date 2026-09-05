from __future__ import annotations

import json
import os
import subprocess
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path


CODE = Path(__file__).resolve().parent
PYTHON = Path("/root/b_false_alarm_atlas/.venv/bin/python")
CONDITIONS = (
    "supervised_frozen",
    "band_joint_frozen",
    "band_joint_band_ttt",
    "mask_joint_frozen",
    "mask_joint_mask_ttt",
)
SOURCE_CONDITIONS = ("detection_only", "band_joint", "mask_joint")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_ready(root: Path) -> bool:
    queue = root / "queue_status.json"
    if not queue.is_file():
        return False
    state = json.loads(queue.read_text())
    if state.get("status") == "failed":
        raise RuntimeError("source queue failed")
    if state.get("status") != "complete" or len(state.get("completed", [])) != 15:
        return False
    for fold in range(5):
        for condition in SOURCE_CONDITIONS:
            run = root / "runs" / f"{condition}_fold{fold}_seed3407"
            completed = json.loads((run / "completed.json").read_text())
            if completed.get("test_partition_read") is not False or completed.get("test_evaluation_count") != 0:
                raise ValueError(f"source run touched test: {run}")
            if not completed.get("gradient_gate_passed", False):
                raise ValueError(f"gradient gate failed: {run}")
            if not (run / "best.pt").is_file():
                raise FileNotFoundError(run / "best.pt")
    return True


def run_jobs(root: Path, jobs: list[tuple[str, int, str]], max_parallel: int, phase: str) -> None:
    logs = root / "evaluation_logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active: dict[int, tuple[tuple[str, int, str], object, subprocess.Popen]] = {}
    completed: list[dict] = []
    failed: list[dict] = []
    while pending or active:
        while pending and not failed and len(active) < max_parallel:
            condition, fold, split = pending.pop(0)
            result = root / "evaluation" / condition / f"fold{fold}_seed3407"
            completion = result / ("validation_metrics.json" if split == "validation" else "test_completed.json")
            if completion.is_file():
                completed.append({"condition": condition, "fold": fold, "split": split, "skipped_existing": True})
                continue
            log_path = logs / f"{condition}_fold{fold}_{split}.log"
            stream = log_path.open("ab", buffering=0)
            command = [
                str(PYTHON), str(CODE / "chbmit_groupkfold_evaluate.py"),
                "--condition", condition, "--fold", str(fold), "--split", split,
                "--seed", "3407", "--output-root", str(root), "--workers", "8",
            ]
            if "ttt" in condition:
                command += ["--batch-size", "32", "--ttt-engine", "scalar"]
            else:
                command += ["--batch-size", "256"]
            if split == "test":
                command.append("--allow-test")
            environment = os.environ.copy()
            environment.update({"PYTHONPATH": str(CODE), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            process = subprocess.Popen(command, cwd=CODE, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            active[process.pid] = ((condition, fold, split), stream, process)
        for pid, (job, stream, process) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            entry = {"condition": job[0], "fold": job[1], "split": job[2], "return_code": code}
            (completed if code == 0 else failed).append(entry)
            del active[pid]
        atomic_json(root / "evaluation_queue_status.json", {
            "status": "failed" if failed else ("phase_complete" if not pending and not active else "running"),
            "phase": phase,
            "completed": completed,
            "failed": failed,
            "running": [{"pid": pid, "condition": job[0], "fold": job[1], "split": job[2]} for pid, (job, _, _) in active.items()],
            "pending": [{"condition": job[0], "fold": job[1], "split": job[2]} for job in pending],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        if active:
            time.sleep(10)
        elif failed:
            break
    if failed:
        raise RuntimeError(f"evaluation {phase} failed: {failed}")


def audit_validation(root: Path) -> None:
    for condition in CONDITIONS:
        for fold in range(5):
            path = root / "evaluation" / condition / f"fold{fold}_seed3407" / "validation_metrics.json"
            payload = json.loads(path.read_text())
            if payload.get("status") != "validation_threshold_frozen" or payload.get("test_evaluation_count") != 0:
                raise ValueError(f"invalid validation lock: {path}")
            selected = payload.get("selected_event_operating_point", {})
            if not 0 < float(selected.get("threshold", 0)) < 1:
                raise ValueError(f"invalid threshold: {path}")
    atomic_json(root / "validation_lock_audit.json", {
        "status": "passed", "fold_conditions": 25,
        "test_evaluation_count": 0, "threshold_source": "validation_only",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/root/b_false_alarm_atlas/outputs/reports/neurottt-chbmit-5fold-v1"))
    args = parser.parse_args()
    root = args.output_root
    while not source_ready(root):
        atomic_json(root / "evaluation_queue_status.json", {
            "status": "waiting_for_source", "phase": "source_training",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        time.sleep(60)
    frozen = [condition for condition in CONDITIONS if "ttt" not in condition]
    adaptive = [condition for condition in CONDITIONS if "ttt" in condition]
    run_jobs(root, [(condition, fold, "validation") for fold in range(5) for condition in frozen], 3, "validation_frozen")
    run_jobs(root, [(condition, fold, "validation") for condition in adaptive for fold in range(5)], 4, "validation_ttt")
    audit_validation(root)
    run_jobs(root, [(condition, fold, "test") for fold in range(5) for condition in frozen], 3, "test_frozen")
    run_jobs(root, [(condition, fold, "test") for condition in adaptive for fold in range(5)], 4, "test_ttt")
    subprocess.run([str(PYTHON), str(CODE / "chbmit_groupkfold_summarize.py"), "--output-root", str(root)], cwd=CODE, check=True)
    atomic_json(root / "pipeline_completed.json", {
        "status": "complete", "source_runs": 15, "validation_fold_conditions": 25,
        "test_fold_conditions": 25, "bootstrap_replicates": 2000,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
