from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
PYTHON = Path("/root/b_false_alarm_atlas/.venv/bin/python")
CONDITIONS = ("detection_only", "band_joint", "mask_joint")
MAX_PARALLEL = 3


def atomic_status(root: Path, payload: dict) -> None:
    path = root / "queue_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/root/b_false_alarm_atlas/outputs/reports/neurottt-chbmit-5fold-v1"))
    parser.add_argument("--band-aux-weight-max", type=float, default=10.0)
    parser.add_argument("--smoke-root", type=Path, default=Path("/root/b_false_alarm_atlas/outputs/reports/neurottt-chbmit-5fold-v1-smoke-v1"))
    parser.add_argument("--mask-smoke-root", type=Path, default=Path("/root/b_false_alarm_atlas/outputs/reports/neurottt-chbmit-5fold-v1-smoke-v4"))
    args = parser.parse_args()
    root = args.output_root
    subprocess.run([
        str(PYTHON), str(CODE / "chbmit_groupkfold_preflight.py"),
        "--output-root", str(root), "--smoke-root", str(args.smoke_root),
        "--mask-smoke-root", str(args.mask_smoke_root),
    ], cwd=CODE, check=True)
    jobs = [(fold, condition) for fold in range(5) for condition in CONDITIONS]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[int, str, object, subprocess.Popen]] = {}
    completed: list[dict] = []
    failed: list[dict] = []
    next_job = 0
    stop_launching = False
    while next_job < len(jobs) or running:
        while not stop_launching and next_job < len(jobs) and len(running) < MAX_PARALLEL:
            fold, condition = jobs[next_job]
            next_job += 1
            log_path = logs / f"{condition}_fold{fold}_seed3407.log"
            stream = log_path.open("ab", buffering=0)
            command = [
                str(PYTHON), str(CODE / "chbmit_groupkfold_train.py"),
                "--condition", condition,
                "--fold", str(fold),
                "--seed", "3407",
                "--output-root", str(root),
                "--batch-size", "128",
                "--effective-batch", "128",
                "--eval-batch-size", "256",
                "--workers", "8",
                "--epochs", "50",
                "--minimum-epochs", "5",
                "--patience", "7",
                "--gradient-warmup", "100",
                "--gradient-interval", "50",
            ]
            if condition == "band_joint":
                command += ["--aux-weight-max", str(args.band_aux_weight_max)]
            environment = os.environ.copy()
            environment.update({"PYTHONPATH": str(CODE), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
            process = subprocess.Popen(command, cwd=CODE, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            running[process.pid] = (fold, condition, stream, process)
        for pid, (fold, condition, stream, process) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            entry = {"fold": fold, "condition": condition, "return_code": return_code, "finished_at": datetime.now(timezone.utc).isoformat()}
            (completed if return_code == 0 else failed).append(entry)
            if return_code != 0:
                stop_launching = True
            del running[pid]
        atomic_status(root, {
            "status": "failed" if failed else ("complete" if len(completed) == len(jobs) else "running"),
            "completed": completed,
            "failed": failed,
            "running": [{"pid": pid, "fold": fold, "condition": condition} for pid, (fold, condition, _, _) in running.items()],
            "queued_remaining": len(jobs) - next_job,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        if running:
            time.sleep(5)
        elif stop_launching:
            break
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
