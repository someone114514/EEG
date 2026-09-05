from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


CODE = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / '.venv/bin/python'
CONDITIONS = ("detection_only", "meta_band", "meta_temporal")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/meta-ttt-chbmit-5fold-v1')
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output_root
    preflight_command = [str(PYTHON), str(CODE / "chbmit_groupkfold_meta_preflight.py"), "--output-root", str(root)]
    if args.resume:
        preflight_command.append("--allow-existing")
    preflight = subprocess.run(
        preflight_command,
        cwd=CODE, env={**os.environ, "PYTHONPATH": str(CODE), "META_PREFLIGHT_IGNORE_PIDS": str(os.getpid())},
    )
    if preflight.returncode != 0:
        atomic_json(root / "queue_status.json", {"status": "blocked_preflight", "return_code": preflight.returncode, "updated_at": utc_now()})
        raise SystemExit(preflight.returncode)

    jobs = [(fold, condition) for fold in range(5) for condition in CONDITIONS]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[int, str, object, subprocess.Popen]] = {}
    completed: list[dict] = []
    # Allow a queue process to be safely resumed after the controlling shell or
    # desktop task is interrupted.  Completed source runs are immutable and
    # must count toward this queue without being launched again.
    existing_completed: set[tuple[int, str]] = set()
    for fold, condition in jobs:
        run = root / "runs" / f"{condition}_fold{fold}_seed3407"
        completed_path = run / "completed.json"
        if not completed_path.exists():
            continue
        payload = json.loads(completed_path.read_text())
        if payload.get("status") != "training_complete":
            raise RuntimeError(f"existing run is not complete: {completed_path}")
        existing_completed.add((fold, condition))
        completed.append({
            "fold": fold,
            "condition": condition,
            "return_code": 0,
            "finished_at": payload.get("completed_at", utc_now()),
            "resumed": True,
        })
    failed: list[dict] = []
    next_job = 0
    stop_launching = False
    while next_job < len(jobs) or running:
        while not stop_launching and next_job < len(jobs) and len(running) < args.max_parallel:
            fold, condition = jobs[next_job]
            next_job += 1
            run = root / "runs" / f"{condition}_fold{fold}_seed3407"
            if (fold, condition) in existing_completed:
                continue
            if (run / "completed.json").exists():
                raise FileExistsError(f"existing run would be overwritten: {run}")
            log_path = logs / f"{condition}_fold{fold}_seed3407.log"
            stream = log_path.open("ab", buffering=0)
            if condition == "detection_only":
                script = "chbmit_groupkfold_train.py"
                command = [str(PYTHON), str(CODE / script), "--condition", condition]
            else:
                script = "chbmit_groupkfold_meta_train.py"
                command = [str(PYTHON), str(CODE / script), "--condition", condition]
            command += [
                "--fold", str(fold), "--seed", "3407", "--output-root", str(root),
                "--batch-size", str(args.batch_size), "--effective-batch", str(args.effective_batch),
                "--eval-batch-size", str(args.eval_batch_size), "--workers", str(args.workers), "--epochs", str(args.epochs),
                "--minimum-epochs", "5", "--patience", "7", "--min-delta", "0.002",
            ]
            if args.max_updates is not None:
                command += ["--max-updates", str(args.max_updates)]
            if args.validation_limit is not None:
                command += ["--validation-limit", str(args.validation_limit)]
            environment = os.environ.copy()
            environment.update({"PYTHONPATH": str(CODE), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
            process = subprocess.Popen(command, cwd=CODE, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            running[process.pid] = (fold, condition, stream, process)
        for pid, (fold, condition, stream, process) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            entry = {"fold": fold, "condition": condition, "return_code": return_code, "finished_at": utc_now()}
            (completed if return_code == 0 else failed).append(entry)
            if return_code != 0:
                stop_launching = True
            del running[pid]
        atomic_json(root / "queue_status.json", {
            "release_id": "meta-ttt-chbmit-5fold-v1", "status": "failed" if failed else ("complete" if len(completed) == len(jobs) else "running"),
            "completed": completed, "failed": failed,
            "running": [{"pid": pid, "fold": fold, "condition": condition} for pid, (fold, condition, _, _) in running.items()],
            "queued_remaining": len(jobs) - next_job, "updated_at": utc_now(),
        })
        if running:
            time.sleep(5)
        elif stop_launching:
            break
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
