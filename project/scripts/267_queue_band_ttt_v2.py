"""Resumable single-phase queue for frozen Band-TTT v2 evaluation.

Run validation and test as separate phases.  The test phase refuses to start
until every validation result exists, so no test job can precede threshold
locking while the complete two-phase pipeline still covers all 64 jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT = ROOT / "outputs" / "reports" / "band-ttt-v2-fold01"
PYTHON = ROOT / ".venv" / "bin" / "python"
EVALUATOR = ROOT / "scripts" / "265_evaluate_band_ttt_v2.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--gpu-slots", type=int, default=4)
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--window-batch-size", type=int, default=128)
    parser.add_argument("--stream-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def gpu_slots(job: dict[str, Any]) -> int:
    """Approximate GPU pressure for config-aware concurrent scheduling."""
    config = str(job["config"])
    # Patient and Patient-Adam have only 4 validation or 5 test streams, so a
    # single process cannot fill the GPU.  Window/Record vectorize 128 states;
    # Chunk uses ordinary batch-30 backward and is also treated as heavyweight.
    if config.startswith("patient_") and not config.startswith("patient_chunk"):
        return 1
    return 2


def main() -> None:
    args = parse_args()
    if args.parallel_workers < 1 or args.parallel_workers > 4:
        raise ValueError("parallel-workers must be 1..4 on the 32-GiB GPU")
    if args.gpu_slots < 1 or args.gpu_slots > 8:
        raise ValueError("gpu-slots must be 1..8")
    manifest = json.loads((OUT / "frozen_manifest.json").read_text())
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "268_import_existing_band_ttt_v2.py")], cwd=ROOT, check=True)
    if args.phase == "test":
        missing_validation = [
            OUT / "evaluation" / config["config_id"] / f"fold{fold}_seed3407" / "validation_completed.json"
            for fold in (0, 1)
            for config in manifest["configurations"]
            if not (OUT / "evaluation" / config["config_id"] / f"fold{fold}_seed3407" / "validation_completed.json").is_file()
        ]
        if missing_validation:
            raise RuntimeError(f"test phase blocked by {len(missing_validation)} missing validation locks")
    jobs = [
        {"config": config["config_id"], "fold": fold, "split": args.phase}
        for fold in (0, 1)
        for config in manifest["configurations"]
    ]
    if len(jobs) != 32:
        raise AssertionError(len(jobs))
    total_jobs = len(jobs)
    logs = OUT / "logs"; logs.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "queue_status.json"
    lock = OUT / "queue.lock"
    if lock.exists():
        raise FileExistsError(f"queue lock exists: {lock}")
    atomic_json(lock, {"pid": os.getpid(), "created_utc": now()})
    running: dict[subprocess.Popen, tuple[dict[str, Any], Any]] = {}
    failures = []
    try:
        pending = []
        for job in jobs:
            completed = OUT / "evaluation" / job["config"] / f"fold{job['fold']}_seed3407" / f"{job['split']}_completed.json"
            if not completed.is_file():
                pending.append(job)
        while pending or running:
            dispatched = True
            while len(running) < args.parallel_workers and pending and dispatched:
                dispatched = False
                for index, job in enumerate(pending):
                    slots_used = sum(gpu_slots(active_job) for active_job, _ in running.values())
                    if slots_used + gpu_slots(job) > args.gpu_slots:
                        continue
                    result_dir = OUT / "evaluation" / job["config"] / f"fold{job['fold']}_seed3407"
                    if job["split"] == "test" and not (result_dir / "validation_completed.json").is_file():
                        continue
                    partial = result_dir / f"{job['split']}_probabilities.parquet"
                    if partial.exists():
                        failures.append({**job, "reason": f"partial output requires audit before retry: {partial}"})
                        pending.pop(index); dispatched = True; break
                    log_path = logs / f"{job['config']}_fold{job['fold']}_{job['split']}.log"
                    stream = log_path.open("a", buffering=1)
                    command = [str(PYTHON), str(EVALUATOR), "--config", job["config"], "--fold", str(job["fold"]), "--split", job["split"], "--window-batch-size", str(args.window_batch_size), "--stream-batch-size", str(args.stream_batch_size), "--workers", str(args.workers)]
                    if job["split"] == "test":
                        command.append("--allow-test")
                    environment = os.environ.copy(); environment["PYTHONPATH"] = str(ROOT / "src"); environment["PYTHONUNBUFFERED"] = "1"
                    stream.write(f"[{now()}] START {' '.join(command)}\n")
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    running[process] = (job, stream); pending.pop(index); dispatched = True; break
            atomic_json(status_path, {"status": "running", "phase": args.phase, "total": total_jobs, "completed": total_jobs - len(pending) - len(running) - len(failures), "pending": pending, "running": [job for job, _ in running.values()], "gpu_slots_used": sum(gpu_slots(job) for job, _ in running.values()), "gpu_slots_total": args.gpu_slots, "failures": failures, "updated_utc": now()})
            if not running and pending:
                failures.extend({**job, "reason": "validation dependency was not completed"} for job in pending)
                pending.clear(); break
            time.sleep(5)
            for process in list(running):
                code = process.poll()
                if code is None:
                    continue
                job, stream = running.pop(process)
                stream.write(f"[{now()}] EXIT {code}\n"); stream.close()
                if code != 0:
                    failures.append({**job, "exit_code": code})
        status = "complete" if not failures else "complete_with_failures"
        atomic_json(status_path, {"status": status, "phase": args.phase, "total": total_jobs, "completed": total_jobs - len(failures), "failures": failures, "updated_utc": now()})
        subprocess.run([str(PYTHON), str(ROOT / "scripts" / "266_summarize_band_ttt_v2.py")], cwd=ROOT, check=False)
    finally:
        released = OUT / "queue.lock.released"
        if released.exists():
            released.unlink()
        os.replace(lock, released)


if __name__ == "__main__":
    main()
