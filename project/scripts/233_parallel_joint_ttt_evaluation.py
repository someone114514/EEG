"""Run independent Joint-TTT fold/seed evaluations concurrently on one GPU."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = os.environ.get("JOINT_TTT_NAMESPACE", "cbramod-joint-ttt-v1-formal")
OUT = ROOT / "outputs/reports" / NAMESPACE / "evaluation"
PROGRESS = OUT / "queue_progress.json"
QUEUE_LOG = OUT / "queue.log"
WORKERS = max(1, int(os.environ.get("JOINT_TTT_EVAL_WORKERS", "4")))
UNITS = [(fold, seed) for fold in range(5) for seed in (17, 42, 3407)]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_progress(status: str, completed: int, active: list[str], error: str | None = None) -> None:
    payload = {
        "status": status,
        "completed": completed,
        "current": active[0] if active else None,
        "active": active,
        "workers": WORKERS,
        "total": len(UNITS),
    }
    if error is not None:
        payload["error"] = error
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, PROGRESS)


def log(message: str) -> None:
    with QUEUE_LOG.open("a") as stream:
        stream.write(f"[{utc()}] {message}\n")


def valid_completed(fold: int, seed: int) -> bool:
    manifest_path = OUT / f"fold{fold}_seed{seed}" / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return (
        int(manifest.get("fold", -1)) == fold
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("namespace") == NAMESPACE
        and manifest.get("threshold_source") == "validation-only median of patient validation selections"
        and int(manifest.get("test_evaluation_count_per_condition", -1)) == 1
        and manifest.get("test_labels_used_for_adaptation") is False
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[int, int]] = []
    completed = 0
    for fold, seed in UNITS:
        checkpoint = ROOT / "outputs/reports" / NAMESPACE / "runs" / f"fold{fold}_seed{seed}" / "checkpoint.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if valid_completed(fold, seed):
            completed += 1
            log(f"resume-skip audited complete fold={fold} seed={seed}")
        else:
            run_dir = OUT / f"fold{fold}_seed{seed}"
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"partial output exists; refusing overwrite: {run_dir}")
            pending.append((fold, seed))

    running: dict[subprocess.Popen, tuple[int, int, object]] = {}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTHONUNBUFFERED"] = "1"
    log(f"parallel start namespace={NAMESPACE} workers={WORKERS} pending={len(pending)} completed={completed}")

    try:
        while pending or running:
            while pending and len(running) < WORKERS:
                fold, seed = pending.pop(0)
                log_path = OUT / f"fold{fold}_seed{seed}.log"
                stream = log_path.open("w")
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/214_evaluate_joint_ttt.py",
                    "--fold", str(fold), "--seed", str(seed),
                    "--device", os.environ.get("JOINT_TTT_EVAL_DEVICE", "cuda"),
                    "--update-after-score",
                ]
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                running[process] = (fold, seed, stream)
                log(f"begin fold={fold} seed={seed} pid={process.pid}")
            active = [f"fold{fold}_seed{seed}" for fold, seed, _ in running.values()]
            write_progress("running", completed, active)
            time.sleep(1)
            for process, (fold, seed, stream) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                stream.close()
                del running[process]
                if return_code != 0:
                    error = f"fold={fold} seed={seed} exited {return_code}"
                    log(error)
                    for other, (_, _, other_stream) in running.items():
                        other.terminate(); other_stream.close()
                    write_progress("failed", completed, [], error)
                    raise SystemExit(return_code)
                if not valid_completed(fold, seed):
                    raise RuntimeError(f"completed process failed manifest audit: fold={fold} seed={seed}")
                completed += 1
                log(f"complete fold={fold} seed={seed}")
        write_progress("complete", completed, [])
        log("parallel queue complete")
    except BaseException:
        for process, (_, _, stream) in running.items():
            if process.poll() is None:
                process.terminate()
            stream.close()
        raise


if __name__ == "__main__":
    main()
