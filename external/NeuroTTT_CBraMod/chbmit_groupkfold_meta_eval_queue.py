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
CONDITIONS = ("supervised_frozen", "meta_band_frozen", "meta_band_ttt", "meta_temporal_frozen", "meta_temporal_ttt")
TTT_BATCH_SIZE = 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_ready(root: Path) -> bool:
    q = root / "queue_status.json"
    if not q.is_file():
        return False
    state = json.loads(q.read_text())
    if state.get("status") == "failed":
        raise RuntimeError("meta source queue failed")
    if state.get("status") != "complete" or len(state.get("completed", [])) != 15:
        return False
    for fold in range(5):
        for condition in ("detection_only", "meta_band", "meta_temporal"):
            run = root / "runs" / f"{condition}_fold{fold}_seed3407"
            done = json.loads((run / "completed.json").read_text())
            if done.get("test_partition_read") is not False or done.get("test_evaluation_count") != 0:
                raise ValueError(f"source run touched test: {run}")
            if not (run / "best.pt").is_file():
                raise FileNotFoundError(run / "best.pt")
    return True


def run_jobs(root: Path, jobs: list[tuple[str, int, str]], max_parallel: int, phase: str) -> None:
    logs = root / "evaluation_logs"; logs.mkdir(parents=True, exist_ok=True)
    pending = list(jobs); active: dict[int, tuple[tuple[str, int, str], object, subprocess.Popen]] = {}
    completed: list[dict] = []; failed: list[dict] = []
    while pending or active:
        while pending and not failed and len(active) < max_parallel:
            condition, fold, split = pending.pop(0)
            run = root / "evaluation" / condition / f"fold{fold}_seed3407"
            marker = run / ("validation_metrics.json" if split == "validation" else "test_completed.json")
            if marker.is_file():
                completed.append({"condition": condition, "fold": fold, "split": split, "skipped_existing": True}); continue
            log_path = logs / f"{condition}_fold{fold}_{split}.log"; stream = log_path.open("ab", buffering=0)
            cmd = [str(PYTHON), str(CODE / "chbmit_groupkfold_meta_evaluate.py"), "--condition", condition, "--fold", str(fold), "--split", split, "--seed", "3407", "--output-root", str(root), "--workers", "8"]
            # Adapted inference keeps one independently updated parameter set per
            # sample.  A smaller batch than the frozen path is mathematically
            # identical but avoids exhausting the 32-GiB GPU on full partitions.
            cmd += ["--batch-size", str(TTT_BATCH_SIZE if "ttt" in condition else 256)]
            if split == "test": cmd.append("--allow-test")
            env = os.environ.copy(); env.update({"PYTHONPATH": str(CODE), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            process = subprocess.Popen(cmd, cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[process.pid] = ((condition, fold, split), stream, process)
        for pid, (job, stream, process) in list(active.items()):
            code = process.poll()
            if code is None: continue
            stream.close(); entry = {"condition": job[0], "fold": job[1], "split": job[2], "return_code": code, "finished_at": utc_now()}
            (completed if code == 0 else failed).append(entry); del active[pid]
        atomic_json(root / "evaluation_queue_status.json", {
            "status": "failed" if failed else ("phase_complete" if not pending and not active else "running"),
            "phase": phase, "completed": completed, "failed": failed,
            "running": [{"pid": pid, "condition": job[0], "fold": job[1], "split": job[2]} for pid, (job, _, _) in active.items()],
            "pending": [{"condition": job[0], "fold": job[1], "split": job[2]} for job in pending], "updated_at": utc_now(),
        })
        if active: time.sleep(10)
        elif failed: break
    if failed: raise RuntimeError(f"evaluation failed: {failed}")


def audit_validation(root: Path) -> None:
    for condition in CONDITIONS:
        for fold in range(5):
            p = root / "evaluation" / condition / f"fold{fold}_seed3407" / "validation_metrics.json"
            d = json.loads(p.read_text())
            if d.get("status") != "validation_threshold_frozen" or d.get("test_evaluation_count") != 0 or d.get("threshold_source") != "validation_only":
                raise ValueError(f"invalid validation lock: {p}")
    atomic_json(root / "validation_lock_audit.json", {"status": "passed", "fold_conditions": 25, "test_evaluation_count": 0, "threshold_source": "validation_only", "completed_at": utc_now()})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/meta-ttt-chbmit-5fold-v1'); args = parser.parse_args(); root = args.output_root
    while not source_ready(root):
        atomic_json(root / "evaluation_queue_status.json", {"status": "waiting_for_source", "phase": "source_training", "updated_at": utc_now()}); time.sleep(60)
    frozen = [c for c in CONDITIONS if "ttt" not in c]; adaptive = [c for c in CONDITIONS if "ttt" in c]
    run_jobs(root, [(condition, fold, "validation") for fold in range(5) for condition in frozen], 3, "validation_frozen")
    run_jobs(root, [(condition, fold, "validation") for fold in range(5) for condition in adaptive], 2, "validation_ttt")
    audit_validation(root)
    run_jobs(root, [(condition, fold, "test") for fold in range(5) for condition in frozen], 3, "test_frozen")
    run_jobs(root, [(condition, fold, "test") for fold in range(5) for condition in adaptive], 2, "test_ttt")
    subprocess.run([str(PYTHON), str(CODE / "chbmit_groupkfold_meta_summarize.py"), "--output-root", str(root)], cwd=CODE, check=True)
    atomic_json(root / "pipeline_completed.json", {"status": "complete", "source_runs": 15, "validation_fold_conditions": 25, "test_fold_conditions": 25, "bootstrap_replicates": 2000, "completed_at": utc_now()})


if __name__ == "__main__": main()
