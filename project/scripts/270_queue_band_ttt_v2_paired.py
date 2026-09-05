"""Resumable paired validation/test queue for frozen Band-TTT v2.

Each (configuration, fold) validation lock is followed by its test job as
soon as GPU capacity permits.  Fold-0 test is retained for every configuration.
Fold-1 test may be stopped by a predeclared futility rule based only on the
already completed fold-0 test; every decision is written to an audit file.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
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
    parser.add_argument("--window-batch-size", type=int, default=128)
    parser.add_argument("--stream-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--disable-futility-pruning", action="store_true")
    return parser.parse_args()


def gpu_slots(job: dict[str, Any]) -> int:
    config = str(job["config"])
    if config.startswith("patient_") and not config.startswith("patient_chunk"):
        return 1
    return 2


def result_dir(config: str, fold: int) -> Path:
    return OUT / "evaluation" / config / f"fold{fold}_seed3407"


def completed(config: str, fold: int, split: str) -> bool:
    return (result_dir(config, fold) / f"{split}_completed.json").is_file()


def skipped(config: str, fold: int) -> bool:
    return (result_dir(config, fold) / "test_skipped.json").is_file()


def discover_external_jobs(owned_pids: set[int]) -> list[dict[str, Any]]:
    """Adopt evaluator processes launched by an interrupted predecessor queue."""
    jobs = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in owned_pids:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "265_evaluate_band_ttt_v2.py" not in command:
            continue
        tokens = shlex.split(command)
        try:
            config = tokens[tokens.index("--config") + 1]
            fold = int(tokens[tokens.index("--fold") + 1])
            split = tokens[tokens.index("--split") + 1]
        except (ValueError, IndexError):
            continue
        jobs.append({"config": config, "fold": fold, "split": split, "adopted_pid": int(entry.name)})
    return jobs


def fold0_futility(config: str) -> dict[str, Any] | None:
    """Return an auditable fold-1 stop decision for a clearly dominated fold-0 test."""
    path = result_dir(config, 0) / "test_completed.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    ttt = payload.get("selected_event_operating_point", {})
    frozen = payload.get("existing_frozen_baseline", {})
    required = ("false_alarm_time_min_per_24h", "event_sensitivity")
    if any(key not in ttt or key not in frozen for key in required):
        return None
    ttt_fa = float(ttt[required[0]])
    frozen_fa = float(frozen[required[0]])
    ttt_sens = float(ttt[required[1]])
    frozen_sens = float(frozen[required[1]])
    fa_ratio = ttt_fa / frozen_fa if frozen_fa > 0 else float("inf")
    sensitivity_delta = ttt_sens - frozen_sens
    reasons = []
    if fa_ratio >= 1.10 and sensitivity_delta <= 0.0:
        reasons.append("false-alarm minutes worsened by >=10% without sensitivity gain")
    if sensitivity_delta <= -0.02 and fa_ratio >= 0.95:
        reasons.append("sensitivity fell by >=2 percentage points without >=5% false-alarm-time gain")
    if fa_ratio >= 1.25:
        reasons.append("false-alarm minutes worsened by >=25%")
    if not reasons:
        return None
    return {
        "status": "test_skipped_by_sequential_futility_rule",
        "config": config,
        "fold": 1,
        "basis_fold": 0,
        "basis_test_path": str(path),
        "ttt_false_alarm_time_min_per_24h": ttt_fa,
        "frozen_false_alarm_time_min_per_24h": frozen_fa,
        "false_alarm_time_ratio": fa_ratio,
        "ttt_event_sensitivity": ttt_sens,
        "frozen_event_sensitivity": frozen_sens,
        "sensitivity_delta": sensitivity_delta,
        "reasons": reasons,
        "interpretation": "exploratory sequential stopping; fold-1 test is missing by design and must not be treated as a complete confirmatory matrix",
        "created_utc": now(),
    }


def apply_futility_rules(configs: list[str], enabled: bool) -> list[dict[str, Any]]:
    decisions = []
    if not enabled:
        return decisions
    for config in configs:
        marker = result_dir(config, 1) / "test_skipped.json"
        if marker.is_file():
            decisions.append(json.loads(marker.read_text()))
            continue
        if completed(config, 1, "test"):
            continue
        decision = fold0_futility(config)
        if decision is not None:
            marker.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(marker, decision)
            decisions.append(decision)
    return decisions


def next_jobs(configs: list[str], running_jobs: list[dict[str, Any]], pruning: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = apply_futility_rules(configs, pruning)
    active = {(job["config"], job["fold"], job["split"]) for job in running_jobs}
    ready_tests = []
    ready_validations = []
    for config in configs:
        for fold in (0, 1):
            if not completed(config, fold, "validation"):
                job = {"config": config, "fold": fold, "split": "validation"}
                if (config, fold, "validation") not in active:
                    ready_validations.append(job)
                continue
            if completed(config, fold, "test") or skipped(config, fold):
                continue
            # Fold 1 test waits for fold 0 test, allowing the declared stop rule.
            if fold == 1 and not (completed(config, 0, "test") or skipped(config, 0)):
                continue
            job = {"config": config, "fold": fold, "split": "test"}
            if (config, fold, "test") not in active:
                ready_tests.append(job)
    # A newly locked test always precedes unrelated validation work.
    return ready_tests + ready_validations, decisions


def main() -> None:
    args = parse_args()
    if not 1 <= args.parallel_workers <= 4:
        raise ValueError("parallel-workers must be 1..4")
    manifest = json.loads((OUT / "frozen_manifest.json").read_text())
    configs = [row["config_id"] for row in manifest["configurations"]]
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "268_import_existing_band_ttt_v2.py")], cwd=ROOT, check=True)
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "queue_status.json"
    lock = OUT / "queue.lock"
    if lock.exists():
        raise FileExistsError(f"queue lock exists: {lock}")
    atomic_json(lock, {"pid": os.getpid(), "mode": "paired", "created_utc": now()})
    running: dict[subprocess.Popen, tuple[dict[str, Any], Any]] = {}
    failures: list[dict[str, Any]] = []
    try:
        while True:
            external_jobs = discover_external_jobs({process.pid for process in running})
            running_jobs = [job for job, _ in running.values()] + external_jobs
            candidates, decisions = next_jobs(configs, running_jobs, not args.disable_futility_pruning)
            dispatched = True
            while len(running) < args.parallel_workers and candidates and dispatched:
                dispatched = False
                for index, job in enumerate(candidates):
                    external_jobs = discover_external_jobs({process.pid for process in running})
                    slots_used = sum(gpu_slots(active) for active, _ in running.values()) + sum(gpu_slots(active) for active in external_jobs)
                    if slots_used + gpu_slots(job) > args.gpu_slots:
                        continue
                    directory = result_dir(job["config"], job["fold"])
                    partial = directory / f"{job['split']}_probabilities.parquet"
                    if partial.exists():
                        failures.append({**job, "reason": f"partial output requires audit: {partial}"})
                        candidates.pop(index)
                        dispatched = True
                        break
                    log_path = logs / f"{job['config']}_fold{job['fold']}_{job['split']}.log"
                    stream = log_path.open("a", buffering=1)
                    command = [str(PYTHON), str(EVALUATOR), "--config", job["config"], "--fold", str(job["fold"]), "--split", job["split"], "--window-batch-size", str(args.window_batch_size), "--stream-batch-size", str(args.stream_batch_size), "--workers", str(args.workers)]
                    if job["split"] == "test":
                        command.append("--allow-test")
                    environment = os.environ.copy()
                    environment["PYTHONPATH"] = str(ROOT / "src")
                    environment["PYTHONUNBUFFERED"] = "1"
                    stream.write(f"[{now()}] START {' '.join(command)}\n")
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    running[process] = (job, stream)
                    candidates.pop(index)
                    dispatched = True
                    break
            done_count = sum(completed(c, f, s) for c in configs for f in (0, 1) for s in ("validation", "test"))
            skip_count = sum(skipped(c, f) for c in configs for f in (0, 1))
            external_jobs = discover_external_jobs({process.pid for process in running})
            all_running_jobs = [job for job, _ in running.values()] + external_jobs
            remaining, decisions = next_jobs(configs, all_running_jobs, not args.disable_futility_pruning)
            atomic_json(status_path, {
                "status": "running" if running or remaining else ("complete" if not failures else "complete_with_failures"),
                "mode": "paired_validation_then_test",
                "total_planned": 64,
                "completed": done_count,
                "skipped_tests": skip_count,
                "pending_ready": remaining,
                "running": all_running_jobs,
                "gpu_slots_used": sum(gpu_slots(job) for job in all_running_jobs),
                "gpu_slots_total": args.gpu_slots,
                "futility_pruning_enabled": not args.disable_futility_pruning,
                "futility_decisions": decisions,
                "failures": failures,
                "updated_utc": now(),
            })
            if not running and not external_jobs and not remaining:
                break
            time.sleep(5)
            for process in list(running):
                code = process.poll()
                if code is None:
                    continue
                job, stream = running.pop(process)
                stream.write(f"[{now()}] EXIT {code}\n")
                stream.close()
                if code != 0:
                    failures.append({**job, "exit_code": code})
        subprocess.run([str(PYTHON), str(ROOT / "scripts" / "266_summarize_band_ttt_v2.py")], cwd=ROOT, check=False)
    finally:
        released = OUT / "queue.lock.released"
        if released.exists():
            released.unlink()
        if lock.exists():
            os.replace(lock, released)


if __name__ == "__main__":
    main()
