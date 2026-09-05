"""Freeze the prespecified Band-TTT v2 matrix and official CHB-MIT record order."""
from __future__ import annotations
import os

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
RELEASE = "band-ttt-v2-fold01"
OUT = ROOT / "outputs" / "reports" / RELEASE
SUMMARY_ROOT = Path(os.environ.get("BFA_RAW_ROOT", "/mnt/d/EEGData/chbmit-1.0.0"))
SOURCE_RELEASE = ROOT / "outputs" / "reports" / "meta-ttt-chbmit-5fold-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def official_record_order(recordings: pd.DataFrame) -> dict[str, object]:
    patients: dict[str, object] = {}
    for patient, group in recordings.groupby("patient_id", sort=True):
        source_cases = sorted(group.source_case_id.astype(str).unique())
        ordered: list[str] = []
        summaries: list[dict[str, str]] = []
        expected = set(group.recording_id.astype(str))
        for source_case in source_cases:
            summary = SUMMARY_ROOT / source_case / f"{source_case}-summary.txt"
            if not summary.is_file():
                raise FileNotFoundError(summary)
            names = re.findall(r"^File Name:\s*(\S+\.edf)\s*$", summary.read_text(errors="replace"), re.MULTILINE)
            selected = [name for name in names if name in expected]
            if len(selected) != len(set(selected)):
                raise RuntimeError(f"duplicate official record for {patient}: {source_case}")
            ordered.extend(selected)
            summaries.append({"source_case": source_case, "path": str(summary), "sha256": sha256(summary)})
        if set(ordered) != expected or len(ordered) != len(expected):
            missing = sorted(expected - set(ordered))
            extra = sorted(set(ordered) - expected)
            raise RuntimeError(f"official order mismatch patient={patient} missing={missing} extra={extra}")
        patients[str(patient)] = {
            "source_cases_in_order": source_cases,
            "recordings": ordered,
            "summary_sources": summaries,
        }
    return {"source": "official CHB-MIT patient summary File Name order; never filename sorting", "patients": patients}


def configurations() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for scope in ("window", "record", "patient"):
        for steps in (1, 3):
            for lr_strategy in ("global", "layerwise"):
                configs.append({"config_id": f"{scope}_k{steps}_{lr_strategy}_sgd", "accumulation_scope": scope, "steps": steps, "lr_strategy": lr_strategy, "optimizer": "sgd", "chunk_size_windows": None})
    for steps in (1, 3):
        configs.append({"config_id": f"patient_k{steps}_global_adam", "accumulation_scope": "patient", "steps": steps, "lr_strategy": "global", "optimizer": "adam", "chunk_size_windows": None})
    for steps in (1, 3):
        configs.append({"config_id": f"patient_chunk30_k{steps}_global_sgd", "accumulation_scope": "patient_chunk", "steps": steps, "lr_strategy": "global", "optimizer": "sgd", "chunk_size_windows": 30})
    if len(configs) != 16 or len({c["config_id"] for c in configs}) != 16:
        raise AssertionError("matrix must contain exactly 16 unique configurations")
    return configs


def main() -> None:
    if (OUT / "frozen_manifest.json").exists():
        raise FileExistsError(f"release already frozen: {OUT / 'frozen_manifest.json'}")
    recordings_path = ROOT / "manifests" / "recordings.parquet"
    recordings = pd.read_parquet(recordings_path)
    order = official_record_order(recordings)
    atomic_json(OUT / "record_order.json", order)
    checkpoints = {}
    for fold in (0, 1):
        checkpoint = SOURCE_RELEASE / "runs" / f"meta_band_fold{fold}_seed3407" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints[str(fold)] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
    manifest = {
        "release_id": RELEASE,
        "status": "frozen_before_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_release": str(SOURCE_RELEASE),
        "source_checkpoints": checkpoints,
        "folds": [0, 1], "splits": ["validation", "test"], "seed": 3407,
        "objective": "deterministic Band-SSL only; seizure labels never used for adaptation",
        "adaptive_scope": "last two Transformer blocks plus Band head",
        "base_lr": "learned bounded alpha stored in each existing meta_band checkpoint",
        "layerwise_ratio": {"penultimate_transformer_block": 0.5, "last_transformer_block": 1.0, "band_head": 1.0},
        "sgd": {"momentum": 0.0, "weight_decay": 0.0},
        "adam": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "state_reset": "patient boundary only"},
        "chunk_size_windows": 30,
        "chunk_causality": "score all windows in chunk j before one K-step mean-SSL update; update applies from chunk j+1",
        "main_causality": "current completed window is adapted before it is scored; no future window is used",
        "threshold_policy": "select independently on validation; lock before test; also report matched frozen-threshold sensitivity analysis",
        "primary_metric": "false_alarm_time_min_per_24h",
        "secondary_metrics": ["event_sensitivity", "fa_per_24h", "detection_delay_mean_s", "runtime", "gpu_peak_memory"],
        "frozen_baseline_policy": "join existing meta_band_frozen outputs; never rerun frozen inference",
        "execution_profile": {
            "window_batch_size": 128,
            "record_stream_batch_size": 128,
            "patient_streams_per_process": "all available (4 validation; 5 test)",
            "parallel_process_cap": 4,
            "gpu_slot_capacity": 4,
            "gpu_slots": {"window": 2, "record": 2, "patient": 1, "patient_adam": 1, "patient_chunk": 2},
            "mha_fastpath": "disabled only for torch.func vmap scopes because pinned PyTorch lacks a native-MHA batching rule; retained for chunk mode",
            "frozen_prefix_cache": "concatenate Band and raw views, run immutable patch embedding plus first 10 blocks once, then vmap only the adaptive two-block tail",
        },
        "configurations": configurations(), "expected_evaluation_jobs": 64,
        "record_order_path": str(OUT / "record_order.json"), "record_order_sha256": sha256(OUT / "record_order.json"),
        "recordings_manifest_sha256": sha256(recordings_path), "test_locked_against_retuning": True,
        "execution_code": {
            str(ROOT / "scripts" / name): sha256(ROOT / "scripts" / name)
            for name in ("265_evaluate_band_ttt_v2.py", "266_summarize_band_ttt_v2.py", "267_queue_band_ttt_v2.py", "268_import_existing_band_ttt_v2.py", "269_wait_then_queue_band_ttt_v2.sh")
        },
    }
    atomic_json(OUT / "frozen_manifest.json", manifest)
    print(json.dumps({"status": "frozen", "output": str(OUT), "configurations": 16, "jobs": 64}, indent=2))


if __name__ == "__main__":
    main()
