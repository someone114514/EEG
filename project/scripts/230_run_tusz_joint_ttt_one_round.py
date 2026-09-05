"""One-round external TTT for Joint-TTT CBraMod on the official TUSZ Eval cohort.

A single completed CHB Joint-TTT checkpoint (fold 0, seed 17) is used as the
source.  The 43-patient/880-EDF TUSZ Eval cohort is streamed once.  The frozen
reference is scored first.  The adapted condition scores each causal 120-s
block and then updates the CBraMod backbone with masked-patch reconstruction.
Early stopping uses only a held-out slice of the already-scored raw block.
TUSZ labels are read only after probabilities are written, for event metrics
and patient bootstrap intervals.

This is external, transductive TTT, not supervised TUSZ retraining and not a
new 5-fold experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events
from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv, tuh_model_view


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = os.environ.get(
    "TUSZ_JOINT_TTT_NAMESPACE", "cbramod-joint-ttt-tusz-one-round-v1"
)
OUT = ROOT / "outputs" / "reports" / NAMESPACE
TUSZ_MANIFEST = (
    ROOT
    / "outputs"
    / "reports"
    / "h6-tusz-raw-pre-router-v1"
    / "manifests"
    / "eval_manifest.json"
)
SOURCE_NAMESPACE = os.environ.get(
    "JOINT_TTT_SOURCE_NAMESPACE", "cbramod-joint-ttt-v1-formal"
)
SOURCE_CHECKPOINT = (
    ROOT / "outputs" / "reports" / SOURCE_NAMESPACE / "runs" / "fold0_seed17" / "checkpoint.pt"
)
SOURCE_EVAL_MANIFEST = (
    ROOT
    / "outputs"
    / "reports"
    / SOURCE_NAMESPACE
    / "evaluation"
    / "fold0_seed17"
    / "manifest.json"
)
MODEL_RATE = 200
WINDOW_S = 10
WINDOW_SAMPLES = MODEL_RATE * WINDOW_S
STEP_S = 2
STEP_SAMPLES = MODEL_RATE * STEP_S
WARMUP_S = 60
BLOCK_S = 120
CONTEXT_WINDOWS = 31
SOURCE_FOLD = 0
SOURCE_SEED = 17
MAX_ADAPT_UPDATES = 5
PATIENCE = 1
MIN_DELTA = 1e-4
ADAPT_MAX_WINDOWS = 8
RECONSTRUCTION_MASK_PATCHES = 5
ADAPT_LR = 1e-5
WEIGHT_DECAY = 1e-5
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260831


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_adaptation_module():
    path = ROOT / "scripts" / "202_cbramod_same_patient_adaptation.py"
    spec = importlib.util.spec_from_file_location("same_patient_adaptation_tusz", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def device_from_arg(value: str | None) -> torch.device:
    device = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def autocast_context(device: torch.device):
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def load_source(module):
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    state = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(state.get("update", -1)) != 5000:
        raise RuntimeError("source checkpoint is not the completed step-5000 checkpoint")
    if not bool(state.get("selected_by_validation", False)):
        raise RuntimeError("source checkpoint was not selected by validation")
    adapter = module.CBraModAdapter(module.PRETRAINED, train_backbone=True)
    adapter.load_state_dict(state["encoder"], strict=True)
    head = module.SharedContextHead()
    head.load_state_dict(state["head"], strict=True)
    encoder_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in adapter.state_dict().items()
    }
    head_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in head.state_dict().items()
    }
    return state, adapter, head, encoder_state, head_state


def reset_source(adapter, head, encoder_state, head_state, device: torch.device) -> None:
    adapter.load_state_dict(encoder_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    adapter.to(device)
    head.to(device)
    adapter.backbone.requires_grad_(True)
    adapter.projection.requires_grad_(False)
    head.requires_grad_(False)
    adapter.eval()
    head.eval()


def window_view(view: np.ndarray) -> np.ndarray:
    if view.ndim != 2 or view.shape[0] != 16:
        raise ValueError(f"unexpected CBraMod view shape {view.shape}")
    if view.shape[-1] < WINDOW_SAMPLES:
        return np.empty((0, 16, WINDOW_SAMPLES), dtype=np.float32)
    all_windows = np.lib.stride_tricks.sliding_window_view(
        view, WINDOW_SAMPLES, axis=-1
    )
    return all_windows[:, ::STEP_SAMPLES, :].transpose(1, 0, 2)


@torch.inference_mode()
def encode_windows(
    adapter, windows: np.ndarray, device: torch.device, micro_windows: int = 8
) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0, 16, 128), dtype=np.float32)
    adapter.eval()
    output: list[np.ndarray] = []
    for begin in range(0, len(windows), micro_windows):
        batch = np.ascontiguousarray(
            windows[begin : begin + micro_windows], dtype=np.float32
        )
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        with autocast_context(device):
            features = adapter.backbone_features(
                tensor.reshape(-1, 16, 10, MODEL_RATE)
            )
            projected = adapter.projection(features)
        output.append(projected.float().cpu().numpy())
    result = np.concatenate(output, axis=0)
    if result.shape != (len(windows), 16, 128) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid encoded windows {result.shape}")
    return result.astype(np.float32, copy=False)


@torch.inference_mode()
def predict_embeddings(
    head, embeddings: np.ndarray, device: torch.device, micro_contexts: int = 32
) -> np.ndarray:
    if len(embeddings) < CONTEXT_WINDOWS:
        return np.empty(0, dtype=np.float32)
    target_indices = np.arange(CONTEXT_WINDOWS - 1, len(embeddings), dtype=int)
    output: list[np.ndarray] = []
    for begin in range(0, len(target_indices), micro_contexts):
        indices = target_indices[begin : begin + micro_contexts]
        contexts = np.stack(
            [
                embeddings[index - (CONTEXT_WINDOWS - 1) : index + 1]
                for index in indices
            ]
        )
        tensor = torch.from_numpy(np.ascontiguousarray(contexts)).to(
            device=device, dtype=torch.float32
        )
        with autocast_context(device):
            probability = torch.sigmoid(head(tensor))
        output.append(probability.float().cpu().numpy())
    result = np.concatenate(output, axis=0)
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise RuntimeError("invalid probability output")
    return result.astype(np.float32, copy=False)


def masked_reconstruction_eval(
    adapter, raw_batch: np.ndarray, device: torch.device, seed: int
) -> float:
    if len(raw_batch) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    adapter.eval()
    batch = torch.from_numpy(
        np.ascontiguousarray(raw_batch, dtype=np.float32)
    ).to(device=device, dtype=torch.float32)
    reshaped = batch.reshape(-1, 16, 10, MODEL_RATE)
    mask = torch.zeros((len(batch), 16, 10), device=device, dtype=torch.bool)
    for index in range(len(batch)):
        chosen = rng.choice(
            10, size=RECONSTRUCTION_MASK_PATCHES, replace=False
        )
        mask[index, :, chosen] = True
    with autocast_context(device):
        reconstruction = adapter.backbone(reshaped, mask=mask)
    selected = mask.unsqueeze(-1).expand_as(reconstruction)
    value = torch.nn.functional.mse_loss(reconstruction[selected], reshaped[selected])
    result = float(value.detach().float().cpu())
    if not math.isfinite(result):
        raise RuntimeError("non-finite reconstruction validation loss")
    return result


def early_stop_adaptation(
    module,
    adapter,
    optimizer,
    raw_batch: np.ndarray,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    if len(raw_batch) < 3:
        return [
            {
                "status": "skipped",
                "reason": "fewer_than_three_windows",
                "windows": int(len(raw_batch)),
            }
        ]
    raw_batch = np.ascontiguousarray(raw_batch, dtype=np.float32)
    split = max(1, len(raw_batch) - 2)
    train_raw = raw_batch[:split]
    validation_raw = raw_batch[split:]
    rng = np.random.default_rng(seed)
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in adapter.backbone.state_dict().items()
    }
    best_validation = masked_reconstruction_eval(
        adapter, validation_raw, device, seed + 1
    )
    no_improve = 0
    history: list[dict[str, Any]] = [
        {
            "update": 0,
            "validation_reconstruction_loss": best_validation,
            "status": "initial",
        }
    ]
    for update in range(1, MAX_ADAPT_UPDATES + 1):
        train_loss, grad_norm = module.ttt_update(
            adapter,
            optimizer,
            train_raw,
            device,
            rng,
            micro_windows=min(4, len(train_raw)),
        )
        validation_loss = masked_reconstruction_eval(
            adapter, validation_raw, device, seed + 1 + update
        )
        improved = validation_loss < best_validation - MIN_DELTA
        history.append(
            {
                "update": update,
                "train_reconstruction_loss": float(train_loss),
                "validation_reconstruction_loss": float(validation_loss),
                "grad_norm_pre_clip": float(grad_norm),
                "improved": bool(improved),
            }
        )
        if improved:
            best_validation = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in adapter.backbone.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > PATIENCE:
            break
    adapter.backbone.load_state_dict(best_state, strict=True)
    adapter.eval()
    improved_updates = [
        int(item.get("update", 0))
        for item in history
        if item.get("improved", False)
    ]
    history[-1]["selected_best_validation_reconstruction_loss"] = float(best_validation)
    history[-1]["updates_selected"] = max(improved_updates) if improved_updates else 0
    return history


def event_metrics(
    probabilities: pd.DataFrame,
    record: dict[str, Any],
    threshold: float,
    condition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table = probabilities.sort_values("window_end_s", kind="stable")
    first_end = (
        float(table.window_end_s.iloc[0])
        if len(table)
        else min(70.0, float(record["duration_s"]))
    )
    truths = [
        (max(first_end, float(event["start_s"])), float(event["end_s"]))
        for event in record.get("seizure_events", [])
        if float(event["end_s"]) > first_end
    ]
    predictions = (
        eventize(
            table.window_end_s.to_numpy(dtype=float),
            table.probability.to_numpy(dtype=float),
            threshold=float(threshold),
        )
        if len(table) >= 2
        else []
    )
    matching = match_events(predictions, truths)
    delays = [
        max(0.0, predictions[pair.prediction_index].start_s - truths[pair.truth_index][0])
        for pair in matching.pairs
    ]
    seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
    nonseizure_s = max(
        0.0, float(record["duration_s"]) - first_end - seizure_seconds
    )
    metric = {
        "condition": condition,
        "patient_id": str(record["patient_id"]),
        "recording": str(record["relative_edf"]),
        "threshold": float(threshold),
        "tp": int(len(matching.pairs)),
        "fp": int(len(matching.unmatched_predictions)),
        "fn": int(len(matching.unmatched_truths)),
        "truth_events": int(len(truths)),
        "nonseizure_s": float(nonseizure_s),
        "delay_count": int(len(delays)),
        "delay_sum_s": float(sum(delays)) if delays else 0.0,
        "median_delay_s": float(np.median(delays)) if delays else np.nan,
    }
    matched_by_prediction = {pair.prediction_index: pair for pair in matching.pairs}
    event_rows = []
    for index, prediction in enumerate(predictions):
        pair = matched_by_prediction.get(index)
        event_rows.append(
            {
                "condition": condition,
                "patient_id": str(record["patient_id"]),
                "recording": str(record["relative_edf"]),
                "event_index": int(index),
                "start_s": float(prediction.start_s),
                "end_s": float(prediction.end_s),
                "peak_probability": float(prediction.peak_probability),
                "matched_truth_index": int(pair.truth_index)
                if pair is not None
                else None,
                "overlap_s": float(pair.overlap_s) if pair is not None else 0.0,
                "is_true_positive": bool(pair is not None),
            }
        )
    return metric, event_rows


def aggregate(record_frame: pd.DataFrame) -> dict[str, Any]:
    if record_frame.empty:
        return {
            "patients": 0,
            "records": 0,
            "truth_events": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "sensitivity": np.nan,
            "fa_per_24h": np.nan,
            "mean_delay_s": np.nan,
            "nonseizure_h": 0.0,
        }
    truth = int(record_frame.truth_events.sum())
    tp = int(record_frame.tp.sum())
    fp = int(record_frame.fp.sum())
    seconds = float(record_frame.nonseizure_s.sum())
    delay_count = int(record_frame.delay_count.sum())
    return {
        "patients": int(record_frame.patient_id.nunique()),
        "records": int(len(record_frame)),
        "truth_events": truth,
        "tp": tp,
        "fp": fp,
        "fn": truth - tp,
        "sensitivity": float(tp / truth) if truth else np.nan,
        "fa_per_24h": float(fp / (seconds / 86400.0)) if seconds > 0 else np.nan,
        "mean_delay_s": float(record_frame.delay_sum_s.sum() / delay_count)
        if delay_count
        else np.nan,
        "nonseizure_h": seconds / 3600.0,
    }


def patient_bootstrap(record_frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    patients = sorted(record_frame.patient_id.astype(str).unique().tolist())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(patients, len(patients), replace=True)
        pieces = [
            record_frame[record_frame.patient_id.astype(str) == patient]
            for patient in sampled
        ]
        summary = aggregate(pd.concat(pieces, ignore_index=True))
        rows.append(
            {
                "condition": condition,
                "replicate": replicate,
                "sensitivity": summary["sensitivity"],
                "fa_per_24h": summary["fa_per_24h"],
                "mean_delay_s": summary["mean_delay_s"],
                "patients_sampled": len(patients),
            }
        )
    return pd.DataFrame(rows)


def process_record(
    module,
    adapter,
    head,
    encoder_state,
    head_state,
    record: dict[str, Any],
    record_index: int,
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    relative = Path(str(record["relative_edf"]))
    edf_path = Path(str(record["dataset_root"])) / relative
    if not edf_path.is_file():
        raise FileNotFoundError(edf_path)
    print(
        f"TUSZ_JOINT_TTT_RECORD_START index={record_index + 1} relative={relative}",
        flush=True,
    )
    raw_uv, source_hz, _ = load_tuh_canonical_raw_uv(edf_path)
    view = tuh_model_view(raw_uv, source_hz, "cbramod")
    windows = window_view(view)
    if len(windows) < CONTEXT_WINDOWS:
        empty = pd.DataFrame(
            {
                "window_end_s": pd.Series(dtype="float64"),
                "probability": pd.Series(dtype="float32"),
            }
        )
        frozen_metric, frozen_events = event_metrics(
            empty, record, threshold, "frozen"
        )
        adapted_metric, adapted_events = event_metrics(
            empty, record, threshold, "adapted"
        )
        return {
            "record_index": record_index,
            "recording": str(relative),
            "patient_id": str(record["patient_id"]),
            "frozen_probability": empty,
            "adapted_probability": empty,
            "frozen_metric": frozen_metric,
            "adapted_metric": adapted_metric,
            "events": frozen_events + adapted_events,
            "adaptation_history": [],
            "embedding_windows": int(len(windows)),
            "scored_blocks": 0,
            "adaptation_updates": 0,
            "short_record": True,
        }

    reset_source(adapter, head, encoder_state, head_state, device)
    frozen_embeddings = encode_windows(adapter, windows, device)
    frozen_probabilities = predict_embeddings(head, frozen_embeddings, device)
    frozen_times = (
        np.arange(CONTEXT_WINDOWS - 1, len(windows), dtype=float) * STEP_S
        + WINDOW_S
    )
    frozen_table = pd.DataFrame(
        {
            "patient_id": str(record["patient_id"]),
            "recording": str(relative),
            "window_end_s": frozen_times,
            "probability": frozen_probabilities,
        }
    )
    frozen_metric, frozen_events = event_metrics(
        frozen_table, record, threshold, "frozen"
    )

    reset_source(adapter, head, encoder_state, head_state, device)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in adapter.backbone.parameters()
            if parameter.requires_grad
        ],
        lr=ADAPT_LR,
        weight_decay=WEIGHT_DECAY,
    )
    target_indices = np.arange(CONTEXT_WINDOWS - 1, len(windows), dtype=int)
    block_ids = np.floor(
        ((target_indices - (CONTEXT_WINDOWS - 1)) * STEP_S) / BLOCK_S
    ).astype(int)
    adapted_rows: list[dict[str, Any]] = []
    adaptation_history: list[dict[str, Any]] = []
    adaptation_updates = 0
    for block_id in sorted(np.unique(block_ids)):
        indices = target_indices[block_ids == block_id]
        first = int(indices[0])
        last = int(indices[-1])
        chunk_start = first - (CONTEXT_WINDOWS - 1)
        chunk_end = last + 1
        embeddings = encode_windows(adapter, windows[chunk_start:chunk_end], device)
        probabilities = predict_embeddings(head, embeddings, device)
        times = indices.astype(float) * STEP_S + WINDOW_S
        if len(probabilities) != len(times):
            raise RuntimeError(
                f"block probability/time mismatch {len(probabilities)} vs {len(times)}"
            )
        adapted_rows.extend(
            {
                "patient_id": str(record["patient_id"]),
                "recording": str(relative),
                "window_end_s": float(t),
                "probability": float(p),
                "block": int(block_id),
            }
            for t, p in zip(times, probabilities, strict=True)
        )
        anchor_windows = windows[indices]
        if len(anchor_windows) > ADAPT_MAX_WINDOWS:
            chosen = np.linspace(
                0, len(anchor_windows) - 1, ADAPT_MAX_WINDOWS, dtype=int
            )
            anchor_windows = anchor_windows[chosen]
        block_history = early_stop_adaptation(
            module,
            adapter,
            optimizer,
            np.asarray(anchor_windows),
            device,
            SOURCE_SEED + record_index * 1000 + int(block_id),
        )
        selected_updates = int(block_history[-1].get("updates_selected", 0))
        adaptation_updates += selected_updates
        adaptation_history.append(
            {
                "block": int(block_id),
                "anchors_scored": int(len(indices)),
                "adaptation_windows": int(len(anchor_windows)),
                "updates_selected": selected_updates,
                "history": block_history,
                "scored_before_update": True,
            }
        )
    adapted_table = pd.DataFrame(adapted_rows)
    adapted_metric, adapted_events = event_metrics(
        adapted_table, record, threshold, "adapted"
    )
    embedding_count = int(len(frozen_embeddings))
    del raw_uv, view, windows, frozen_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "record_index": record_index,
        "recording": str(relative),
        "patient_id": str(record["patient_id"]),
        "frozen_probability": frozen_table,
        "adapted_probability": adapted_table,
        "frozen_metric": frozen_metric,
        "adapted_metric": adapted_metric,
        "events": frozen_events + adapted_events,
        "adaptation_history": adaptation_history,
        "embedding_windows": embedding_count,
        "scored_blocks": int(len(adaptation_history)),
        "adaptation_updates": adaptation_updates,
        "short_record": False,
    }


def run(args: argparse.Namespace) -> None:
    if (OUT / "manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite completed output: {OUT / 'manifest.json'}")
    if not TUSZ_MANIFEST.is_file():
        raise FileNotFoundError(TUSZ_MANIFEST)
    if not SOURCE_EVAL_MANIFEST.is_file():
        raise FileNotFoundError(
            f"CHB Joint-TTT evaluation is not complete: {SOURCE_EVAL_MANIFEST}"
        )
    source_eval = json.loads(SOURCE_EVAL_MANIFEST.read_text())
    if (
        source_eval.get("ttt_method") != "joint"
        or int(source_eval.get("fold", -1)) != SOURCE_FOLD
        or int(source_eval.get("seed", -1)) != SOURCE_SEED
    ):
        raise RuntimeError(
            "source evaluation manifest is not the frozen Joint fold0/seed17 evaluation"
        )
    threshold = float(source_eval["threshold"])
    tusz_manifest = json.loads(TUSZ_MANIFEST.read_text())
    if tusz_manifest.get("partition") != "eval":
        raise RuntimeError("this run requires the official TUSZ Eval partition")
    records = list(tusz_manifest["records"])
    set_seed(SOURCE_SEED)
    device = device_from_arg(args.device)
    module = load_adaptation_module()
    source_state, adapter, head, encoder_state, head_state = load_source(module)
    OUT.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "release_id": NAMESPACE,
        "status": "running",
        "dataset": "TUSZ_v2.0.6",
        "partition": "eval",
        "patient_count": int(tusz_manifest["patient_count"]),
        "record_count": int(len(records)),
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "source_checkpoint_update": 5000,
        "source_checkpoint_selected_by_validation": bool(
            source_state.get("selected_by_validation", False)
        ),
        "source_evaluation_manifest": str(SOURCE_EVAL_MANIFEST.relative_to(ROOT)),
        "source_evaluation_manifest_sha256": sha256(SOURCE_EVAL_MANIFEST),
        "tusz_manifest": str(TUSZ_MANIFEST.relative_to(ROOT)),
        "tusz_manifest_sha256": sha256(TUSZ_MANIFEST),
        "seed": SOURCE_SEED,
        "source_fold": SOURCE_FOLD,
        "model": "cbramod",
        "input": {
            "sampling_hz": MODEL_RATE,
            "channels": 16,
            "window_s": WINDOW_S,
            "step_s": STEP_S,
            "context_windows": CONTEXT_WINDOWS,
            "context_history_s": 60,
        },
        "adaptation": {
            "objective": "masked-patch reconstruction only",
            "trainable_scope": "full CBraMod backbone",
            "head_frozen": True,
            "projection_frozen": True,
            "block_s": BLOCK_S,
            "max_updates": MAX_ADAPT_UPDATES,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "heldout_windows_per_block": 2,
            "max_adaptation_windows_per_block": ADAPT_MAX_WINDOWS,
            "score_before_update": True,
            "test_labels_used": False,
        },
        "early_stopping_source": (
            "unsupervised reconstruction loss on held-out raw windows from the "
            "already-scored block; no seizure labels"
        ),
        "threshold": threshold,
        "threshold_source": (
            "CHB Joint-TTT fold0/seed17 validation-only evaluation; "
            "TUSZ Eval not used for selection"
        ),
        "test_evaluation_count_per_condition": 1,
        "tusz_labels_used_for_selection": False,
        "tusz_labels_used_for_adaptation": False,
        "model_training_on_tusz": False,
        "created_utc": now(),
    }
    atomic_json(OUT / "manifest.json", run_manifest)
    atomic_json(
        OUT / "progress.json",
        {
            "status": "running",
            "partition": "eval",
            "model": "cbramod",
            "source_fold": SOURCE_FOLD,
            "seed": SOURCE_SEED,
            "completed_records": 0,
            "total_records": len(records),
            "current_record": None,
            "updated_utc": now(),
        },
    )
    record_dir = OUT / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    frozen_tables: list[pd.DataFrame] = []
    adapted_tables: list[pd.DataFrame] = []
    frozen_metrics: list[dict[str, Any]] = []
    adapted_metrics: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, original_record in enumerate(records):
        record = dict(original_record)
        record["dataset_root"] = tusz_manifest.get(
            "dataset_root", "/mnt/d/TUH_EEG/TUSZ_v2.0.6"
        )
        frozen_path = record_dir / f"record_{index:04d}_frozen.parquet"
        adapted_path = record_dir / f"record_{index:04d}_adapted.parquet"
        metadata_path = record_dir / f"record_{index:04d}.json"
        if frozen_path.exists() or adapted_path.exists() or metadata_path.exists():
            raise RuntimeError(
                f"partial record artifact exists; refusing to overwrite index {index}: "
                f"{record_dir}"
            )
        atomic_json(
            OUT / "progress.json",
            {
                "status": "running",
                "partition": "eval",
                "model": "cbramod",
                "source_fold": SOURCE_FOLD,
                "seed": SOURCE_SEED,
                "completed_records": index,
                "total_records": len(records),
                "current_record": str(record["relative_edf"]),
                "updated_utc": now(),
            },
        )
        result = process_record(
            module,
            adapter,
            head,
            encoder_state,
            head_state,
            record,
            index,
            threshold,
            device,
        )
        result["frozen_probability"].to_parquet(frozen_path, index=False)
        result["adapted_probability"].to_parquet(adapted_path, index=False)
        atomic_json(
            metadata_path,
            {
                "record_index": index,
                "relative_edf": result["recording"],
                "patient_id": result["patient_id"],
                "frozen_metric": result["frozen_metric"],
                "adapted_metric": result["adapted_metric"],
                "events": result["events"],
                "adaptation_history": result["adaptation_history"],
                "embedding_windows": result["embedding_windows"],
                "scored_blocks": result["scored_blocks"],
                "adaptation_updates": result["adaptation_updates"],
                "short_record": result["short_record"],
                "threshold": threshold,
                "created_utc": now(),
            },
        )
        frozen_tables.append(result["frozen_probability"])
        adapted_tables.append(result["adapted_probability"])
        frozen_metrics.append(result["frozen_metric"])
        adapted_metrics.append(result["adapted_metric"])
        event_rows.extend(result["events"])
        history_rows.append(
            {
                "record_index": index,
                "recording": result["recording"],
                "patient_id": result["patient_id"],
                "embedding_windows": result["embedding_windows"],
                "scored_blocks": result["scored_blocks"],
                "adaptation_updates": result["adaptation_updates"],
                "short_record": result["short_record"],
            }
        )
        atomic_json(
            OUT / "progress.json",
            {
                "status": "running",
                "partition": "eval",
                "model": "cbramod",
                "source_fold": SOURCE_FOLD,
                "seed": SOURCE_SEED,
                "completed_records": index + 1,
                "total_records": len(records),
                "current_record": None,
                "embedding_windows": int(
                    sum(row["embedding_windows"] for row in history_rows)
                ),
                "adaptation_updates": int(
                    sum(row["adaptation_updates"] for row in history_rows)
                ),
                "elapsed_s": time.monotonic() - started,
                "updated_utc": now(),
            },
        )
        print(
            f"TUSZ_JOINT_TTT_RECORD_DONE index={index + 1}/{len(records)} "
            f"blocks={result['scored_blocks']} updates={result['adaptation_updates']}",
            flush=True,
        )

    frozen_frame = pd.DataFrame(frozen_metrics)
    adapted_frame = pd.DataFrame(adapted_metrics)
    all_frozen_probabilities = (
        pd.concat(frozen_tables, ignore_index=True)
        if frozen_tables
        else pd.DataFrame()
    )
    all_adapted_probabilities = (
        pd.concat(adapted_tables, ignore_index=True)
        if adapted_tables
        else pd.DataFrame()
    )
    all_events = pd.DataFrame(event_rows)
    frozen_summary = aggregate(frozen_frame)
    adapted_summary = aggregate(adapted_frame)
    bootstrap = pd.concat(
        [
            patient_bootstrap(frozen_frame, "frozen"),
            patient_bootstrap(adapted_frame, "adapted"),
        ],
        ignore_index=True,
    )
    all_frozen_probabilities.to_parquet(
        OUT / "frozen_probabilities.parquet", index=False
    )
    all_adapted_probabilities.to_parquet(
        OUT / "adapted_probabilities.parquet", index=False
    )
    frozen_frame.to_csv(OUT / "frozen_record_metrics.csv", index=False)
    adapted_frame.to_csv(OUT / "adapted_record_metrics.csv", index=False)
    all_events.to_parquet(OUT / "events.parquet", index=False)
    pd.DataFrame(history_rows).to_csv(
        OUT / "adaptation_history_summary.csv", index=False
    )
    bootstrap.to_parquet(OUT / "patient_bootstrap.parquet", index=False)
    delta = {
        "sensitivity": float(adapted_summary["sensitivity"] - frozen_summary["sensitivity"])
        if np.isfinite(adapted_summary["sensitivity"])
        and np.isfinite(frozen_summary["sensitivity"])
        else np.nan,
        "fa_per_24h": float(adapted_summary["fa_per_24h"] - frozen_summary["fa_per_24h"])
        if np.isfinite(adapted_summary["fa_per_24h"])
        and np.isfinite(frozen_summary["fa_per_24h"])
        else np.nan,
        "mean_delay_s": float(adapted_summary["mean_delay_s"] - frozen_summary["mean_delay_s"])
        if np.isfinite(adapted_summary["mean_delay_s"])
        and np.isfinite(frozen_summary["mean_delay_s"])
        else np.nan,
    }
    output_paths = [
        "frozen_probabilities.parquet",
        "adapted_probabilities.parquet",
        "frozen_record_metrics.csv",
        "adapted_record_metrics.csv",
        "events.parquet",
        "adaptation_history_summary.csv",
        "patient_bootstrap.parquet",
    ]
    summary = {
        "release_id": NAMESPACE,
        "status": "complete",
        "dataset": "TUSZ_v2.0.6",
        "partition": "eval",
        "source_fold": SOURCE_FOLD,
        "seed": SOURCE_SEED,
        "threshold": threshold,
        "threshold_source": run_manifest["threshold_source"],
        "frozen": frozen_summary,
        "adapted": adapted_summary,
        "delta_adapted_minus_frozen": delta,
        "records_completed": len(records),
        "expected_records": len(records),
        "probability_rows": {
            "frozen": len(all_frozen_probabilities),
            "adapted": len(all_adapted_probabilities),
        },
        "patient_bootstrap_replicates_per_condition": BOOTSTRAP_REPLICATES,
        "outputs": {},
        "created_utc": now(),
    }
    summary["outputs"] = {name: sha256(OUT / name) for name in output_paths}
    atomic_json(OUT / "summary.json", summary)
    run_manifest.update(
        {
            "status": "complete",
            "records_completed": len(records),
            "summary": "summary.json",
            "outputs": summary["outputs"],
            "finished_utc": now(),
        }
    )
    atomic_json(OUT / "manifest.json", run_manifest)
    atomic_json(
        OUT / "progress.json",
        {
            "status": "complete",
            "partition": "eval",
            "model": "cbramod",
            "source_fold": SOURCE_FOLD,
            "seed": SOURCE_SEED,
            "completed_records": len(records),
            "total_records": len(records),
            "updated_utc": now(),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
