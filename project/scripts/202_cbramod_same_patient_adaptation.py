"""Same-patient CBraMod adaptation upper-bound experiment.

This experiment is deliberately separate from H7.  It compares a frozen
CBraMod detector with two *online* same-patient adaptation conditions:

* ``ttt``: unlabeled masked-patch reconstruction updates of the CBraMod
  backbone only;
* ``supervised_oracle``: label-informed updates of the CBraMod backbone and
  downstream projection while the task head remains frozen.

Every chronological block is scored before any update from that block.  Thus
the supervised condition is an online patient-label upper bound, not an
independent generalisation result.  The experiment uses one pre-registered
outer-test fold and three representative patients; it never changes the
source detector threshold.

The script reads the deterministic CBraMod signal cache produced by the
official CHB preprocessing profile.  It does not retrain the CHB detector
head, reselect a threshold, or touch earlier H3/H6/H7 outputs.
"""
from __future__ import annotations

import argparse
import hashlib
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
from torch import nn
from torch.nn import functional as F

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events
from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.shared_head import SharedContextHead


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT = ROOT / "outputs/reports" / os.environ.get(
    "CBRAMOD_TTT_NAMESPACE", "cbramod-chb-same-patient-ttt-oracle-v1"
)
WINDOWS = ROOT / "manifests/windows.parquet"
RECORDINGS = ROOT / "manifests/recordings.parquet"
SEIZURES = ROOT / "manifests/seizures.parquet"
CV_MANIFEST = ROOT / "manifests/groupkfold_cv_v1/cv_manifest.json"
FOLD_FILE = ROOT / "manifests/groupkfold_cv_v1/fold_0.json"
SOURCE_CHECKPOINT = ROOT / "runs/v3-groupkfold-confirmatory-v1/cbramod/split0_seed17_main/checkpoints/step_05000.pt"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"
SIGNAL_CACHE = Path(os.environ.get("BFA_CACHE_ROOT", "/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod"))
THRESHOLD = 0.01
MODEL_RATE = 200
WINDOW_S = 10
WINDOW_STEP_S = 2
WARMUP_S = 60
ANCHOR_STEP_S = 10
BLOCK_S = 120
CONTEXT_WINDOWS = 31
CONTEXT_HISTORY_S = (CONTEXT_WINDOWS - 1) * WINDOW_STEP_S
SEEDS = (17, 42, 3407)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def source_state() -> dict[str, Any]:
    state = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(state.get("update", -1)) != 5000:
        raise RuntimeError("source detector checkpoint is not step 5000")
    return state


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = pd.read_parquet(WINDOWS)
    recordings = pd.read_parquet(RECORDINGS)
    seizures = pd.read_parquet(SEIZURES)
    required_windows = {"patient", "recording", "start", "end", "label", "relative_path"}
    if not required_windows.issubset(windows.columns):
        raise RuntimeError(f"windows manifest lacks {sorted(required_windows - set(windows.columns))}")
    return windows, recordings, seizures


def choose_patients(
    windows: pd.DataFrame, recordings: pd.DataFrame, seizures: pd.DataFrame
) -> tuple[list[str], dict[str, Any]]:
    fold = json.loads(FOLD_FILE.read_text())
    outer_test = list(fold["test"])
    duration = recordings.groupby("patient_id", sort=False)["duration_s"].sum()
    seizure_counts = seizures.groupby("patient_id", sort=False).size()
    rows = []
    for patient in outer_test:
        rows.append(
            {
                "patient_id": patient,
                "seizures": int(seizure_counts.get(patient, 0)),
                "duration_s": float(duration.get(patient, 0.0)),
                "recordings": int((recordings.patient_id == patient).sum()),
            }
        )
    table = pd.DataFrame(rows)
    if len(table) < 3:
        raise RuntimeError("fold 0 has fewer than three outer-test patients")
    # Pre-register one high-burden, one middle-burden and one low-burden case.
    # Ties are resolved by duration and then canonical patient ID; no model
    # output or post-adaptation result enters this choice.
    high = table.sort_values(["seizures", "duration_s", "patient_id"], ascending=[False, False, True]).iloc[0]
    low = table.sort_values(["seizures", "duration_s", "patient_id"], ascending=[True, True, True]).iloc[0]
    middle_candidates = table[~table.patient_id.isin({high.patient_id, low.patient_id})].copy()
    middle = middle_candidates.iloc[(middle_candidates.seizures - middle_candidates.seizures.median()).abs().argsort().iloc[0]]
    selected = [str(high.patient_id), str(middle.patient_id), str(low.patient_id)]
    return selected, {
        "selection_rule": "fold0 outer-test; high, median-nearest, and low seizure burden; ties duration then patient ID",
        "outer_test_patients": outer_test,
        "candidate_table": table.to_dict(orient="records"),
        "selected_patients": selected,
        "selected_roles": {selected[0]: "high_seizure_burden", selected[1]: "middle_seizure_burden", selected[2]: "low_seizure_burden"},
    }


def tensor_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_detector(
    state: dict[str, Any],
    method: str,
    device: torch.device,
    *,
    unfreeze_head: bool = False,
) -> tuple[CBraModAdapter, SharedContextHead]:
    if method not in {"frozen", "ttt", "supervised_oracle"}:
        raise ValueError(method)
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.projection.load_state_dict(
        {"weight": state["encoder"]["projection.weight"], "bias": state["encoder"]["projection.bias"]}
    )
    head = SharedContextHead()
    head.load_state_dict(state["head"], strict=True)
    # The context head is frozen for the prespecified adaptation experiments.
    # ``unfreeze_head`` is opt-in and is reserved for a clearly labelled
    # diagnostic ceiling; it never changes the original runs.
    head.requires_grad_(bool(unfreeze_head and method == "supervised_oracle"))
    adapter.backbone.requires_grad_(method != "frozen")
    adapter.projection.requires_grad_(method == "supervised_oracle")
    adapter.to(device)
    head.to(device)
    adapter.eval()
    head.eval()
    return adapter, head


def signal_path(relative_path: str) -> Path:
    path = SIGNAL_CACHE / Path(relative_path).with_suffix(".npy")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def anchor_rows(windows: pd.DataFrame, patient: str, recording: str) -> pd.DataFrame:
    rows = windows[(windows.patient == patient) & (windows.recording == recording)].copy()
    rows = rows[rows.start >= WARMUP_S]
    # The 2-s source index gives an exact, reproducible 10-s anchor grid.
    rows["_start_index"] = np.rint(rows.start.to_numpy(float) / WINDOW_STEP_S).astype(int)
    rows = rows[rows["_start_index"] % int(ANCHOR_STEP_S / WINDOW_STEP_S) == 0]
    return rows.sort_values("start", kind="stable").reset_index(drop=True)


def context_from_view(view: np.ndarray, start_s: float) -> np.ndarray:
    start = int(round(start_s * MODEL_RATE))
    starts = start + np.arange(CONTEXT_WINDOWS, dtype=int) * int(WINDOW_STEP_S * MODEL_RATE)
    stop = starts[-1] + WINDOW_S * MODEL_RATE
    if starts[0] < 0 or stop > view.shape[-1]:
        raise ValueError(f"context outside signal: {start_s} / {view.shape}")
    return np.stack(
        [view[:, left : left + WINDOW_S * MODEL_RATE] for left in starts], axis=0
    ).astype(np.float32, copy=False)


def raw_windows_for_starts(view: np.ndarray, starts_s: np.ndarray) -> np.ndarray:
    starts = np.rint(starts_s * MODEL_RATE).astype(int)
    stop = starts + WINDOW_S * MODEL_RATE
    if starts.min(initial=0) < 0 or stop.max(initial=0) > view.shape[-1]:
        raise ValueError(f"raw window outside signal: {starts_s.min()}..{starts_s.max()} / {view.shape}")
    return np.stack(
        [view[:, left : left + WINDOW_S * MODEL_RATE] for left in starts], axis=0
    ).astype(np.float32, copy=False)


@torch.no_grad()
def score_contexts(
    adapter: CBraModAdapter,
    head: SharedContextHead,
    contexts: list[np.ndarray],
    device: torch.device,
    micro_contexts: int = 2,
) -> np.ndarray:
    adapter.eval()
    head.eval()
    if not contexts:
        return np.empty(0, dtype=np.float32)
    output: list[np.ndarray] = []
    for begin in range(0, len(contexts), micro_contexts):
        batch = np.stack(contexts[begin : begin + micro_contexts], axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device=device, dtype=torch.float32)
        flat = tensor.reshape(-1, 16, 10, MODEL_RATE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features = adapter.backbone_features(flat)
            projected = adapter.projection(features)
            projected = projected.reshape(tensor.shape[0], CONTEXT_WINDOWS, 16, 128)
            logits = head(projected)
            probability = torch.sigmoid(logits)
        output.append(probability.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def ttt_update(
    adapter: CBraModAdapter,
    optimizer: torch.optim.Optimizer,
    raw_batch: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    micro_windows: int = 4,
) -> tuple[float, float]:
    adapter.train()
    adapter.projection.eval()
    optimizer.zero_grad(set_to_none=True)
    losses: list[tuple[float, int]] = []
    for begin in range(0, len(raw_batch), micro_windows):
        batch = torch.from_numpy(np.ascontiguousarray(raw_batch[begin : begin + micro_windows])).to(device=device, dtype=torch.float32)
        mask = torch.zeros((len(batch), 16, 10), device=device, dtype=torch.bool)
        for index in range(len(batch)):
            chosen = rng.choice(10, size=5, replace=False)
            mask[index, :, chosen] = True
        reconstruction = adapter.backbone(batch.reshape(-1, 16, 10, MODEL_RATE), mask=mask)
        expanded = mask.unsqueeze(-1).expand_as(reconstruction)
        loss = F.mse_loss(reconstruction[expanded], batch.reshape(-1, 16, 10, MODEL_RATE)[expanded])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite TTT reconstruction loss")
        fraction = len(batch) / len(raw_batch)
        (loss * fraction).backward()
        losses.append((float(loss.detach().cpu()), len(batch)))
    trainable = [p for p in adapter.backbone.parameters() if p.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
    if not math.isfinite(grad_norm):
        raise RuntimeError("non-finite TTT gradient")
    optimizer.step()
    mean_loss = sum(value * count for value, count in losses) / max(1, len(raw_batch))
    return float(mean_loss), grad_norm


def supervised_update(
    adapter: CBraModAdapter,
    head: SharedContextHead,
    optimizer: torch.optim.Optimizer,
    replay: list[tuple[np.ndarray, float]],
    device: torch.device,
    micro_contexts: int = 1,
    steps: int = 1,
    max_positive: int = 4,
    max_negative: int = 4,
) -> tuple[float, float, int]:
    positives = [item for item in replay if item[1] == 1.0]
    negatives = [item for item in replay if item[1] == 0.0]
    if not positives or not negatives:
        return float("nan"), float("nan"), 0
    # Balance labels using only already-scored blocks; this avoids a trivial
    # negative collapse during long seizure-free periods.
    if max_positive <= 0 or max_negative <= 0:
        raise ValueError("max_positive and max_negative must be positive")
    count = min(max_positive, len(positives))
    negative_count = min(max_negative, len(negatives))
    # Keep the most recent examples from each class, with an explicit cap on
    # the negative side.  The default remains the original balanced 4+4
    # replay; larger negative caps are opt-in diagnostic stress tests.
    selected = positives[-count:] + negatives[-negative_count:]
    contexts = [item[0] for item in selected]
    labels = np.asarray([item[1] for item in selected], dtype=np.float32)
    if steps <= 0:
        raise ValueError("steps must be positive")
    adapter.train()
    head.train()
    all_losses: list[float] = []
    all_grad_norms: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses: list[tuple[float, int]] = []
        for begin in range(0, len(contexts), micro_contexts):
            batch = np.stack(contexts[begin : begin + micro_contexts], axis=0)
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device=device, dtype=torch.float32)
            flat = tensor.reshape(-1, 16, 10, MODEL_RATE)
            features = adapter.backbone_features(flat)
            projected = adapter.projection(features)
            projected = projected.reshape(tensor.shape[0], CONTEXT_WINDOWS, 16, 128)
            logits = head(projected)
            target = torch.from_numpy(labels[begin : begin + len(batch)]).to(device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite supervised adaptation loss")
            fraction = len(batch) / len(contexts)
            (loss * fraction).backward()
            losses.append((float(loss.detach().cpu()), len(batch)))
        trainable = [
            p
            for p in list(adapter.backbone.parameters())
            + list(adapter.projection.parameters())
            + list(head.parameters())
            if p.requires_grad
        ]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite supervised adaptation gradient")
        optimizer.step()
        mean_loss = sum(value * count_ for value, count_ in losses) / max(1, len(contexts))
        all_losses.append(float(mean_loss))
        all_grad_norms.append(float(grad_norm))
    return float(np.mean(all_losses)), float(np.mean(all_grad_norms)), len(contexts)


def score_events(
    table: pd.DataFrame,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
    patient: str,
) -> dict[str, Any]:
    true_positive = 0
    false_alarm = 0
    truth_count = 0
    delays: list[float] = []
    nonseizure_seconds = 0.0
    for recording, group in table.groupby("recording", sort=False):
        group = group.sort_values("time_s", kind="stable")
        rec = recordings[recordings.recording_id == recording]
        if rec.empty:
            continue
        duration = float(rec.duration_s.iloc[0])
        truths = [
            (max(WARMUP_S, float(row.start_s)), float(row.end_s))
            for row in seizures[(seizures.patient_id == patient) & (seizures.recording_id == recording)].itertuples(index=False)
            if float(row.end_s) > WARMUP_S
        ]
        predictions = eventize(
            group.time_s.to_numpy(dtype=float),
            group.probability.to_numpy(dtype=float),
            threshold=THRESHOLD,
        )
        matched = match_events(predictions, truths)
        true_positive += len(matched.pairs)
        false_alarm += len(matched.unmatched_predictions)
        truth_count += len(truths)
        for pair in matched.pairs:
            delay = predictions[pair.prediction_index].start_s - truths[pair.truth_index][0]
            delays.append(max(0.0, float(delay)))
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        nonseizure_seconds += max(0.0, duration - WARMUP_S - seizure_seconds)
    hours = nonseizure_seconds / 3600.0
    return {
        "event_sensitivity": float(true_positive / truth_count) if truth_count else float("nan"),
        "true_positive_events": int(true_positive),
        "false_alarm_events": int(false_alarm),
        "truth_events": int(truth_count),
        "fa_per_24h": float(false_alarm * 24.0 / hours) if hours > 0 else float("nan"),
        "detection_delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
        "detection_delay_median_s": float(np.median(delays)) if delays else float("nan"),
        "nonseizure_hours": float(hours),
    }


def save_checkpoint(path: Path, adapter: CBraModAdapter, head: SharedContextHead, optimizer: torch.optim.Optimizer | None, method: str, patient: str, block: int) -> str:
    state = {
        "encoder": adapter.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "method": method,
        "patient_id": patient,
        "block": block,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "created_utc": now(),
    }
    torch.save(state, path)
    return sha256(path)


def run_patient(
    patient: str,
    method: str,
    windows: pd.DataFrame,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
    state: dict[str, Any],
    device: torch.device,
    seed: int,
    smoke_blocks: int | None = None,
    ttt_lr: float = 1e-5,
    supervised_lr: float = 1e-6,
    supervised_steps: int = 1,
    unfreeze_head: bool = False,
    supervised_max_positive: int = 4,
    supervised_max_negative: int = 4,
) -> dict[str, Any]:
    run_dir = OUT / "runs" / f"{method}__{patient}__seed{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    set_seed(seed)
    adapter, head = build_detector(state, method, device, unfreeze_head=unfreeze_head)
    initial_backbone_hash = tensor_hash(adapter.backbone)
    initial_projection_hash = tensor_hash(adapter.projection)
    initial_head_hash = tensor_hash(head)
    optimizer: torch.optim.Optimizer | None = None
    if method == "ttt":
        optimizer = torch.optim.AdamW(
            [p for p in adapter.backbone.parameters() if p.requires_grad],
            lr=ttt_lr,
            weight_decay=1e-5,
        )
    elif method == "supervised_oracle":
        optimizer = torch.optim.AdamW(
            [
                p
                for p in list(adapter.backbone.parameters())
                + list(adapter.projection.parameters())
                + list(head.parameters())
                if p.requires_grad
            ],
            lr=supervised_lr,
            weight_decay=1e-5,
        )
    probabilities: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    replay: list[tuple[np.ndarray, float]] = []
    adaptation_updates = 0
    scored_blocks = 0
    started = time.monotonic()
    patient_recordings = recordings[recordings.patient_id == patient].sort_values("recording_id", kind="stable")
    for recording in patient_recordings.recording_id.astype(str):
        rows = anchor_rows(windows, patient, recording)
        if len(rows) < 2:
            continue
        view = np.load(signal_path(str(patient_recordings.loc[patient_recordings.recording_id == recording, "relative_path"].iloc[0])), mmap_mode="r", allow_pickle=False)
        anchors = rows.start.to_numpy(dtype=float)
        block_ids = np.floor((anchors - WARMUP_S) / BLOCK_S).astype(int)
        for block_id in sorted(np.unique(block_ids)):
            if smoke_blocks is not None and scored_blocks >= smoke_blocks:
                break
            block_rows = rows.iloc[np.flatnonzero(block_ids == block_id)].copy()
            anchor_times = block_rows.start.to_numpy(dtype=float)
            min_start = float(anchor_times.min() - CONTEXT_HISTORY_S)
            max_start = float(anchor_times.max())
            unique_starts = np.arange(
                int(round(min_start / WINDOW_STEP_S)),
                int(round(max_start / WINDOW_STEP_S)) + 1,
                dtype=int,
            ) * WINDOW_STEP_S
            raw_unique = raw_windows_for_starts(view, unique_starts)
            index_by_start = {int(round(value / WINDOW_STEP_S)): idx for idx, value in enumerate(unique_starts)}
            contexts: list[np.ndarray] = []
            for anchor in anchor_times:
                context_start_index = int(round((anchor - CONTEXT_HISTORY_S) / WINDOW_STEP_S))
                context_indices = [index_by_start[context_start_index + i] for i in range(CONTEXT_WINDOWS)]
                contexts.append(raw_unique[context_indices])
            # The current block is always evaluated before any adaptation.
            probs = score_contexts(adapter, head, contexts, device)
            for row, probability in zip(block_rows.itertuples(index=False), probs, strict=True):
                probabilities.append(
                    {
                        "patient": patient,
                        "recording": recording,
                        "start_s": float(row.start),
                        "end_s": float(row.end),
                        "time_s": float(row.end),
                        "label": float(row.label) if pd.notna(row.label) else np.nan,
                        "probability": float(probability),
                        "block": int(block_id),
                        "scored_before_update": True,
                    }
                )
            loss = float("nan")
            grad_norm = float("nan")
            update_samples = 0
            if method == "ttt":
                # Adapt only after all current-block probabilities are stored.
                anchor_indices = [index_by_start[int(round(value / WINDOW_STEP_S))] for value in anchor_times]
                loss, grad_norm = ttt_update(adapter, optimizer, raw_unique[anchor_indices], device, np.random.default_rng(seed + 100000 + scored_blocks))
                adaptation_updates += 1
            elif method == "supervised_oracle":
                for context, row in zip(contexts, block_rows.itertuples(index=False), strict=True):
                    if pd.notna(row.label):
                        replay.append((context.copy(), float(row.label)))
                # Keep a bounded chronological replay; it contains no future
                # rows because it is appended only after the block was scored.
                replay = replay[-32:]
                loss, grad_norm, update_samples = supervised_update(
                    adapter,
                    head,
                    optimizer,
                    replay,
                    device,
                    steps=supervised_steps,
                    max_positive=supervised_max_positive,
                    max_negative=supervised_max_negative,
                )
                if update_samples:
                    adaptation_updates += 1
            scored_blocks += 1
            history.append(
                {
                    "recording": recording,
                    "block": int(block_id),
                    "anchors_scored": int(len(block_rows)),
                    "scored_before_update": True,
                    "adapted_after_score": method != "frozen",
                    "adaptation_update": int(adaptation_updates),
                    "loss": loss,
                    "grad_norm_pre_clip": grad_norm,
                    "update_samples": int(update_samples),
                    "elapsed_s": time.monotonic() - started,
                }
            )
            if method != "frozen" and adaptation_updates and adaptation_updates % 100 == 0:
                save_checkpoint(run_dir / f"checkpoint_block_{scored_blocks:05d}.pt", adapter, head, optimizer, method, patient, scored_blocks)
        if smoke_blocks is not None and scored_blocks >= smoke_blocks:
            break
        del view
    probability_table = pd.DataFrame(probabilities)
    probability_table.to_parquet(run_dir / "probabilities.parquet", index=False)
    metrics = score_events(probability_table, recordings, seizures, patient)
    checkpoint_hash = None
    if method != "frozen":
        checkpoint_hash = save_checkpoint(run_dir / "checkpoint.pt", adapter, head, optimizer, method, patient, scored_blocks)
    manifest = {
        "release_id": "cbramod-chb-same-patient-ttt-oracle-v1",
        "status": "smoke_complete" if smoke_blocks is not None else "complete",
        "dataset": "CHB-MIT",
        "outer_fold": 0,
        "patient_id": patient,
        "method": method,
        "seed": seed,
        "same_patient_upper_bound": method == "supervised_oracle",
        "transductive_online_adaptation": method == "ttt",
        "label_informed_adaptation": method == "supervised_oracle",
        "scoring_protocol": "score each 120-s chronological block before any update from that block",
        "context_windows": CONTEXT_WINDOWS,
        "context_history_s": CONTEXT_HISTORY_S,
        "anchor_step_s": ANCHOR_STEP_S,
        "adaptation_block_s": BLOCK_S,
        "threshold": THRESHOLD,
        "threshold_source": "frozen CHB validation-only split0 seed17 step5000",
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "pretrained_cbramod_sha256": sha256(PRETRAINED),
        "updates": int(adaptation_updates),
        "scored_blocks": int(scored_blocks),
        "scored_windows": int(len(probability_table)),
        "training_read_test_labels": method == "supervised_oracle",
        "head_frozen": not bool(unfreeze_head and method == "supervised_oracle"),
        "supervised_steps_per_update": int(supervised_steps),
        "supervised_max_positive": int(supervised_max_positive),
        "supervised_max_negative": int(supervised_max_negative),
        "initial_backbone_sha256": initial_backbone_hash,
        "initial_projection_sha256": initial_projection_hash,
        "initial_head_sha256": initial_head_hash,
        "final_backbone_sha256": tensor_hash(adapter.backbone),
        "final_projection_sha256": tensor_hash(adapter.projection),
        "final_head_sha256": tensor_hash(head),
        "head_unchanged": initial_head_hash == tensor_hash(head),
        "projection_unchanged": initial_projection_hash == tensor_hash(adapter.projection),
        "checkpoint": str((run_dir / "checkpoint.pt").relative_to(ROOT)) if checkpoint_hash else None,
        "checkpoint_sha256": checkpoint_hash,
        "probabilities": str((run_dir / "probabilities.parquet").relative_to(ROOT)),
        "metrics": metrics,
        "history": str((run_dir / "history.json").relative_to(ROOT)),
        "created_utc": now(),
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True) + "\n")
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(run_dir / "progress.json", {"status": manifest["status"], "patient_id": patient, "method": method, "scored_blocks": scored_blocks, "updates": adaptation_updates, "updated_utc": now()})
    del adapter, head
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return manifest


def create_manifest(selected: list[str], selection: dict[str, Any]) -> None:
    path = OUT / "experiment_manifest.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing manifest: {path}")
    payload = {
        "release_id": "cbramod-chb-same-patient-ttt-oracle-v1",
        "status": "frozen_before_training",
        "dataset": "CHB-MIT",
        "outer_fold": 0,
        "selected_patients": selected,
        "selection": selection,
        "methods": ["frozen", "ttt", "supervised_oracle"],
        "protocol": {
            "score_before_update": True,
            "ttt": "unlabeled masked-patch reconstruction; CBraMod backbone only",
            "supervised_oracle": "patient labels after score; backbone and projection updated; task head frozen",
            "current_block_labels_cannot_change_current_block_scores": True,
            "independent_generalization_claim": False,
        },
        "source_hashes": {
            "windows": sha256(WINDOWS),
            "recordings": sha256(RECORDINGS),
            "seizures": sha256(SEIZURES),
            "cv_manifest": sha256(CV_MANIFEST),
            "fold_file": sha256(FOLD_FILE),
            "source_checkpoint": sha256(SOURCE_CHECKPOINT),
            "pretrained": sha256(PRETRAINED),
        },
        "created_utc": now(),
    }
    atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["manifest", "smoke", "run"])
    parser.add_argument("--seed", type=int, choices=SEEDS, default=17)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke-blocks", type=int, default=25)
    parser.add_argument("--ttt-lr", type=float, default=1e-5)
    parser.add_argument("--supervised-lr", type=float, default=1e-6)
    parser.add_argument("--supervised-steps", type=int, default=1)
    parser.add_argument("--supervised-max-positive", type=int, default=4)
    parser.add_argument("--supervised-max-negative", type=int, default=4)
    parser.add_argument("--unfreeze-head", action="store_true")
    parser.add_argument("--patients", nargs="*", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    windows, recordings, seizures = load_tables()
    selected, selection = choose_patients(windows, recordings, seizures)
    if args.patients is not None and args.patients:
        if set(args.patients) != set(selected):
            raise RuntimeError(f"patients must equal frozen selection {selected}")
        selected = list(args.patients)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "manifest":
        create_manifest(selected, selection)
        print(json.dumps({"status": "frozen_before_training", "patients": selected, "out": str(OUT)}, indent=2))
        return
    if not (OUT / "experiment_manifest.json").exists():
        raise RuntimeError("run manifest is absent; run manifest command first")
    state = source_state()
    methods = ["frozen", "ttt", "supervised_oracle"]
    results: list[dict[str, Any]] = []
    for patient in selected:
        for method in methods:
            result = run_patient(
                patient,
                method,
                windows,
                recordings,
                seizures,
                state,
                device,
                args.seed,
                smoke_blocks=args.smoke_blocks if args.command == "smoke" else None,
                ttt_lr=args.ttt_lr,
                supervised_lr=args.supervised_lr,
                supervised_steps=args.supervised_steps,
                unfreeze_head=args.unfreeze_head,
                supervised_max_positive=args.supervised_max_positive,
                supervised_max_negative=args.supervised_max_negative,
            )
            results.append(result)
            print(json.dumps({"status": result["status"], "patient": patient, "method": method, "metrics": result["metrics"]}, sort_keys=True), flush=True)
    summary = {
        "release_id": "cbramod-chb-same-patient-ttt-oracle-v1",
        "status": "smoke_complete" if args.command == "smoke" else "complete",
        "patients": selected,
        "methods": methods,
        "seed": args.seed,
        "results": results,
        "created_utc": now(),
    }
    atomic_json(OUT / ("smoke_summary.json" if args.command == "smoke" else "summary.json"), summary)


if __name__ == "__main__":
    main()
