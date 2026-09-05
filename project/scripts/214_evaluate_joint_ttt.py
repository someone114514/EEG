"""Evaluate a trained Joint- or Meta-TTT CBraMod checkpoint without target labels.

Validation is run first to choose one event threshold per fold/seed.  The
outer-test patients are then streamed once with the threshold frozen.  During
adaptation only the masked-patch reconstruction loss is used; seizure labels
are read only after the stream for metric calculation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
JOINT_NAMESPACE = os.environ.get("JOINT_TTT_NAMESPACE", "cbramod-joint-ttt-v1-formal")
OUT = ROOT / "outputs/reports" / JOINT_NAMESPACE / "evaluation"
SEEDS = (17, 42, 3407)


def device_from_arg(value: str | None) -> torch.device:
    """Resolve the evaluation device without relying on the adaptation module."""
    device = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_adaptation_module():
    path = ROOT / "scripts/202_cbramod_same_patient_adaptation.py"
    spec = importlib.util.spec_from_file_location("same_patient_adaptation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def load_ttt_checkpoint(module, fold: int, seed: int):
    path = ROOT / "outputs/reports" / JOINT_NAMESPACE / "runs" / f"fold{fold}_seed{seed}" / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state.get("update", -1)) != 5000:
        raise ValueError(f"TTT checkpoint is not step 5000: {path}")
    adapter = module.CBraModAdapter(module.PRETRAINED, train_backbone=True)
    adapter.load_state_dict(state["encoder"], strict=True)
    head = module.SharedContextHead()
    head.load_state_dict(state["head"], strict=True)
    return path, state, adapter, head


def fold_patients(fold: int, split: str) -> list[str]:
    manifest = json.loads((ROOT / "manifests/groupkfold_cv_v1" / f"fold_{fold}.json").read_text())
    return [str(x) for x in manifest[split]]


def configure_adaptation(adapter, method: str) -> str:
    """Set the label-free adaptation scope to match the training method."""
    if method == "joint":
        adapter.backbone.requires_grad_(True)
        return "full_backbone"
    if method == "meta":
        # Meta training only exposes the final two transformer blocks during
        # the inner support update.  Keep that exact scope at evaluation time;
        # adapting the full backbone would be a different experiment.
        adapter.backbone.requires_grad_(False)
        for name, parameter in adapter.backbone.named_parameters():
            parameter.requires_grad_(
                name.startswith("encoder.layers.10.")
                or name.startswith("encoder.layers.11.")
            )
        return "last_two_backbone_blocks"
    raise ValueError(f"unknown TTT method: {method}")


@torch.inference_mode()
def score_block_fast(module, adapter, head, view: np.ndarray, anchor_times: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Score one causal block while encoding every unique 10-s window once.

    Adjacent 70-s contexts overlap heavily: a 120-s block normally contains
    12 anchors but only 86 unique 10-s windows, versus 12 * 31 = 372 window
    encodings in the legacy path.  The backbone is frozen *within* a scored
    block, so reusing those embeddings is mathematically equivalent and does
    not change the score-before-update protocol.
    """
    adapter.eval(); head.eval()
    anchor_times = np.asarray(anchor_times, dtype=np.float64)
    offsets = (
        -float(module.CONTEXT_HISTORY_S)
        + np.arange(module.CONTEXT_WINDOWS, dtype=np.float64) * float(module.WINDOW_STEP_S)
    )
    start_samples = np.rint(
        (anchor_times[:, None] + offsets[None, :]) * int(module.MODEL_RATE)
    ).astype(np.int64)
    unique_samples, inverse = np.unique(start_samples.reshape(-1), return_inverse=True)
    window_samples = int(module.WINDOW_S * module.MODEL_RATE)
    if unique_samples.min(initial=0) < 0 or (unique_samples + window_samples).max(initial=0) > view.shape[-1]:
        raise ValueError(f"context outside signal: {anchor_times.min()}..{anchor_times.max()} / {view.shape}")
    raw_unique = np.stack(
        [view[:, left : left + window_samples] for left in unique_samples], axis=0
    ).astype(np.float32, copy=False)

    # One block has about 86 unique windows.  Keep the knob configurable for
    # smaller GPUs while using the available 32-GB device efficiently.
    window_batch = max(1, int(os.environ.get("JOINT_TTT_SCORE_WINDOW_BATCH", "128")))
    projected_parts: list[torch.Tensor] = []
    for begin in range(0, len(raw_unique), window_batch):
        batch = torch.from_numpy(np.ascontiguousarray(raw_unique[begin : begin + window_batch])).to(
            device=device, dtype=torch.float32
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features = adapter.backbone_features(
                batch.reshape(-1, 16, module.WINDOW_S, module.MODEL_RATE)
            )
            projected_parts.append(adapter.projection(features))
    projected_unique = torch.cat(projected_parts, dim=0)
    inverse_tensor = torch.as_tensor(inverse, device=device, dtype=torch.long)
    projected_contexts = projected_unique.index_select(0, inverse_tensor).reshape(
        len(anchor_times), module.CONTEXT_WINDOWS, 16, 128
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        probabilities = torch.sigmoid(head(projected_contexts))

    # The final member of each context is the current 10-s causal window and
    # is the exact input used by the prequential reconstruction update.
    last_indices = inverse.reshape(len(anchor_times), module.CONTEXT_WINDOWS)[:, -1]
    current_raw = np.ascontiguousarray(raw_unique[last_indices])
    return probabilities.float().cpu().numpy().astype(np.float32, copy=False), current_raw


def stream_patient(module, adapter, head, patient: str, windows, recordings, device, *, adapt: bool, threshold: float, seed: int, update_after_score: bool, method: str) -> pd.DataFrame:
    adapter.to(device); head.to(device)
    adapter.eval(); head.eval()
    optimizer = None
    if adapt:
        configure_adaptation(adapter, method)
        adapter.projection.requires_grad_(False)
        head.requires_grad_(False)
        trainable = [p for p in adapter.backbone.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"no trainable backbone parameters for method={method}")
        optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=1e-5)
    rows_out: list[dict[str, Any]] = []
    recording_rows = recordings[recordings.patient_id.astype(str) == str(patient)].sort_values("recording_id", kind="stable")
    rng = np.random.default_rng(seed + 200000)
    for recording in recording_rows.recording_id.astype(str):
        anchors = module.anchor_rows(windows, patient, recording)
        if len(anchors) == 0:
            continue
        relative = str(recording_rows.loc[recording_rows.recording_id.astype(str) == recording, "relative_path"].iloc[0])
        view = np.load(module.signal_path(relative), mmap_mode="r", allow_pickle=False)
        anchor_times = anchors.start.to_numpy(dtype=float)
        block_ids = np.floor((anchor_times - module.WARMUP_S) / module.BLOCK_S).astype(int)
        for block_id in sorted(np.unique(block_ids)):
            block_rows = anchors.iloc[np.flatnonzero(block_ids == block_id)].copy()
            probabilities, raw = score_block_fast(
                module, adapter, head, view, block_rows.start.to_numpy(float), device
            )
            # Standard transductive TTT adapts on the already observed causal
            # block before scoring it.  ``update_after_score`` is the stricter
            # prequential sensitivity analysis and updates only after scores
            # have been stored.
            if adapt and not update_after_score:
                module.ttt_update(adapter, optimizer, raw, device, rng, micro_windows=len(raw))
            for row, probability in zip(block_rows.itertuples(index=False), probabilities, strict=True):
                rows_out.append({"patient": patient, "recording": recording, "start_s": float(row.start), "end_s": float(row.end), "time_s": float(row.end), "label": float(row.label) if pd.notna(row.label) else np.nan, "probability": float(probability), "block": int(block_id), "threshold": float(threshold), "adapted_after_score": bool(adapt and update_after_score)})
            if adapt and update_after_score:
                module.ttt_update(adapter, optimizer, raw, device, rng, micro_windows=len(raw))
        del view
    return pd.DataFrame(rows_out)


def aggregate_metrics(module, table: pd.DataFrame, recordings: pd.DataFrame, seizures: pd.DataFrame) -> dict[str, float]:
    """Aggregate event counts across validation patients at one threshold."""
    true_positive = false_alarm = truth_count = 0
    nonseizure_hours = 0.0
    delays: list[float] = []
    for patient, group in table.groupby("patient", sort=True):
        metric = module.score_events(group, recordings, seizures, str(patient))
        true_positive += int(metric["true_positive_events"])
        false_alarm += int(metric["false_alarm_events"])
        truth_count += int(metric["truth_events"])
        nonseizure_hours += float(metric["nonseizure_hours"])
        if np.isfinite(metric["detection_delay_mean_s"]):
            delays.append(float(metric["detection_delay_mean_s"]))
    return {
        "true_positive_events": float(true_positive),
        "false_alarm_events": float(false_alarm),
        "truth_events": float(truth_count),
        "event_sensitivity": float(true_positive / truth_count) if truth_count else float("nan"),
        "fa_per_24h": float(false_alarm * 24.0 / nonseizure_hours) if nonseizure_hours > 0 else float("nan"),
        "nonseizure_hours": float(nonseizure_hours),
        "detection_delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
    }


def select_threshold(module, table: pd.DataFrame, recordings: pd.DataFrame, seizures: pd.DataFrame, baseline_table: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.concatenate([np.linspace(0.001, 0.05, 20), np.linspace(0.06, 0.99, 48)]))
    baseline = aggregate_metrics(module, baseline_table, recordings, seizures)
    allowed_sensitivity = float(baseline["event_sensitivity"]) - 0.03
    sweep: list[dict[str, Any]] = []
    for threshold in candidates:
        module.THRESHOLD = float(threshold)
        metric = aggregate_metrics(module, table, recordings, seizures)
        sweep.append({"threshold": float(threshold), **metric})
    valid = [row for row in sweep if np.isfinite(row["event_sensitivity"]) and row["event_sensitivity"] >= allowed_sensitivity]
    if not valid:
        best = max(sweep, key=lambda row: (row["event_sensitivity"], -row["fa_per_24h"]))
    else:
        best = min(valid, key=lambda row: (row["fa_per_24h"], -row["event_sensitivity"], row["threshold"]))
    return float(best["threshold"]), {"baseline": baseline, "allowed_sensitivity": allowed_sensitivity, "selected": best, "sweep": sweep}


def run(args: argparse.Namespace) -> None:
    if args.fold not in range(5) or args.seed not in SEEDS:
        raise ValueError("invalid fold or seed")
    module = load_adaptation_module(); windows, recordings, seizures = module.load_tables(); device = device_from_arg(args.device)
    method = args.method
    checkpoint, state, _, _ = load_ttt_checkpoint(module, args.fold, args.seed)
    validation_patients = fold_patients(args.fold, "validation"); test_patients = fold_patients(args.fold, "test")
    validation_rows: list[dict[str, Any]] = []; validation_baseline: list[dict[str, Any]] = []; validation_frozen_tables: list[pd.DataFrame] = []; validation_adapted_tables: list[pd.DataFrame] = []
    for patient in validation_patients:
        _, _, frozen_adapter, frozen_head = load_ttt_checkpoint(module, args.fold, args.seed)
        baseline = stream_patient(module, frozen_adapter, frozen_head, patient, windows, recordings, device, adapt=False, threshold=0.01, seed=args.seed, update_after_score=False, method=method)
        module.THRESHOLD = 0.01; base_metric = module.score_events(baseline, recordings, seizures, patient); validation_baseline.append({"patient": patient, **base_metric}); validation_frozen_tables.append(baseline)
        _, _, adapt_adapter, adapt_head = load_ttt_checkpoint(module, args.fold, args.seed)
        adapted = stream_patient(module, adapt_adapter, adapt_head, patient, windows, recordings, device, adapt=True, threshold=0.01, seed=args.seed, update_after_score=args.update_after_score, method=method)
        validation_adapted_tables.append(adapted)
    if not validation_adapted_tables:
        raise RuntimeError("no validation patients produced rows")
    validation_frozen = pd.concat(validation_frozen_tables, ignore_index=True)
    validation_adapted = pd.concat(validation_adapted_tables, ignore_index=True)
    threshold, threshold_details = select_threshold(module, validation_adapted, recordings, seizures, validation_frozen)
    module.THRESHOLD = threshold
    validation_rows = validation_adapted.assign(threshold=threshold).to_dict(orient="records")
    test_rows: list[dict[str, Any]] = []; test_metrics: list[dict[str, Any]] = []; test_frozen_metrics: list[dict[str, Any]] = []
    module.THRESHOLD = threshold
    for patient in test_patients:
        # Score the same frozen checkpoint once as a paired comparator.  This
        # does not alter selection: the threshold was fixed on validation
        # above, and test labels are read only to compute the final metric.
        _, _, frozen_adapter, frozen_head = load_ttt_checkpoint(module, args.fold, args.seed)
        frozen_table = stream_patient(module, frozen_adapter, frozen_head, patient, windows, recordings, device, adapt=False, threshold=threshold, seed=args.seed, update_after_score=False, method=method)
        frozen_table["threshold"] = threshold
        test_frozen_metrics.append({"patient": patient, **module.score_events(frozen_table, recordings, seizures, patient)})
        _, _, adapt_adapter, adapt_head = load_ttt_checkpoint(module, args.fold, args.seed)
        test_table = stream_patient(module, adapt_adapter, adapt_head, patient, windows, recordings, device, adapt=True, threshold=threshold, seed=args.seed, update_after_score=args.update_after_score, method=method)
        test_table["threshold"] = threshold
        test_rows.extend(test_table.to_dict(orient="records"))
        test_metrics.append({"patient": patient, **module.score_events(test_table, recordings, seizures, patient)})
    evaluation_dir = OUT / f"fold{args.fold}_seed{args.seed}"; evaluation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_baseline).to_csv(evaluation_dir / "validation_frozen_metrics.csv", index=False)
    (evaluation_dir / "validation_threshold_details.json").write_text(json.dumps(threshold_details, indent=2, sort_keys=True, allow_nan=True) + "\n")
    pd.DataFrame(validation_rows).to_parquet(evaluation_dir / "validation_adapted_probabilities.parquet", index=False)
    pd.DataFrame(test_frozen_metrics).to_csv(evaluation_dir / "test_frozen_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_parquet(evaluation_dir / "test_adapted_probabilities.parquet", index=False)
    pd.DataFrame(test_metrics).to_csv(evaluation_dir / "test_metrics.csv", index=False)
    manifest = {"release_id": f"cbramod-{method}-ttt-v1-evaluation", "namespace": JOINT_NAMESPACE, "ttt_method": method, "adaptation_trainable_scope": "full_backbone" if method == "joint" else "last_two_backbone_blocks", "fold": args.fold, "seed": args.seed, "source_checkpoint": str(checkpoint.relative_to(ROOT)), "source_checkpoint_sha256": module.sha256(checkpoint), "validation_patients": validation_patients, "test_patients": test_patients, "adaptation_objective": "masked-patch reconstruction only; no target labels", "update_after_score": bool(args.update_after_score), "threshold": threshold, "threshold_source": "validation-only median of patient validation selections", "test_evaluation_count_per_condition": 1, "test_labels_used_for_adaptation": False, "test_frozen_metrics": test_frozen_metrics, "test_adapted_metrics": test_metrics, "outputs": ["validation_frozen_metrics.csv", "validation_threshold_details.json", "validation_adapted_probabilities.parquet", "test_frozen_metrics.csv", "test_adapted_probabilities.parquet", "test_metrics.csv"], "created_utc": module.now()}
    atomic_json(evaluation_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", **manifest}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--fold", type=int, required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--device"); parser.add_argument("--method", choices=("joint", "meta"), default=os.environ.get("TTT_METHOD", "joint")); parser.add_argument("--update-after-score", action="store_true"); args = parser.parse_args(); run(args)


if __name__ == "__main__": main()
