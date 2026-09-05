from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from .data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, load_rows, make_eval_loader
from .evaluate import SEIZURE_OFFSET_POST_S, SEIZURE_ONSET_PRE_S, eventize, match_events
from bfa.evaluation.eventize import Event
from .meta_model import CHBMetaTTTModel
from .model import CHBJointModel
from .meta_train import _batch_adapted_logits


CONDITIONS = {
    "supervised_frozen": ("detection_only", None),
    "meta_band_frozen": ("meta_band", None),
    "meta_band_ttt": ("meta_band", "band"),
    "meta_temporal_frozen": ("meta_temporal", None),
    "meta_temporal_ttt": ("meta_temporal", "temporal"),
}
BFA_ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SEIZURE_SCORING_REVISION = "event_collar_30s_60s_raw_alarm_time_v1"


def utc_now() -> str:
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


def module_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(args: argparse.Namespace, source: str, device: torch.device) -> tuple[torch.nn.Module, Path, dict[str, Any]]:
    checkpoint = args.output_root / "runs" / f"{source}_fold{args.fold}_seed{args.seed}" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if source == "detection_only":
        model: torch.nn.Module = CHBJointModel(args.pretrained)
    else:
        model = CHBMetaTTTModel(args.pretrained)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, state


@torch.inference_mode()
def frozen_probabilities(model: torch.nn.Module, loader, device: torch.device, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    seen = 0
    for signal, target, _ in loader:
        if limit is not None and seen >= limit:
            break
        if limit is not None:
            keep = min(len(signal), limit - seen)
            signal, target = signal[:keep], target[:keep]
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.detect(signal)
        labels.append(target.numpy())
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        seen += len(target)
    return np.concatenate(labels), np.concatenate(probabilities)


def ttt_probabilities(model: CHBMetaTTTModel, loader, device: torch.device, objective: str, limit: int | None, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    seen = 0
    alpha_tensor = torch.tensor(float(alpha), device=device)
    source_hash = module_hash(model)
    for signal, target, sample_ids in loader:
        if limit is not None and seen >= limit:
            break
        if limit is not None:
            keep = min(len(signal), limit - seen)
            signal, target, sample_ids = signal[:keep], target[:keep], list(sample_ids[:keep])
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.enable_grad():
            logits = _batch_adapted_logits(model, signal, list(sample_ids), objective, alpha_tensor)
        labels.append(target.numpy())
        probabilities.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
        seen += len(target)
    if module_hash(model) != source_hash:
        raise RuntimeError("functional TTT mutated source parameters")
    return np.concatenate(labels), np.concatenate(probabilities)


def _union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    total = 0.0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def _scoring_truths(
    seizures: pd.DataFrame,
    recording_id: str,
    evaluation_start: float,
    duration: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return event-scoring and raw seizure intervals.

    The raw seizure annotations remain unchanged.  For validation threshold
    selection and for the single test evaluation, event matching uses the
    interval ``[onset - 30 s, offset + 60 s]`` clipped to the evaluable part
    of the EDF.  The paired raw interval is retained for alarm-time
    subtraction and delay measurement.  Keeping both intervals prevents an
    onset/offset collar from making a long predicted alarm appear shorter.
    Training windows and their labels are not altered.
    """
    expanded: list[tuple[float, float]] = []
    raw_output: list[tuple[float, float]] = []
    rows = seizures[seizures.recording_id == recording_id]
    for row in rows.itertuples(index=False):
        raw_start = float(row.start_s)
        raw_end = float(row.end_s)
        # Keep the original evaluator's event inclusion rule: an annotation
        # that ended before evaluation began is not counted merely because a
        # post-offset collar would extend into the evaluated region.
        if raw_end <= evaluation_start:
            continue
        raw_start_clipped = max(evaluation_start, raw_start)
        raw_end_clipped = min(duration, raw_end)
        start = max(evaluation_start, raw_start - SEIZURE_ONSET_PRE_S)
        end = min(duration, raw_end + SEIZURE_OFFSET_POST_S)
        if raw_end_clipped > raw_start_clipped and end > start:
            expanded.append((start, end))
            raw_output.append((raw_start_clipped, raw_end_clipped))
    return expanded, raw_output


def score_probabilities(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, threshold: float) -> dict[str, Any]:
    true_positives = false_alarms = truth_events = 0
    false_alarm_seconds = 0.0
    total_monitoring_seconds = 0.0
    nonseizure_seconds = 0.0
    delays: list[float] = []
    for recording_id, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        evaluation_start = float(ordered.end.iloc[0])
        recording = recordings[recordings.recording_id == recording_id]
        if recording.empty:
            raise ValueError(f"missing recording metadata: {recording_id}")
        duration = float(recording.duration_s.iloc[0])
        truths, raw_truths = _scoring_truths(seizures, recording_id, evaluation_start, duration)
        raw_predictions = eventize(ordered.end.to_numpy(dtype=float), ordered.probability.to_numpy(dtype=float), threshold=threshold)
        predictions = [
            Event(max(0.0, float(prediction.start_s)), min(duration, float(prediction.end_s)), float(prediction.peak_probability))
            for prediction in raw_predictions
            if min(duration, float(prediction.end_s)) > max(0.0, float(prediction.start_s))
        ]
        matched = match_events(predictions, truths)
        true_positives += len(matched.pairs)
        false_alarms += len(matched.unmatched_predictions)
        truth_events += len(truths)
        # Alarm duration is measured against the original seizure annotation,
        # not the permissive event-matching collar.
        truth_union = _union_length(raw_truths)
        for prediction in predictions:
            overlap = _union_length([
                (max(prediction.start_s, start), min(prediction.end_s, end))
                for start, end in raw_truths
                if min(prediction.end_s, end) > max(prediction.start_s, start)
            ])
            false_alarm_seconds += max(0.0, prediction.end_s - prediction.start_s - overlap)
        for pair in matched.pairs:
            delays.append(max(0.0, predictions[pair.prediction_index].start_s - raw_truths[pair.truth_index][0]))
        total_monitoring_seconds += max(0.0, duration - evaluation_start)
        nonseizure_seconds += max(0.0, duration - evaluation_start - truth_union)
    total_hours = total_monitoring_seconds / 3600.0
    nonseizure_hours = nonseizure_seconds / 3600.0
    return {
        "threshold": float(threshold),
        "seizure_scoring_revision": SEIZURE_SCORING_REVISION,
        "seizure_scoring_collar_onset_pre_s": SEIZURE_ONSET_PRE_S,
        "seizure_scoring_collar_offset_post_s": SEIZURE_OFFSET_POST_S,
        "alarm_time_truth_definition": "raw seizure interval; collar excluded from alarm-time subtraction",
        "false_alarm_event_rate_definition": "event matching uses collar-expanded truths; denominator uses raw non-seizure monitoring time",
        "true_positive_events": int(true_positives),
        "false_alarm_events": int(false_alarms),
        "truth_events": int(truth_events),
        "event_sensitivity": float(true_positives / truth_events) if truth_events else 1.0,
        "false_alarm_time_seconds": float(false_alarm_seconds),
        "false_alarm_time_min_per_24h": float(false_alarm_seconds / 60.0 * 24.0 / total_hours) if total_hours else 0.0,
        "false_alarm_time_s_per_24h": float(false_alarm_seconds * 24.0 / total_hours) if total_hours else 0.0,
        "fa_per_24h": float(false_alarms * 24.0 / nonseizure_hours) if nonseizure_hours else 0.0,
        "total_monitoring_hours": float(total_hours),
        "nonseizure_hours": float(nonseizure_hours),
        "detection_delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
        "detection_delay_median_s": float(np.median(delays)) if delays else float("nan"),
        "detection_delay_sum_s": float(np.sum(delays)) if delays else 0.0,
        "detection_delay_count": int(len(delays)),
    }


def window_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "window_auroc": float(roc_auc_score(y, p)),
        "window_auprc": float(average_precision_score(y, p)),
        "window_balanced_accuracy_0p5": float(balanced_accuracy_score(y, p >= 0.5)),
    }


def select_validation_threshold(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [score_probabilities(table, seizures, recordings, float(threshold)) for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    feasible = [row for row in rows if row["event_sensitivity"] >= 0.80]
    if feasible:
        selected = min(feasible, key=lambda row: (row["false_alarm_time_min_per_24h"], row["fa_per_24h"], row["detection_delay_mean_s"] if np.isfinite(row["detection_delay_mean_s"]) else float("inf"), -row["threshold"]))
    else:
        selected = min(rows, key=lambda row: (-row["event_sensitivity"], row["false_alarm_time_min_per_24h"], row["fa_per_24h"], -row["threshold"]))
    return pd.DataFrame(rows), selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/meta-ttt-chbmit-5fold-v1')
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parents[1] / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--recordings", type=Path, default=BFA_ROOT / "manifests/recordings.parquet")
    parser.add_argument("--seizures", type=Path, default=BFA_ROOT / "manifests/seizures.parquet")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 3407:
        raise ValueError("formal v1 locks seed=3407")
    if args.split == "test" and not args.allow_test:
        raise PermissionError("test requires --allow-test after validation lock")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source, objective = CONDITIONS[args.condition]
    model, checkpoint, checkpoint_state = load_model(args, source, device)
    alpha = float(checkpoint_state.get("alpha", 1e-4)) if objective else 0.0
    rows = load_rows(args.fold, args.split, args.windows, args.fold_root)
    if args.limit is not None:
        rng = np.random.default_rng(args.seed + args.fold + 30_000)
        parts = []
        for label in (0.0, 1.0):
            group = rows[rows.label == label]
            count = min(len(group), max(1, args.limit // 2))
            parts.append(group.iloc[np.sort(rng.choice(len(group), size=count, replace=False))])
        rows = pd.concat(parts, ignore_index=True).sort_values(["patient", "recording", "start"], kind="stable").reset_index(drop=True)
    loader = make_eval_loader(rows, batch_size=args.batch_size if objective is None else min(args.batch_size, 64), workers=args.workers, cache_root=args.cache_root)
    result_dir = args.output_root / "evaluation" / args.condition / f"fold{args.fold}_seed{args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    probability_path = result_dir / f"{args.split}_probabilities.parquet"
    if probability_path.exists():
        raise FileExistsError(probability_path)
    started = time.monotonic()
    if objective:
        y, probability = ttt_probabilities(model, loader, device, objective, args.limit, alpha)
    else:
        y, probability = frozen_probabilities(model, loader, device, args.limit)
    if len(probability) != len(rows) and args.limit is None:
        raise RuntimeError("probability count mismatch")
    used_rows = rows.iloc[: len(probability)].copy()
    table = used_rows[["patient", "recording", "start", "end", "label", "sample_id", "relative_path"]].copy()
    table["probability"] = probability
    table.to_parquet(probability_path, index=False)
    recordings = pd.read_parquet(args.recordings)
    seizures = pd.read_parquet(args.seizures)
    base = {
        "release_id": "meta-ttt-chbmit-5fold-v1", "condition": args.condition,
        "source_condition": source, "ttt_objective": objective, "fold": args.fold, "seed": args.seed,
        "split": args.split, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "probability_sha256": sha256(probability_path), "rows": len(table),
        "elapsed_s": time.monotonic() - started, "window_metrics": window_metrics(table.label.to_numpy(dtype=int), table.probability.to_numpy(dtype=float)),
        "ttt_steps": 1 if objective else 0, "ttt_optimizer": "SGD" if objective else None,
        "ttt_lr": alpha if objective else None, "episodic_restore": bool(objective),
        "threshold_metric": "false_alarm_time_min_per_24h_then_fa_per_24h",
        "seizure_scoring_revision": SEIZURE_SCORING_REVISION,
        "seizure_scoring_truth_definition": "raw interval expanded to onset-30s and offset+60s for event matching, clipped to evaluable EDF",
        "alarm_time_truth_definition": "raw seizure interval for false-alarm time and its total-monitoring-time rate",
        "detection_delay_reference": "raw onset, clipped to evaluation start",
        "seizure_scoring_collar_onset_pre_s": SEIZURE_ONSET_PRE_S,
        "seizure_scoring_collar_offset_post_s": SEIZURE_OFFSET_POST_S,
    }
    if args.split == "validation":
        sweep, selected = select_validation_threshold(table, seizures, recordings) if args.limit is None else (pd.DataFrame(), {"threshold": 0.5, "smoke_only": True})
        sweep_path = result_dir / "validation_threshold_sweep.parquet"
        sweep.to_parquet(sweep_path, index=False)
        payload = {**base, "status": "validation_threshold_frozen", "selected_event_operating_point": selected, "threshold_sweep_sha256": sha256(sweep_path), "test_evaluation_count": 0, "threshold_source": "validation_only"}
        atomic_json(result_dir / "validation_metrics.json", payload)
    else:
        lock_path = result_dir / "validation_metrics.json"
        if not lock_path.is_file():
            raise FileNotFoundError(lock_path)
        lock = json.loads(lock_path.read_text())
        if lock.get("checkpoint_sha256") != sha256(checkpoint) or lock.get("threshold_source") != "validation_only":
            raise ValueError("validation lock/checkpoint mismatch")
        threshold = float(lock["selected_event_operating_point"]["threshold"])
        metric = score_probabilities(table, seizures, recordings, threshold)
        waterfall = pd.DataFrame([{"patient": patient, **score_probabilities(group, seizures, recordings, threshold)} for patient, group in table.groupby("patient", sort=True)])
        waterfall_path = result_dir / "test_patient_waterfall.csv"
        waterfall.to_csv(waterfall_path, index=False)
        atomic_json(result_dir / "test_completed.json", {**base, "status": "test_complete", "threshold_source": "validation_only", "selected_event_operating_point": metric, "patient_waterfall_sha256": sha256(waterfall_path), "test_evaluation_count": 1})
    return base


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
