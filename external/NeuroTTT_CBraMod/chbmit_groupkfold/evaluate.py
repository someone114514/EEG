from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.nn import functional as F
from torch.func import functional_call, grad, vmap

from .data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, load_rows, make_eval_loader
from .model import CHBJointModel
from .transforms import deterministic_band_view, deterministic_patch_mask

BFA_ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
if str(BFA_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BFA_ROOT / "src"))
from bfa.evaluation.eventize import eventize  # noqa: E402
from bfa.evaluation.match import match_events  # noqa: E402

CONDITIONS = {
    "supervised_frozen": ("detection_only", None),
    "band_joint_frozen": ("band_joint", None),
    "band_joint_band_ttt": ("band_joint", "band"),
    "mask_joint_frozen": ("mask_joint", None),
    "mask_joint_mask_ttt": ("mask_joint", "mask"),
}

# Event-level seizure scoring collar.  The collar is applied identically to
# validation threshold selection, test scoring, and patient bootstrap; it is
# not used to alter the detector windows or training labels.
SEIZURE_ONSET_PRE_S = 30.0
SEIZURE_OFFSET_POST_S = 60.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
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
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_model(args, source_condition: str, device: torch.device) -> tuple[CHBJointModel, Path]:
    run = args.output_root / "runs" / f"{source_condition}_fold{args.fold}_seed{args.seed}"
    checkpoint = run / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("condition") != source_condition or int(state.get("fold", -1)) != args.fold:
        raise ValueError("checkpoint identity mismatch")
    model = CHBJointModel(args.pretrained)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), checkpoint


@torch.inference_mode()
def frozen_probabilities(model: CHBJointModel, loader, device: torch.device, limit: int | None) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    seen = 0
    for signal, _, _ in loader:
        if limit is not None and seen >= limit:
            break
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.detect(signal)
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
        seen += len(signal)
    values = np.concatenate(output)
    return values[:limit]


def scalar_ttt_batch(model: CHBJointModel, signal: torch.Tensor, sample_ids: list[str], objective: str, lr: float) -> np.ndarray:
    """Exact upstream episodic same-sample update; deliberately batch-size one."""
    source_hash = module_hash(model)
    probabilities: list[float] = []
    for index, sample_id in enumerate(sample_ids):
        x = signal[index : index + 1]
        adaptive_ids = {id(parameter) for parameter in model.adaptive_parameters(objective)}
        saved = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if id(parameter) in adaptive_ids}
        for parameter in model.detector.parameters():
            parameter.requires_grad_(False)
        if objective == "band":
            for parameter in model.backbone.proj_out.parameters():
                parameter.requires_grad_(False)
        else:
            for parameter in model.band_head.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.SGD(model.adaptive_parameters(objective), lr=lr)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if objective == "band":
            filtered, label = deterministic_band_view(x, [sample_id])
            loss = F.cross_entropy(model(filtered, mode="band"), label)
        else:
            mask = deterministic_patch_mask([sample_id], x.shape[1], x.shape[2], 0.5, x.device)
            reconstruction = model(x, mode="mask", mask=mask)
            expanded = mask.unsqueeze(-1).expand_as(reconstruction)
            loss = F.mse_loss(reconstruction[expanded], x[expanded])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            probabilities.append(float(torch.sigmoid(model.detect(x).float()).item()))
        named = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in saved.items():
                named[name].copy_(value)
        model.zero_grad(set_to_none=True)
    if module_hash(model) != source_hash:
        raise RuntimeError("episodic TTT failed to restore source parameters")
    return np.asarray(probabilities, dtype=np.float32)


class FastScalarEpisodicTTT:
    """Exact scalar TTT with one persistent source snapshot and manual SGD."""

    def __init__(self, model: CHBJointModel, objective: str, lr: float) -> None:
        self.model = model
        self.objective = objective
        self.lr = float(lr)
        for parameter in model.detector.parameters():
            parameter.requires_grad_(False)
        if objective == "band":
            for parameter in model.backbone.proj_out.parameters():
                parameter.requires_grad_(False)
        else:
            for parameter in model.band_head.parameters():
                parameter.requires_grad_(False)
        self.parameters = model.adaptive_parameters(objective)
        self.source = [parameter.detach().clone() for parameter in self.parameters]
        self.source_hash = module_hash(model)
        self.samples = 0

    def _restore(self) -> None:
        with torch.no_grad():
            for parameter, source in zip(self.parameters, self.source, strict=True):
                parameter.copy_(source)
                parameter.grad = None

    def batch(self, signal: torch.Tensor, sample_ids: list[str]) -> np.ndarray:
        probabilities: list[float] = []
        for index, sample_id in enumerate(sample_ids):
            x = signal[index : index + 1]
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            if self.objective == "band":
                transformed, label = deterministic_band_view(x, [sample_id])
                loss = F.cross_entropy(self.model(transformed, mode="band"), label)
            else:
                mask = deterministic_patch_mask([sample_id], x.shape[1], x.shape[2], 0.5, x.device)
                reconstruction = self.model(x, mode="mask", mask=mask)
                expanded = mask.unsqueeze(-1).expand_as(reconstruction)
                loss = F.mse_loss(reconstruction[expanded], x[expanded])
            loss.backward()
            with torch.no_grad():
                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-self.lr)
            self.model.eval()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
                probabilities.append(float(torch.sigmoid(self.model.detect(x).float()).item()))
            self._restore()
            self.samples += 1
        return np.asarray(probabilities, dtype=np.float32)

    def finalize(self) -> None:
        self._restore()
        if module_hash(self.model) != self.source_hash:
            raise RuntimeError("fast scalar TTT failed to restore source parameters")


def adaptive_named_parameters(model: CHBJointModel, objective: str) -> dict[str, torch.Tensor]:
    prefixes = ["backbone.patch_embedding.", "backbone.encoder."]
    prefixes.append("band_head." if objective == "band" else "backbone.proj_out.")
    return {name: parameter for name, parameter in model.named_parameters() if any(name.startswith(prefix) for prefix in prefixes)}


def vmap_ttt_batch(model: CHBJointModel, signal: torch.Tensor, sample_ids: list[str], objective: str, lr: float) -> np.ndarray:
    """Parallel independent one-step SGD without modifying the source module."""
    source_hash = module_hash(model)
    parameters = adaptive_named_parameters(model, objective)
    model.train()
    if objective == "band":
        transformed, labels = deterministic_band_view(signal, sample_ids)

        def ssl_loss(current, sample, label):
            logits = functional_call(model, current, (sample.unsqueeze(0),), {"mode": "band"}, strict=False)
            return F.cross_entropy(logits, label.unsqueeze(0))

        task_gradients = vmap(grad(ssl_loss), in_dims=(None, 0, 0), randomness="different")(
            parameters, transformed, labels
        )
    else:
        masks = deterministic_patch_mask(sample_ids, signal.shape[1], signal.shape[2], 0.5, signal.device)

        def ssl_loss(current, sample, mask):
            reconstruction = functional_call(
                model,
                current,
                (sample.unsqueeze(0),),
                {"mode": "mask", "mask": mask.unsqueeze(0)},
                strict=False,
            ).squeeze(0)
            expanded = mask.unsqueeze(-1).to(reconstruction.dtype)
            return ((reconstruction - sample).square() * expanded).sum() / expanded.sum().clamp_min(1)

        task_gradients = vmap(grad(ssl_loss), in_dims=(None, 0, 0), randomness="different")(
            parameters, signal, masks
        )
    adapted = {
        name: parameter.unsqueeze(0) - lr * task_gradients[name]
        for name, parameter in parameters.items()
    }
    model.eval()

    def predict(current, sample):
        return functional_call(
            model, current, (sample.unsqueeze(0),), {"mode": "detect"}, strict=False
        ).squeeze(0)

    logits = vmap(predict, in_dims=(0, 0), randomness="same")(adapted, signal)
    if module_hash(model) != source_hash:
        raise RuntimeError("functional episodic TTT mutated source parameters")
    return torch.sigmoid(logits.float()).detach().cpu().numpy().astype(np.float32, copy=False)


def ttt_probabilities(model: CHBJointModel, loader, device: torch.device, objective: str, limit: int | None, lr: float, engine: str) -> np.ndarray:
    output: list[np.ndarray] = []
    seen = 0
    fast_scalar = FastScalarEpisodicTTT(model, objective, lr) if engine == "scalar" else None
    for signal, _, sample_ids in loader:
        if limit is not None and seen >= limit:
            break
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        if limit is not None:
            keep = min(len(signal), limit - seen)
            signal = signal[:keep]
            sample_ids = list(sample_ids[:keep])
        if engine == "scalar":
            values = fast_scalar.batch(signal, list(sample_ids))
        elif engine == "vmap":
            values = vmap_ttt_batch(model, signal, list(sample_ids), objective, lr)
        else:
            raise ValueError(engine)
        output.append(values)
        seen += len(signal)
    if fast_scalar is not None:
        fast_scalar.finalize()
    return np.concatenate(output)


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Return the duration of the union of valid half-open intervals."""
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
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
    """Return expanded event-scoring intervals and paired raw intervals.

    Expanded intervals are used only for event matching/sensitivity. Raw
    intervals remain the reference for alarm-time subtraction, the
    non-seizure denominator, and clinically interpretable onset delay.
    """
    scoring: list[tuple[float, float]] = []
    raw: list[tuple[float, float]] = []
    for row in seizures[seizures.recording_id == recording_id].itertuples(index=False):
        raw_start = float(row.start_s)
        raw_end = float(row.end_s)
        if raw_end <= evaluation_start:
            continue
        raw_start_clipped = max(evaluation_start, raw_start)
        raw_end_clipped = min(duration, raw_end)
        scoring_start = max(evaluation_start, raw_start - SEIZURE_ONSET_PRE_S)
        scoring_end = min(duration, raw_end + SEIZURE_OFFSET_POST_S)
        if raw_end_clipped <= raw_start_clipped or scoring_end <= scoring_start:
            continue
        raw.append((raw_start_clipped, raw_end_clipped))
        scoring.append((scoring_start, scoring_end))
    return scoring, raw


def score_probabilities(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    true_positives = false_alarms = truth_events = 0
    false_alarm_time_seconds = 0.0
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
        scoring_truths, raw_truths = _scoring_truths(seizures, recording_id, evaluation_start, duration)
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.probability.to_numpy(dtype=float),
            threshold=threshold,
        )
        matched = match_events(predictions, scoring_truths)
        true_positives += len(matched.pairs)
        false_alarms += len(matched.unmatched_predictions)
        truth_events += len(scoring_truths)
        # Alarm duration is measured against the original seizure annotation.
        # A prediction matched through the collar can therefore still
        # contribute time before onset or after offset.
        for prediction in predictions:
            overlap = _union_length([
                (max(prediction.start_s, start), min(prediction.end_s, end))
                for start, end in raw_truths
                if min(prediction.end_s, end) > max(prediction.start_s, start)
            ])
            false_alarm_time_seconds += max(0.0, prediction.end_s - prediction.start_s - overlap)
        for pair in matched.pairs:
            # Detection delay remains anchored to the raw annotated onset;
            # the collar is an event-matching tolerance, not a new onset.
            delays.append(max(0.0, predictions[pair.prediction_index].start_s - raw_truths[pair.truth_index][0]))
        raw_seizure_seconds = _union_length(raw_truths)
        total_monitoring_seconds += max(0.0, duration - evaluation_start)
        nonseizure_seconds += max(0.0, duration - evaluation_start - raw_seizure_seconds)
    hours = nonseizure_seconds / 3600.0
    monitoring_hours = total_monitoring_seconds / 3600.0
    return {
        "threshold": float(threshold),
        "seizure_scoring_collar_onset_pre_s": SEIZURE_ONSET_PRE_S,
        "seizure_scoring_collar_offset_post_s": SEIZURE_OFFSET_POST_S,
        "seizure_scoring_truth_definition": "raw interval expanded for event matching/sensitivity only; clipped to evaluable EDF",
        "alarm_time_truth_definition": "original unexpanded seizure interval; collar excluded from alarm-time subtraction and denominator",
        "true_positive_events": int(true_positives),
        "false_alarm_events": int(false_alarms),
        "false_alarm_time_seconds": float(false_alarm_time_seconds),
        "truth_events": int(truth_events),
        "event_sensitivity": float(true_positives / truth_events) if truth_events else 1.0,
        "total_monitoring_hours": float(monitoring_hours),
        "nonseizure_hours": float(hours),
        "fa_per_24h": float(false_alarms * 24.0 / hours) if hours else 0.0,
        "false_alarm_time_s_per_24h": float(false_alarm_time_seconds * 24.0 / hours) if hours else 0.0,
        "false_alarm_time_min_per_24h": float(false_alarm_time_seconds / 60.0 * 24.0 / hours) if hours else 0.0,
        "detection_delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
        "detection_delay_median_s": float(np.median(delays)) if delays else float("nan"),
    }


def select_validation_threshold(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [score_probabilities(table, seizures, recordings, float(threshold)) for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    feasible = [row for row in rows if row["event_sensitivity"] >= 0.80]
    selected = min(
        feasible,
        key=lambda row: (row["false_alarm_time_s_per_24h"], row["fa_per_24h"], -row["threshold"]),
    ) if feasible else min(
        rows,
        key=lambda row: (-row["event_sensitivity"], row["false_alarm_time_s_per_24h"], row["fa_per_24h"], -row["threshold"]),
    )
    return pd.DataFrame(rows), selected


def window_metrics(table: pd.DataFrame) -> dict[str, float]:
    labels = table.label.to_numpy(dtype=int)
    probability = table.probability.to_numpy(dtype=float)
    return {
        "window_auroc": float(roc_auc_score(labels, probability)),
        "window_auprc": float(average_precision_score(labels, probability)),
        "window_balanced_accuracy_0p5": float(balanced_accuracy_score(labels, probability >= 0.5)),
    }


def patient_waterfall(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.DataFrame([
        {"patient": patient, **score_probabilities(group, seizures, recordings, threshold)}
        for patient, group in table.groupby("patient", sort=True)
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/neurottt-chbmit-5fold-v1')
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parents[1] / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--recordings", type=Path, default=BFA_ROOT / "manifests/recordings.parquet")
    parser.add_argument("--seizures", type=Path, default=BFA_ROOT / "manifests/seizures.parquet")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--ttt-engine", choices=("scalar", "vmap"), default="scalar")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 3407:
        raise ValueError("formal v1 locks seed=3407")
    if args.split == "test" and not args.allow_test:
        raise PermissionError("outer test requires explicit --allow-test after validation lock")
    set_seed(args.seed + args.fold)
    device = torch.device(args.device)
    source_condition, objective = CONDITIONS[args.condition]
    model, checkpoint = load_model(args, source_condition, device)
    checkpoint_hash = sha256(checkpoint)
    rows = load_rows(args.fold, args.split, args.windows, args.fold_root)
    if args.limit is not None:
        rng = np.random.default_rng(args.seed + args.fold + 30_000)
        parts = []
        for label in (0.0, 1.0):
            group = rows[rows.label == label]
            count = min(len(group), max(1, args.limit // 2))
            parts.append(group.iloc[np.sort(rng.choice(len(group), size=count, replace=False))])
        rows = pd.concat(parts, ignore_index=True).sort_values(["patient", "recording", "start"], kind="stable").reset_index(drop=True)
    loader = make_eval_loader(rows, batch_size=args.batch_size, workers=args.workers, cache_root=args.cache_root)
    result_dir = args.output_root / "evaluation" / args.condition / f"fold{args.fold}_seed{args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    probability_path = result_dir / f"{args.split}_probabilities.parquet"
    if probability_path.exists():
        raise FileExistsError(probability_path)
    validation_metrics_path = result_dir / "validation_metrics.json"
    if args.split == "test":
        if not validation_metrics_path.is_file():
            raise FileNotFoundError("validation threshold is not frozen")
        validation_lock = json.loads(validation_metrics_path.read_text())
        if validation_lock["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("checkpoint changed after validation threshold selection")
        threshold = float(validation_lock["selected_event_operating_point"]["threshold"])
    else:
        threshold = float("nan")
    started = time.monotonic()
    if objective is None:
        probability = frozen_probabilities(model, loader, device, args.limit)
    else:
        probability = ttt_probabilities(model, loader, device, objective, args.limit, args.ttt_lr, args.ttt_engine)
    if len(probability) != len(rows):
        raise RuntimeError("probability count mismatch")
    table = rows[["patient", "recording", "start", "end", "label", "sample_id", "relative_path"]].copy()
    table["probability"] = probability
    table.to_parquet(probability_path, index=False)
    recordings = pd.read_parquet(args.recordings)
    seizures = pd.read_parquet(args.seizures)
    base = {
        "release_id": "neurottt-chbmit-5fold-v1",
        "condition": args.condition,
        "source_condition": source_condition,
        "ttt_objective": objective,
        "fold": args.fold,
        "seed": args.seed,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "probability_sha256": sha256(probability_path),
        "rows": len(table),
        "elapsed_s": time.monotonic() - started,
        "window_metrics": window_metrics(table),
        "ttt_steps": 1 if objective else 0,
        "ttt_optimizer": "SGD" if objective else None,
        "ttt_lr": args.ttt_lr if objective else None,
        "episodic_restore": bool(objective),
        "ttt_engine": args.ttt_engine if objective else None,
    }
    if args.split == "validation":
        if args.limit is not None:
            selected = {
                "threshold": 0.5,
                "event_sensitivity": float("nan"),
                "fa_per_24h": float("nan"),
                "smoke_only": True,
            }
            sweep = pd.DataFrame([selected])
        else:
            sweep, selected = select_validation_threshold(table, seizures, recordings)
        sweep_path = result_dir / "validation_threshold_sweep.parquet"
        sweep.to_parquet(sweep_path, index=False)
        result = {**base, "status": "validation_threshold_frozen", "selected_event_operating_point": selected, "threshold_sweep_sha256": sha256(sweep_path), "test_evaluation_count": 0}
        atomic_json(validation_metrics_path, result)
    else:
        metric = score_probabilities(table, seizures, recordings, threshold)
        waterfall = patient_waterfall(table, seizures, recordings, threshold)
        waterfall_path = result_dir / "test_patient_waterfall.csv"
        waterfall.to_csv(waterfall_path, index=False)
        result = {**base, "status": "test_complete", "threshold_source": "validation_only", "selected_event_operating_point": metric, "patient_waterfall_sha256": sha256(waterfall_path), "test_evaluation_count": 1}
        atomic_json(result_dir / "test_completed.json", result)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
