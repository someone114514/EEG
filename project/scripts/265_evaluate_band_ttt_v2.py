"""Causal Band-TTT v2 evaluator for the frozen 16-condition fold-0/1 matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
CODE_ROOT = Path(os.environ.get("NEUROTTT_CODE_ROOT", "/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod"))
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from chbmit_groupkfold.data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, WindowDataset, load_rows  # noqa: E402
from chbmit_groupkfold.meta_evaluate import score_probabilities, select_validation_threshold  # noqa: E402
from chbmit_groupkfold.meta_model import CHBMetaTTTModel  # noqa: E402
from chbmit_groupkfold.transforms import deterministic_band_view  # noqa: E402

RELEASE = "band-ttt-v2-fold01"
SOURCE_RELEASE = ROOT / "outputs" / "reports" / "meta-ttt-chbmit-5fold-v1"
PRETRAINED = CODE_ROOT / "pretrained_weights" / "pretrained_weights.pth"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
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


def load_release(output_root: Path, config_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_path = output_root / "frozen_manifest.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"run freeze script first: {frozen_path}")
    frozen = json.loads(frozen_path.read_text())
    matches = [item for item in frozen["configurations"] if item["config_id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate config_id={config_id}")
    if len(frozen["configurations"]) != 16 or frozen["expected_evaluation_jobs"] != 64:
        raise ValueError("frozen matrix invariant failed")
    return frozen, matches[0]


def load_model(fold: int, device: torch.device) -> tuple[CHBMetaTTTModel, Path, float]:
    checkpoint = SOURCE_RELEASE / "runs" / f"meta_band_fold{fold}_seed3407" / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CHBMetaTTTModel(PRETRAINED)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint, float(state.get("alpha", 1e-4))


def configure_adaptive(model: CHBMetaTTTModel) -> dict[str, torch.nn.Parameter]:
    model.requires_grad_(False)
    for parameter in model.backbone.encoder.layers[-2].parameters():
        parameter.requires_grad_(True)
    for parameter in model.backbone.encoder.layers[-1].parameters():
        parameter.requires_grad_(True)
    model.band_head.requires_grad_(True)
    adaptive = model.adaptive_named_parameters("band")
    if not adaptive or any(not parameter.requires_grad for parameter in adaptive.values()):
        raise RuntimeError("invalid Band-TTT adaptive parameter scope")
    return adaptive


@torch.no_grad()
def frozen_prefix(model: CHBMetaTTTModel, signal: torch.Tensor) -> torch.Tensor:
    """Run the immutable patch embedder and first ten blocks once per view."""
    features = model.backbone.patch_embedding(signal)
    for layer in model.backbone.encoder.layers[:10]:
        features = layer(features)
    return features


def frozen_prefix_pair(
    model: CHBMetaTTTModel,
    transformed: torch.Tensor,
    signal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode both immutable prefixes in one native batched MHA call."""
    if len(transformed) != len(signal):
        raise ValueError("paired prefix batches must have equal length")
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(True)
    try:
        combined = frozen_prefix(model, torch.cat((transformed, signal), dim=0))
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    return combined.split(len(signal), dim=0)


class BandTail(torch.nn.Module):
    """A functional-call-friendly view of the only adaptive model tail."""

    def __init__(self, model: CHBMetaTTTModel) -> None:
        super().__init__()
        self.layer10 = model.backbone.encoder.layers[10]
        self.layer11 = model.backbone.encoder.layers[11]
        self.band_head = model.band_head
        self.detector = model.detector

    def forward(self, prefix: torch.Tensor, *, mode: str) -> torch.Tensor:
        features = self.layer11(self.layer10(prefix))
        if mode == "band":
            return self.band_head(features.mean(dim=1).flatten(1))
        if mode == "detect":
            return self.detector(features)
        raise ValueError(mode)

    def adaptive_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if name.startswith("layer10.") or name.startswith("layer11.") or name.startswith("band_head.")
        }


def live_tail(model: CHBMetaTTTModel, prefix: torch.Tensor) -> torch.Tensor:
    value = model.backbone.encoder.layers[10](prefix)
    return model.backbone.encoder.layers[11](value)


def lr_multiplier(name: str, strategy: str) -> float:
    if strategy == "global":
        return 1.0
    if strategy != "layerwise":
        raise ValueError(strategy)
    if ".encoder.layers.10." in name or name.startswith("layer10."):
        return 0.5
    if ".encoder.layers.11." in name or name.startswith("layer11.") or name.startswith("band_head."):
        return 1.0
    raise ValueError(f"unexpected adaptive parameter for layerwise LR: {name}")


def make_optimizer(model: CHBMetaTTTModel, kind: str, strategy: str, alpha: float):
    adaptive = configure_adaptive(model)
    groups = [
        {"params": [parameter], "lr": alpha * lr_multiplier(name, strategy)}
        for name, parameter in adaptive.items()
    ]
    if kind == "sgd":
        return torch.optim.SGD(groups, momentum=0.0, weight_decay=0.0)
    if kind == "adam":
        if strategy != "global":
            raise ValueError("Adam is prespecified only for global LR")
        return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    raise ValueError(kind)


def ordered_rows(rows: pd.DataFrame, order_payload: dict[str, Any]) -> pd.DataFrame:
    pieces = []
    for patient in sorted(rows.patient.astype(str).unique()):
        patient_rows = rows[rows.patient.astype(str) == patient].copy()
        official = order_payload["patients"][patient]["recordings"]
        rank = {recording: index for index, recording in enumerate(official)}
        unknown = set(patient_rows.recording.astype(str)) - set(rank)
        if unknown:
            raise RuntimeError(f"recordings absent from official order patient={patient}: {sorted(unknown)}")
        patient_rows["_record_rank"] = patient_rows.recording.astype(str).map(rank)
        patient_rows = patient_rows.sort_values(["_record_rank", "start"], kind="stable").drop(columns="_record_rank")
        pieces.append(patient_rows)
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != len(rows) or set(result.sample_id) != set(rows.sample_id):
        raise RuntimeError("ordered-row conservation failed")
    return result


def data_loader(rows: pd.DataFrame, batch_size: int, cache_root: Path, workers: int = 0) -> DataLoader:
    return DataLoader(
        WindowDataset(rows, cache_root=cache_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def predict(model: CHBMetaTTTModel, signal: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Keep the formal v1 FP32 evaluation numerics so deltas against its frozen
    # baseline measure adaptation rather than a precision-mode change.
    with torch.inference_mode():
        return torch.sigmoid(model.detect(signal).float())


def ssl_steps_from_prefix(
    model: CHBMetaTTTModel,
    optimizer: torch.optim.Optimizer,
    transformed_prefix: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(transformed_prefix, mode="band"), labels)
        loss.backward()
        optimizer.step()


def window_independent_probabilities(
    model: CHBMetaTTTModel,
    rows: pd.DataFrame,
    device: torch.device,
    alpha: float,
    steps: int,
    strategy: str,
    batch_size: int,
    cache_root: Path,
    workers: int = 8,
    parts_dir: Path | None = None,
) -> np.ndarray:
    """Vectorized independent K-step adaptation without higher-order graphs."""
    configure_adaptive(model)
    tail = BandTail(model).to(device).eval()
    adaptive = tail.adaptive_state()
    source_hash = module_hash(model)
    output: list[np.ndarray] = []
    buffered_indices: list[np.ndarray] = []
    seen = 0
    part_index = 0
    resume_rows = 0
    if parts_dir is not None:
        parts_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(parts_dir.glob("part_*.parquet"))
        if existing:
            indexed = pd.concat([pd.read_parquet(path, columns=["row_index"]) for path in existing], ignore_index=True)
            if not np.array_equal(indexed.row_index.to_numpy(), np.arange(len(indexed))):
                raise RuntimeError("window part checkpoint indices are not contiguous")
            resume_rows = len(indexed)
            part_index = len(existing)
    for batch_index, (signal, _, sample_ids) in enumerate(data_loader(rows, batch_size, cache_root, workers)):
        if seen < resume_rows:
            if seen + len(signal) > resume_rows:
                raise RuntimeError("window checkpoint is not batch aligned")
            seen += len(signal)
            continue
        row_indices = np.arange(seen, seen + len(signal), dtype=np.int64)
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        ids = list(sample_ids)
        transformed, labels = deterministic_band_view(signal, ids)
        transformed_prefix, signal_prefix = frozen_prefix_pair(model, transformed, signal)
        def loss_one(current: dict[str, torch.Tensor], sample: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            logits = functional_call(tail, current, (sample.unsqueeze(0),), {"mode": "band"}, strict=False)
            return F.cross_entropy(logits, label.unsqueeze(0))

        current: dict[str, torch.Tensor] = dict(adaptive)
        batched = False
        for _ in range(steps):
            gradients = vmap(grad(loss_one), in_dims=(0 if batched else None, 0, 0), randomness="same")(current, transformed_prefix, labels)
            current = {
                name: (parameter - alpha * lr_multiplier(name, strategy) * gradients[name]).detach()
                for name, parameter in current.items()
            }
            batched = True

        def detect_one(current_params: dict[str, torch.Tensor], sample: torch.Tensor) -> torch.Tensor:
            return functional_call(tail, current_params, (sample.unsqueeze(0),), {"mode": "detect"}, strict=False).squeeze(0)

        logits = vmap(detect_one, in_dims=(0, 0), randomness="same")(current, signal_prefix)
        output.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
        buffered_indices.append(row_indices)
        seen += len(signal)
        if parts_dir is not None and (len(output) >= 100 or seen == len(rows)):
            frame = pd.DataFrame({"row_index": np.concatenate(buffered_indices), "probability": np.concatenate(output)})
            frame.to_parquet(parts_dir / f"part_{part_index:05d}.parquet", index=False)
            output.clear(); buffered_indices.clear(); part_index += 1
            print(f"window checkpoint {seen}/{len(rows)}", flush=True)
    if module_hash(model) != source_hash:
        raise RuntimeError("independent functional updates mutated source model")
    if parts_dir is None:
        return np.concatenate(output)
    parts = pd.concat([pd.read_parquet(path) for path in sorted(parts_dir.glob("part_*.parquet"))], ignore_index=True)
    if len(parts) != len(rows) or not np.array_equal(parts.row_index.to_numpy(), np.arange(len(rows))):
        raise RuntimeError("window part checkpoint conservation failed")
    return parts.probability.to_numpy(dtype=np.float32)


def restore_adaptive(adaptive: dict[str, torch.nn.Parameter], source: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in adaptive.items():
            parameter.copy_(source[name])


def batched_stream_probabilities(
    model: CHBMetaTTTModel,
    rows: pd.DataFrame,
    device: torch.device,
    alpha: float,
    config: dict[str, Any],
    cache_root: Path,
    progress_path: Path,
    stream_batch_size: int,
) -> np.ndarray:
    """Run independent record/patient streams in parallel with batched states.

    Time remains strictly serial inside each stream.  Parallelism exists only
    across independent records (record reset) or independent patients
    (patient reset), so it cannot leak future EEG into an update.
    """
    configure_adaptive(model)
    tail = BandTail(model).to(device).eval()
    adaptive = tail.adaptive_state()
    source_hash = module_hash(model)
    dataset = WindowDataset(rows, cache_root=cache_root, max_open=max(32, stream_batch_size * 2))
    if config["accumulation_scope"] == "record":
        streams = [group.index.to_numpy(dtype=np.int64) for _, group in rows.groupby(["patient", "recording"], sort=False)]
    elif config["accumulation_scope"] == "patient":
        streams = [group.index.to_numpy(dtype=np.int64) for _, group in rows.groupby("patient", sort=True)]
    else:
        raise ValueError(config["accumulation_scope"])
    streams.sort(key=len, reverse=True)
    probability = np.empty(len(rows), dtype=np.float32)
    processed = 0
    for begin in range(0, len(streams), stream_batch_size):
        batch_streams = streams[begin:begin + stream_batch_size]
        batch_streams.sort(key=len, reverse=True)
        count = len(batch_streams)
        current = {
            name: parameter.detach().unsqueeze(0).expand(count, *parameter.shape).clone()
            for name, parameter in adaptive.items()
        }
        first = next(iter(current.values()))
        moments1 = {name: torch.zeros_like(parameter) for name, parameter in current.items()} if config["optimizer"] == "adam" else None
        moments2 = {name: torch.zeros_like(parameter) for name, parameter in current.items()} if config["optimizer"] == "adam" else None
        adam_step = 0

        def loss_one(parameters: dict[str, torch.Tensor], sample: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            logits = functional_call(tail, parameters, (sample.unsqueeze(0),), {"mode": "band"}, strict=False)
            return F.cross_entropy(logits, label.unsqueeze(0))

        def detect_one(parameters: dict[str, torch.Tensor], sample: torch.Tensor) -> torch.Tensor:
            return functional_call(tail, parameters, (sample.unsqueeze(0),), {"mode": "detect"}, strict=False).squeeze(0)

        del first
        for time_index in range(len(batch_streams[0])):
            active_count = sum(time_index < len(stream) for stream in batch_streams)
            if active_count == 0:
                break
            if len(next(iter(current.values()))) != active_count:
                current = {name: value[:active_count] for name, value in current.items()}
                if moments1 is not None and moments2 is not None:
                    moments1 = {name: value[:active_count] for name, value in moments1.items()}
                    moments2 = {name: value[:active_count] for name, value in moments2.items()}
            row_indices = [int(batch_streams[index][time_index]) for index in range(active_count)]
            samples = [dataset[index] for index in row_indices]
            signal = torch.stack([sample[0] for sample in samples]).to(device=device, dtype=torch.float32, non_blocking=True)
            sample_ids = [sample[2] for sample in samples]
            transformed, labels = deterministic_band_view(signal, sample_ids)
            transformed_prefix, signal_prefix = frozen_prefix_pair(model, transformed, signal)
            for _ in range(int(config["steps"])):
                gradients = vmap(grad(loss_one), in_dims=(0, 0, 0), randomness="same")(current, transformed_prefix, labels)
                if config["optimizer"] == "sgd":
                    current = {
                        name: (parameter - alpha * lr_multiplier(name, str(config["lr_strategy"])) * gradients[name]).detach()
                        for name, parameter in current.items()
                    }
                else:
                    assert moments1 is not None and moments2 is not None
                    adam_step += 1
                    moments1 = {name: (0.9 * moments1[name] + 0.1 * gradients[name]).detach() for name in current}
                    moments2 = {name: (0.999 * moments2[name] + 0.001 * gradients[name].square()).detach() for name in current}
                    correction1 = 1.0 - 0.9 ** adam_step
                    correction2 = 1.0 - 0.999 ** adam_step
                    current = {
                        name: (parameter - alpha * (moments1[name] / correction1) / ((moments2[name] / correction2).sqrt() + 1e-8)).detach()
                        for name, parameter in current.items()
                    }
            logits = vmap(detect_one, in_dims=(0, 0), randomness="same")(current, signal_prefix)
            values = torch.sigmoid(logits.float()).detach().cpu().numpy()
            probability[np.asarray(row_indices)] = values
            processed += active_count
        atomic_json(progress_path, {"status": "running", "processed_rows": processed, "total_rows": len(rows), "completed_streams": min(begin + stream_batch_size, len(streams)), "total_streams": len(streams), "updated_utc": utc_now()})
        print(f"stream checkpoint {processed}/{len(rows)} streams {min(begin + stream_batch_size, len(streams))}/{len(streams)}", flush=True)
    if not np.isfinite(probability).all() or module_hash(model) != source_hash:
        raise RuntimeError("batched stream produced invalid probabilities or mutated source model")
    return probability


def accumulating_probabilities(
    model: CHBMetaTTTModel,
    rows: pd.DataFrame,
    device: torch.device,
    alpha: float,
    config: dict[str, Any],
    load_batch_size: int,
    cache_root: Path,
    progress_path: Path,
) -> np.ndarray:
    adaptive = configure_adaptive(model)
    source = {name: parameter.detach().clone() for name, parameter in adaptive.items()}
    outputs: list[np.ndarray] = []
    processed = 0
    scope = config["accumulation_scope"]
    for patient in sorted(rows.patient.astype(str).unique()):
        patient_rows = rows[rows.patient.astype(str) == patient].copy()
        restore_adaptive(adaptive, source)
        patient_optimizer = make_optimizer(model, config["optimizer"], config["lr_strategy"], alpha)
        groups = [patient_rows] if scope in {"patient", "patient_chunk"} else [group for _, group in patient_rows.groupby("recording", sort=False)]
        for group in groups:
            if scope == "record":
                restore_adaptive(adaptive, source)
                optimizer = make_optimizer(model, config["optimizer"], config["lr_strategy"], alpha)
            else:
                optimizer = patient_optimizer
            if scope == "patient_chunk":
                chunk_size = int(config["chunk_size_windows"])
                loader = data_loader(group, chunk_size, cache_root)
                for signal, _, sample_ids in loader:
                    signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
                    # Strict prequential chunk rule: score first, update second.
                    outputs.append(predict(model, signal, device).cpu().numpy())
                    transformed, labels = deterministic_band_view(signal, list(sample_ids))
                    ssl_steps_from_prefix(model, optimizer, transformed, labels, int(config["steps"]))
                    processed += len(signal)
            else:
                loader = data_loader(group, load_batch_size, cache_root)
                for signal_batch, _, sample_ids_batch in loader:
                    for index in range(len(signal_batch)):
                        signal = signal_batch[index:index + 1].to(device=device, dtype=torch.float32, non_blocking=True)
                        sample_ids = [str(sample_ids_batch[index])]
                        # A completed causal window is adapted, then scored; no future window is used.
                        transformed, labels = deterministic_band_view(signal, sample_ids)
                        ssl_steps_from_prefix(model, optimizer, transformed, labels, int(config["steps"]))
                        outputs.append(predict(model, signal, device).cpu().numpy())
                        processed += 1
        atomic_json(progress_path, {"status": "running", "patient": patient, "processed_rows": processed, "total_rows": len(rows), "updated_utc": utc_now()})
        print(f"patient checkpoint {patient} {processed}/{len(rows)}", flush=True)
    probability = np.concatenate(outputs)
    if len(probability) != len(rows):
        raise RuntimeError(f"probability count mismatch: {len(probability)} != {len(rows)}")
    return probability


def existing_frozen_lock(fold: int) -> tuple[float, dict[str, Any]]:
    result_dir = SOURCE_RELEASE / "evaluation" / "meta_band_frozen" / f"fold{fold}_seed3407"
    lock = json.loads((result_dir / "validation_metrics.json").read_text())
    threshold = float(lock["selected_event_operating_point"]["threshold"])
    return threshold, lock


def patient_waterfall(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.DataFrame([
        {"patient": patient, **score_probabilities(group, seizures, recordings, threshold)}
        for patient, group in table.groupby("patient", sort=True)
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "reports" / RELEASE)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--recordings", type=Path, default=ROOT / "manifests" / "recordings.parquet")
    parser.add_argument("--seizures", type=Path, default=ROOT / "manifests" / "seizures.parquet")
    parser.add_argument("--window-batch-size", type=int, default=128)
    parser.add_argument("--load-batch-size", type=int, default=32)
    parser.add_argument("--stream-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test:
        raise PermissionError("test requires --allow-test after its validation lock exists")
    frozen, config = load_release(args.output_root, args.config)
    result_dir = args.output_root / "evaluation" / args.config / f"fold{args.fold}_seed3407"
    result_dir.mkdir(parents=True, exist_ok=True)
    completed_path = result_dir / f"{args.split}_completed.json"
    probability_path = result_dir / f"{args.split}_probabilities.parquet"
    if completed_path.exists() or probability_path.exists():
        raise FileExistsError(f"refusing overwrite: {completed_path} / {probability_path}")
    validation_lock_path = result_dir / "validation_completed.json"
    if args.split == "test" and not validation_lock_path.is_file():
        raise FileNotFoundError(f"validation must complete first: {validation_lock_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, alpha = load_model(args.fold, device)
    if sha256(checkpoint) != frozen["source_checkpoints"][str(args.fold)]["sha256"]:
        raise ValueError("checkpoint differs from frozen manifest")
    rows = load_rows(args.fold, args.split, args.windows, args.fold_root)
    order = json.loads(Path(frozen["record_order_path"]).read_text())
    rows = ordered_rows(rows, order)
    if args.limit is not None:
        rows = rows.iloc[:args.limit].copy().reset_index(drop=True)
    recordings = pd.read_parquet(args.recordings)
    seizures = pd.read_parquet(args.seizures)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # With frozen prefix parameters, eval-mode MultiheadAttention selects
    # aten::_native_multi_head_attention.  That kernel has no torch.func vmap
    # batching rule in the pinned PyTorch build, so vmap silently falls back to
    # a per-sample loop.  The decomposed attention path is numerically
    # equivalent here and preserves actual batched execution for every
    # state-vectorized scope.  Keep the native fastpath for causal chunk mode,
    # which uses ordinary batched forward/backward calls rather than vmap.
    if config["accumulation_scope"] in {"window", "record", "patient"}:
        torch.backends.mha.set_fastpath_enabled(False)
    started = time.monotonic()
    if config["accumulation_scope"] == "window":
        probability = window_independent_probabilities(model, rows, device, alpha, int(config["steps"]), str(config["lr_strategy"]), args.window_batch_size, args.cache_root, args.workers, result_dir / f"{args.split}_parts")
    elif config["accumulation_scope"] in {"record", "patient"}:
        probability = batched_stream_probabilities(model, rows, device, alpha, config, args.cache_root, result_dir / f"{args.split}_progress.json", args.stream_batch_size)
    else:
        probability = accumulating_probabilities(model, rows, device, alpha, config, args.load_batch_size, args.cache_root, result_dir / f"{args.split}_progress.json")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    table = rows[["patient", "recording", "start", "end", "label", "sample_id", "relative_path"]].copy()
    table["probability"] = probability
    table.to_parquet(probability_path, index=False)
    frozen_threshold, frozen_lock = existing_frozen_lock(args.fold)
    base = {
        "release_id": RELEASE, "status": f"{args.split}_complete", "config": config,
        "fold": args.fold, "seed": 3407, "split": args.split, "rows": len(table),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "learned_alpha": alpha,
        "probability_path": str(probability_path), "probability_sha256": sha256(probability_path),
        "elapsed_s": elapsed, "rows_per_s": len(table) / elapsed if elapsed else float("inf"),
        "gpu_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "execution_parameters": {
            "window_batch_size": args.window_batch_size,
            "stream_batch_size": args.stream_batch_size,
            "data_loader_workers": args.workers,
            "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
        },
        "test_labels_used_for_adaptation": False, "create_graph": False,
        "official_record_order_sha256": frozen["record_order_sha256"],
        "frozen_baseline_threshold": frozen_threshold,
        "matched_frozen_threshold_metrics": score_probabilities(table, seizures, recordings, frozen_threshold),
        "created_utc": utc_now(),
    }
    if args.split == "validation":
        sweep, selected = select_validation_threshold(table, seizures, recordings)
        sweep_path = result_dir / "validation_threshold_sweep.parquet"
        sweep.to_parquet(sweep_path, index=False)
        payload = {**base, "threshold_source": "validation_only", "selected_event_operating_point": selected, "threshold_sweep_sha256": sha256(sweep_path), "test_evaluation_count": 0}
    else:
        lock = json.loads(validation_lock_path.read_text())
        if lock["checkpoint_sha256"] != sha256(checkpoint) or lock["config"] != config or lock.get("threshold_source") != "validation_only":
            raise ValueError("validation lock mismatch")
        threshold = float(lock["selected_event_operating_point"]["threshold"])
        own_metric = score_probabilities(table, seizures, recordings, threshold)
        own_waterfall = patient_waterfall(table, seizures, recordings, threshold)
        matched_waterfall = patient_waterfall(table, seizures, recordings, frozen_threshold)
        own_waterfall.to_csv(result_dir / "test_patient_waterfall.csv", index=False)
        matched_waterfall.to_csv(result_dir / "test_patient_waterfall_matched_frozen_threshold.csv", index=False)
        existing_test = json.loads((SOURCE_RELEASE / "evaluation" / "meta_band_frozen" / f"fold{args.fold}_seed3407" / "test_completed.json").read_text())
        payload = {**base, "threshold_source": "validation_only", "selected_event_operating_point": own_metric, "test_evaluation_count": 1, "existing_frozen_baseline": existing_test["selected_event_operating_point"]}
    atomic_json(completed_path, payload)
    return payload


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
