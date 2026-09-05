from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from .data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, load_rows, make_eval_loader, make_train_loader
from .meta_model import CHBMetaTTTModel
from .transforms import (
    deterministic_band_view,
    deterministic_temporal_view,
    band_reject_view,
    temporal_rearrange_view,
)


OBJECTIVES = {"meta_band": "band", "meta_temporal": "temporal"}


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


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def bounded_alpha(raw: torch.Tensor) -> torch.Tensor:
    return 1e-6 + (1e-3 - 1e-6) * torch.sigmoid(raw)


@contextmanager
def second_order_sdp(device: torch.device) -> Iterator[None]:
    """Select an SDP kernel with a working second-order derivative.

    The RTX 5090/PyTorch build can run the forward and first backward pass
    through FlashAttention, but its FlashAttention backward does not expose
    the derivative needed by exact Meta-TTT.  The math backend remains on
    CUDA and is slower, but it supports the required differentiable inner
    update.  Evaluation paths do not need this context because they use
    first-order per-sample gradients.
    """
    if device.type != "cuda":
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        yield


def _inner_view(model: CHBMetaTTTModel, signal: torch.Tensor, sample_ids: list[str], objective: str, *, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if training:
        if objective == "band":
            labels = _balanced_labels(len(signal), signal.device)
            return band_reject_view(signal, labels), labels
        return temporal_rearrange_view(signal)
    if objective == "band":
        return deterministic_band_view(signal, sample_ids)
    return deterministic_temporal_view(signal, sample_ids)


def _balanced_labels(batch: int, device: torch.device) -> torch.Tensor:
    if batch < 1:
        return torch.empty(0, dtype=torch.long, device=device)
    # Five equally represented bands, with a random permutation.  For small
    # smoke batches this is as balanced as the batch size permits.
    labels = torch.arange(batch, device=device) % 5
    return labels[torch.randperm(batch, device=device)]


def _mapping_for_adaptation(model: CHBMetaTTTModel, objective: str, updated: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    mapping = dict(updated or {})
    if not mapping:
        mapping = model.adaptive_named_parameters(objective)
    return mapping


def differentiable_meta_step(
    model: CHBMetaTTTModel,
    signal: torch.Tensor,
    target: torch.Tensor,
    sample_ids: list[str],
    objective: str,
    alpha_raw: torch.Tensor,
    *,
    training: bool = True,
) -> dict[str, torch.Tensor]:
    """One exact second-order Meta-TTT update objective.

    The SSL gradient is taken on the transformed view.  A one-step parameter
    update is constructed with ``create_graph=True`` and the classification
    loss is evaluated on the unmodified EEG.  Thus gradients of the outer loss
    flow through the simulated test-time update.  Only the last two encoder
    blocks and the matching SSL head are adapted episodically; all other
    parameters remain in the functional call unchanged.
    """
    if objective not in {"band", "temporal"}:
        raise ValueError(objective)
    transformed, ssl_target = _inner_view(model, signal, sample_ids, objective, training=training)
    adaptive = model.adaptive_named_parameters(objective)
    adaptive_values = list(adaptive.values())
    with second_order_sdp(signal.device):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=signal.device.type == "cuda"):
            ssl_logits = model(transformed, mode=objective)
            ssl_loss = F.cross_entropy(ssl_logits, ssl_target)
        gradients = torch.autograd.grad(
            ssl_loss,
            adaptive_values,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        alpha = bounded_alpha(alpha_raw)
        updated = {
            name: parameter - alpha * gradient if gradient is not None else parameter
            for (name, parameter), gradient in zip(adaptive.items(), gradients, strict=True)
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=signal.device.type == "cuda"):
            adapted_logits = functional_call(model, updated, (signal,), {"mode": "detect"}, strict=False)
            classification_loss = F.binary_cross_entropy_with_logits(adapted_logits, target)
            pre_logits = model.detect(signal)
            pre_loss = F.binary_cross_entropy_with_logits(pre_logits, target)
    return {
        "classification_loss": classification_loss,
        "pre_update_classification_loss": pre_loss,
        "ssl_loss": ssl_loss,
        "adapted_logits": adapted_logits,
        "pre_update_logits": pre_logits,
        "alpha": alpha,
        "ssl_target": ssl_target,
        "transformed": transformed,
    }


@torch.inference_mode()
def validate_frozen(model: CHBMetaTTTModel, loader, device: torch.device, limit: int | None) -> dict[str, float]:
    model.eval()
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
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    return {
        "rows": int(len(y)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "balanced_accuracy_0p5": float(balanced_accuracy_score(y, p >= 0.5)),
        "positive_prevalence": float(y.mean()),
    }


def _batch_adapted_logits(
    model: CHBMetaTTTModel,
    signal: torch.Tensor,
    sample_ids: list[str],
    objective: str,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Vectorized independent one-step updates for evaluation.

    This is mathematically the same per-sample update as the scalar episodic
    rule, while avoiding a Python loop over the batch.  It is used only in
    eval mode, where dropout is disabled; a scalar parity check is run before
    the formal queue accepts it.
    """
    model.eval()
    adaptive = model.adaptive_named_parameters(objective)
    transformed, labels = _inner_view(model, signal, sample_ids, objective, training=False)

    def ssl_loss(current: dict[str, torch.Tensor], sample: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, current, (sample.unsqueeze(0),), {"mode": objective}, strict=False)
        return F.cross_entropy(logits, label.unsqueeze(0))

    from torch.func import grad, vmap

    gradients = vmap(grad(ssl_loss), in_dims=(None, 0, 0), randomness="same")(adaptive, transformed, labels)
    updated = {name: parameter.unsqueeze(0) - alpha * gradients[name] for name, parameter in adaptive.items()}

    def predict(current: dict[str, torch.Tensor], sample: torch.Tensor) -> torch.Tensor:
        return functional_call(model, current, (sample.unsqueeze(0),), {"mode": "detect"}, strict=False).squeeze(0)

    logits = vmap(predict, in_dims=(0, 0), randomness="same")(updated, signal)
    return logits


def adapted_probabilities(model: CHBMetaTTTModel, loader, device: torch.device, objective: str, limit: int | None, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    seen = 0
    alpha_tensor = torch.tensor(float(alpha), device=device)
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
    return np.concatenate(labels), np.concatenate(probabilities)


def post_ttt_validation(model: CHBMetaTTTModel, loader, device: torch.device, objective: str, limit: int | None, alpha: float) -> dict[str, float]:
    y, p = adapted_probabilities(model, loader, device, objective, limit, alpha)
    return {
        "rows": int(len(y)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "balanced_accuracy_0p5": float(balanced_accuracy_score(y, p >= 0.5)),
        "positive_prevalence": float(y.mean()),
    }


def checkpoint_payload(model, optimizer, scheduler, alpha_raw, *, args, epoch, update, best_metric, patience_count, history):
    return {
        "release_id": "meta-ttt-chbmit-5fold-v1",
        "condition": args.condition,
        "objective": OBJECTIVES[args.condition],
        "fold": args.fold,
        "seed": args.seed,
        "epoch": epoch,
        "update": update,
        "model": model.state_dict(),
        "alpha_raw": alpha_raw.detach().cpu(),
        "alpha": float(bounded_alpha(alpha_raw).detach().cpu()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
        "patience_count": patience_count,
        "history": history,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "saved_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(OBJECTIVES), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/meta-ttt-chbmit-5fold-v1')
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parents[1] / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--initial-alpha", type=float, default=1e-4)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 3407:
        raise ValueError("formal v1 locks seed=3407")
    if not 1e-6 < args.initial_alpha < 1e-3:
        raise ValueError("initial alpha must be strictly within [1e-6,1e-3]")
    if args.effective_batch % args.batch_size:
        raise ValueError("effective batch must be divisible by batch size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    set_seed(args.seed + args.fold)
    objective = OBJECTIVES[args.condition]
    train_rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    validation_rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    if args.validation_limit is not None:
        rng = np.random.default_rng(args.seed + 10_000 + args.fold)
        parts = []
        for label in (0.0, 1.0):
            group = validation_rows[validation_rows.label == label]
            count = min(len(group), max(1, args.validation_limit // 2))
            parts.append(group.iloc[np.sort(rng.choice(len(group), size=count, replace=False))])
        validation_rows = pd.concat(parts, ignore_index=True).sort_values(["patient", "recording", "start"], kind="stable").reset_index(drop=True)
    positives = int((train_rows.label == 1.0).sum())
    updates_per_epoch = max(1, math.ceil(2 * positives / args.effective_batch))
    accumulation = args.effective_batch // args.batch_size
    train_loader, sampler = make_train_loader(
        train_rows, batch_size=args.batch_size, steps=updates_per_epoch * accumulation,
        seed=args.seed + args.fold, workers=args.workers, cache_root=args.cache_root,
    )
    validation_loader = make_eval_loader(validation_rows, batch_size=args.eval_batch_size, workers=args.workers, cache_root=args.cache_root)
    model = CHBMetaTTTModel(args.pretrained).to(device)
    alpha_init_logit = math.log((args.initial_alpha - 1e-6) / (1e-3 - args.initial_alpha))
    alpha_raw = nn.Parameter(torch.tensor(alpha_init_logit, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(list(model.parameters()) + [alpha_raw], lr=args.lr, weight_decay=args.weight_decay)
    total_updates = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    run_dir = args.output_root / "runs" / f"{args.condition}_fold{args.fold}_seed{args.seed}"
    if (run_dir / "completed.json").exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_id": "meta-ttt-chbmit-5fold-v1", "status": "running",
        "started_at": utc_now(), "condition": args.condition, "objective": objective,
        "fold": args.fold, "seed": args.seed, "test_partition_read": False,
        "source_training": "post_update_classification_only",
        "inner_update": "last_two_transformer_blocks_plus_ssl_head",
        "outer_objective": "post_update_detection_bce_only",
        "inner_step_size": {"type": "learned_bounded_scalar", "lower": 1e-6, "upper": 1e-3, "initial": args.initial_alpha},
        "input": "16 channels x 10 patches x 200 samples; existing cache already microvolts/100; no second scaling",
        "train_rows": len(train_rows), "train_positive_rows": positives, "validation_rows": len(validation_rows),
        "train_patients": sorted(train_rows.patient.astype(str).unique()),
        "validation_patients": sorted(validation_rows.patient.astype(str).unique()),
        "updates_per_epoch": updates_per_epoch, "accumulation": accumulation,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_hashes": {
            "windows": sha256(args.windows), "cv_manifest": sha256(args.fold_root / "cv_manifest.json"),
            "fold": sha256(args.fold_root / f"fold_{args.fold}.json"), "pretrained": sha256(args.pretrained),
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    history: list[dict[str, Any]] = []
    best_metric = -float("inf")
    patience_count = 0
    global_update = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_values: list[dict[str, float]] = []
        epoch_started = time.monotonic()
        for micro_index, (signal, target, sample_ids) in enumerate(train_loader):
            signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            values = differentiable_meta_step(model, signal, target, list(sample_ids), objective, alpha_raw, training=True)
            post_loss = values["classification_loss"]
            if not torch.isfinite(post_loss) or not torch.isfinite(values["ssl_loss"]):
                raise RuntimeError("non-finite meta loss")
            (post_loss / accumulation).backward()
            epoch_values.append({
                "post_classification_loss": float(post_loss.detach().cpu()),
                "pre_classification_loss": float(values["pre_update_classification_loss"].detach().cpu()),
                "ssl_loss": float(values["ssl_loss"].detach().cpu()),
                "alpha": float(values["alpha"].detach().cpu()),
            })
            if (micro_index + 1) % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [alpha_raw], 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_update += 1
            if global_update % 10 == 0:
                atomic_json(run_dir / "progress.json", {
                    "status": "training", "condition": args.condition, "objective": objective,
                    "fold": args.fold, "seed": args.seed, "epoch": epoch, "update": global_update,
                    "updates_per_epoch": updates_per_epoch, "maximum_updates": min(total_updates, args.max_updates or total_updates),
                    "post_classification_loss": float(np.mean([v["post_classification_loss"] for v in epoch_values[-10 * accumulation:]])),
                    "pre_classification_loss": float(np.mean([v["pre_classification_loss"] for v in epoch_values[-10 * accumulation:]])),
                    "ssl_loss": float(np.mean([v["ssl_loss"] for v in epoch_values[-10 * accumulation:]])),
                    "alpha": float(bounded_alpha(alpha_raw).detach().cpu()),
                    "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
                    "elapsed_s": time.monotonic() - started,
                })
            if args.max_updates is not None and global_update >= args.max_updates:
                break
        model.eval()
        frozen = validate_frozen(model, validation_loader, device, args.validation_limit)
        # Smoke runs use a small validation subset. Formal runs use every
        # validation row; thresholds are still selected later by the evaluator.
        adapted = post_ttt_validation(model, validation_loader, device, objective, args.validation_limit, float(bounded_alpha(alpha_raw).detach().cpu()))
        row = {
            "epoch": epoch, "update": global_update,
            "train": {key: float(np.mean([v[key] for v in epoch_values])) for key in epoch_values[0]},
            "alpha": float(bounded_alpha(alpha_raw).detach().cpu()),
            "frozen_validation": frozen, "post_ttt_validation": adapted,
            "epoch_seconds": time.monotonic() - epoch_started, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = adapted["auprc"] > best_metric + args.min_delta or (abs(adapted["auprc"] - best_metric) <= args.min_delta and adapted["auroc"] > max((item["post_ttt_validation"]["auroc"] for item in history[:-1]), default=-1.0))
        if improved:
            best_metric = adapted["auprc"]; patience_count = 0
            atomic_torch_save(run_dir / "best.pt", checkpoint_payload(model, optimizer, scheduler, alpha_raw, args=args, epoch=epoch, update=global_update, best_metric=best_metric, patience_count=patience_count, history=history))
        else:
            patience_count += 1
        atomic_torch_save(run_dir / "last.pt", checkpoint_payload(model, optimizer, scheduler, alpha_raw, args=args, epoch=epoch, update=global_update, best_metric=best_metric, patience_count=patience_count, history=history))
        atomic_json(run_dir / "history.json", {"epochs": history})
        print(json.dumps(row, allow_nan=True), flush=True)
        if (args.max_updates is not None and global_update >= args.max_updates) or (epoch + 1 >= args.minimum_epochs and patience_count >= args.patience):
            break
    alpha_value = float(bounded_alpha(alpha_raw).detach().cpu())
    completed = {
        **manifest, "status": "smoke_complete" if args.max_updates is not None else "training_complete",
        "completed_at": utc_now(), "epochs_completed": len(history), "updates_completed": global_update,
        "best_validation_post_ttt_auprc": best_metric, "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"), "alpha_final": alpha_value,
        "alpha_at_boundary": bool(alpha_value <= 1.1e-6 or alpha_value >= 0.9999e-3),
        "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
        "elapsed_s": time.monotonic() - started, "test_evaluation_count": 0,
    }
    atomic_json(run_dir / "completed.json", completed)
    return completed


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
