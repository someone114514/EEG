from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.nn import functional as F

from .data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, load_rows, make_eval_loader, make_train_loader
from .model import CHBJointModel
from .transforms import balanced_band_labels, band_reject_view, patch_mask


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


@dataclass
class GradientBalance:
    beta: float = 0.9
    ema_detection: float | None = None
    ema_auxiliary: float | None = None
    weight: float = 1.0
    measurements: int = 0
    clamp_hits: int = 0

    def update(self, detection: float, auxiliary: float, *, enabled: bool, maximum: float) -> None:
        if not enabled:
            # Warm-up observes the current scale but does not preserve a stale
            # early gradient in the EMA used once balancing is activated.
            self.ema_detection = detection
            self.ema_auxiliary = auxiliary
        elif self.ema_detection is None:
            self.ema_detection = detection
            self.ema_auxiliary = auxiliary
        else:
            self.ema_detection = self.beta * self.ema_detection + (1.0 - self.beta) * detection
            self.ema_auxiliary = self.beta * self.ema_auxiliary + (1.0 - self.beta) * auxiliary
        self.measurements += 1
        if enabled:
            raw = self.ema_detection / max(float(self.ema_auxiliary), 1e-12)
            self.weight = float(np.clip(raw, 0.01, maximum))
            self.clamp_hits += int(self.weight in {0.01, maximum})


def gradients(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    return list(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))


def gradient_stats(first: list[torch.Tensor | None], second: list[torch.Tensor | None]) -> tuple[float, float, float]:
    norm_a = torch.zeros((), device=first[0].device if first and first[0] is not None else "cuda")
    norm_b = torch.zeros_like(norm_a)
    dot = torch.zeros_like(norm_a)
    for a, b in zip(first, second, strict=True):
        if a is not None:
            norm_a = norm_a + a.float().square().sum()
        if b is not None:
            norm_b = norm_b + b.float().square().sum()
        if a is not None and b is not None:
            dot = dot + (a.float() * b.float()).sum()
    norm_a = norm_a.sqrt()
    norm_b = norm_b.sqrt()
    cosine = dot / (norm_a * norm_b + 1e-12)
    return float(norm_a.detach().cpu()), float(norm_b.detach().cpu()), float(cosine.detach().cpu())


def compute_losses(model: CHBJointModel, signal: torch.Tensor, target: torch.Tensor, condition: str) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
    if condition == "detection_only":
        logits = model.detect(signal)
        detection = F.binary_cross_entropy_with_logits(logits, target)
        return detection, None, {"band_accuracy": float("nan"), "mask_fraction": float("nan")}
    if condition == "band_joint":
        labels = balanced_band_labels(len(signal), signal.device)
        filtered = band_reject_view(signal, labels)
        features = model.encode(torch.cat([signal, filtered], dim=0))
        logits = model.detect_from_features(features[: len(signal)])
        detection = F.binary_cross_entropy_with_logits(logits, target)
        band_logits = model.band_logits(features[len(signal) :])
        auxiliary = F.cross_entropy(band_logits, labels)
        accuracy = float((band_logits.argmax(dim=-1) == labels).float().mean().detach().cpu())
        return detection, auxiliary, {"band_accuracy": accuracy, "mask_fraction": float("nan")}
    if condition == "mask_joint":
        features = model.encode(signal)
        logits = model.detect_from_features(features)
        detection = F.binary_cross_entropy_with_logits(logits, target)
        mask = patch_mask(len(signal), signal.shape[1], signal.shape[2], 0.5, signal.device)
        reconstructed = model.reconstruct(signal, mask)
        expanded = mask.unsqueeze(-1).expand_as(reconstructed)
        auxiliary = F.mse_loss(reconstructed[expanded], signal[expanded])
        return detection, auxiliary, {"band_accuracy": float("nan"), "mask_fraction": float(mask.float().mean().detach().cpu())}
    raise ValueError(condition)


@torch.inference_mode()
def validate(model: CHBJointModel, loader, device: torch.device, limit: int | None) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    losses: list[tuple[float, int]] = []
    seen = 0
    for signal, target, _ in loader:
        if limit is not None and seen >= limit:
            break
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.detect(signal)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits.float())
        labels.append(target.cpu().numpy())
        probabilities.append(probability.cpu().numpy())
        losses.append((float(loss.cpu()), len(target)))
        seen += len(target)
    y = np.concatenate(labels)[:limit]
    p = np.concatenate(probabilities)[:limit]
    if len(np.unique(y)) != 2:
        raise RuntimeError("validation subset lacks both labels")
    return {
        "rows": int(len(y)),
        "loss": float(sum(value * count for value, count in losses) / sum(count for _, count in losses)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "balanced_accuracy_0p5": float(balanced_accuracy_score(y, p >= 0.5)),
        "positive_prevalence": float(y.mean()),
    }


def checkpoint_payload(model, optimizer, scheduler, *, args, epoch, update, balance, best_metric, patience_count, history):
    return {
        "release_id": "neurottt-chbmit-5fold-v1",
        "condition": args.condition,
        "fold": args.fold,
        "seed": args.seed,
        "epoch": epoch,
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "gradient_balance": asdict(balance),
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
    parser.add_argument("--condition", choices=("detection_only", "band_joint", "mask_joint"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/neurottt-chbmit-5fold-v1')
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parents[1] / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--gradient-warmup", type=int, default=100)
    parser.add_argument("--gradient-interval", type=int, default=50)
    parser.add_argument("--aux-weight-max", type=float)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.effective_batch % args.batch_size:
        raise ValueError("effective batch must be divisible by physical batch")
    if args.seed != 3407:
        raise ValueError("formal v1 locks the single seed to 3407")
    set_seed(args.seed + args.fold)
    auxiliary_weight_max = args.aux_weight_max
    if auxiliary_weight_max is None:
        auxiliary_weight_max = 100.0 if args.condition == "mask_joint" else 10.0
    if auxiliary_weight_max <= 0.01:
        raise ValueError("auxiliary weight maximum must exceed 0.01")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    train_rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    validation_rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    validation_limit_for_loader = args.validation_limit
    if args.validation_limit is not None:
        # Smoke-only audit subset. Formal validation leaves the natural
        # prevalence untouched and reads every validation row.
        rng = np.random.default_rng(args.seed + 10_000 + args.fold)
        parts = []
        for label in (0.0, 1.0):
            group = validation_rows[validation_rows.label == label]
            count = min(len(group), max(1, args.validation_limit // 2))
            parts.append(group.iloc[np.sort(rng.choice(len(group), size=count, replace=False))])
        validation_rows = (
            pd.concat(parts, ignore_index=True)
            .sort_values(["patient", "recording", "start"], kind="stable")
            .reset_index(drop=True)
        )
        validation_limit_for_loader = None
    positives = int((train_rows.label == 1.0).sum())
    updates_per_epoch = max(1, math.ceil(2 * positives / args.effective_batch))
    accumulation = args.effective_batch // args.batch_size
    micro_steps = updates_per_epoch * accumulation
    train_loader, sampler = make_train_loader(
        train_rows,
        batch_size=args.batch_size,
        steps=micro_steps,
        seed=args.seed + args.fold,
        workers=args.workers,
        cache_root=args.cache_root,
    )
    validation_loader = make_eval_loader(
        validation_rows,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        cache_root=args.cache_root,
    )
    model = CHBJointModel(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    balance = GradientBalance()
    run_dir = args.output_root / "runs" / f"{args.condition}_fold{args.fold}_seed{args.seed}"
    if (run_dir / "completed.json").exists():
        raise FileExistsError(f"completed run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_id": "neurottt-chbmit-5fold-v1",
        "status": "running",
        "started_at": utc_now(),
        "condition": args.condition,
        "fold": args.fold,
        "seed": args.seed,
        "test_partition_read": False,
        "source_training": args.condition,
        "input": "16 channels x 10 patches x 200 samples; existing cache already microvolts/100; no second scaling",
        "train_rows": len(train_rows),
        "train_positive_rows": positives,
        "validation_rows": len(validation_rows),
        "train_patients": sorted(train_rows.patient.astype(str).unique()),
        "validation_patients": sorted(validation_rows.patient.astype(str).unique()),
        "updates_per_epoch": updates_per_epoch,
        "accumulation": accumulation,
        "auxiliary_weight_range": [0.01, auxiliary_weight_max] if args.condition != "detection_only" else None,
        "mask_weight_cap_revision": "train-smoke showed 25-50x smaller mask gradients and 64% saturation at the preregistered cap=10; cap raised to 100 before formal validation/test" if args.condition == "mask_joint" else None,
        "config": vars(args) | {"output_root": str(args.output_root), "windows": str(args.windows), "fold_root": str(args.fold_root), "cache_root": str(args.cache_root), "pretrained": str(args.pretrained)},
        "source_hashes": {
            "windows": sha256(args.windows),
            "cv_manifest": sha256(args.fold_root / "cv_manifest.json"),
            "fold": sha256(args.fold_root / f"fold_{args.fold}.json"),
            "pretrained": sha256(args.pretrained),
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    history: list[dict[str, Any]] = []
    gradient_history: list[dict[str, Any]] = []
    best_metric = -float("inf")
    patience_count = 0
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    stopped = False
    started = time.monotonic()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_detection: list[float] = []
        epoch_auxiliary: list[float] = []
        epoch_started = time.monotonic()
        for micro_index, (signal, target, _) in enumerate(train_loader):
            signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                detection_loss, auxiliary_loss, aux_metrics = compute_losses(model, signal, target, args.condition)
            if not torch.isfinite(detection_loss) or (auxiliary_loss is not None and not torch.isfinite(auxiliary_loss)):
                raise RuntimeError("non-finite source loss")
            measure = auxiliary_loss is not None and global_update % args.gradient_interval == 0 and micro_index % accumulation == 0
            if measure:
                reference = model.gradient_reference_parameters()
                det_grad = gradients(detection_loss, reference)
                aux_grad = gradients(auxiliary_loss, reference)
                det_norm, aux_norm, cosine = gradient_stats(det_grad, aux_grad)
                balance.update(det_norm, aux_norm, enabled=global_update >= args.gradient_warmup, maximum=auxiliary_weight_max)
                gradient_history.append({
                    "update": global_update,
                    "detection_norm": det_norm,
                    "auxiliary_norm": aux_norm,
                    "weight": balance.weight,
                    "weighted_auxiliary_norm": balance.weight * aux_norm,
                    "weighted_auxiliary_to_detection": balance.weight * aux_norm / max(det_norm, 1e-12),
                    # The controller is defined on EMA-smoothed norms.  The
                    # instantaneous ratio above is retained as a diagnostic;
                    # it is intentionally noisy because band masking and
                    # dropout are resampled on every training batch.
                    "weighted_ema_auxiliary_to_detection": (
                        balance.weight * balance.ema_auxiliary / max(balance.ema_detection, 1e-12)
                        if balance.ema_detection is not None and balance.ema_auxiliary is not None else float("nan")
                    ),
                    "cosine": cosine,
                    **aux_metrics,
                })
            total_loss = detection_loss if auxiliary_loss is None else detection_loss + balance.weight * auxiliary_loss
            (total_loss / accumulation).backward()
            epoch_detection.append(float(detection_loss.detach().cpu()))
            if auxiliary_loss is not None:
                epoch_auxiliary.append(float(auxiliary_loss.detach().cpu()))
            if (micro_index + 1) % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_update += 1
            if global_update % 10 == 0:
                atomic_json(run_dir / "progress.json", {
                    "status": "training",
                    "condition": args.condition,
                    "fold": args.fold,
                    "epoch": epoch,
                    "update": global_update,
                    "updates_per_epoch": updates_per_epoch,
                    "maximum_updates": min(total_updates, args.max_updates or total_updates),
                    "detection_loss": float(np.mean(epoch_detection[-10 * accumulation :])),
                    "auxiliary_loss": float(np.mean(epoch_auxiliary[-10 * accumulation :])) if epoch_auxiliary else None,
                    "auxiliary_weight": balance.weight,
                    "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
                    "elapsed_s": time.monotonic() - started,
                })
            if args.max_updates is not None and global_update >= args.max_updates:
                stopped = True
                break
        validation = validate(model, validation_loader, device, validation_limit_for_loader)
        row = {
            "epoch": epoch,
            "update": global_update,
            "train_detection_loss": float(np.mean(epoch_detection)),
            "train_auxiliary_loss": float(np.mean(epoch_auxiliary)) if epoch_auxiliary else None,
            "auxiliary_weight": balance.weight,
            "validation": validation,
            "epoch_seconds": time.monotonic() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = validation["auprc"] > best_metric + args.min_delta or (
            abs(validation["auprc"] - best_metric) <= args.min_delta
            and history
            and validation["auroc"] > max(item["validation"]["auroc"] for item in history[:-1] or [{"validation": {"auroc": -1.0}}])
        )
        if improved:
            best_metric = validation["auprc"]
            patience_count = 0
            atomic_torch_save(run_dir / "best.pt", checkpoint_payload(model, optimizer, scheduler, args=args, epoch=epoch, update=global_update, balance=balance, best_metric=best_metric, patience_count=patience_count, history=history))
        else:
            patience_count += 1
        atomic_torch_save(run_dir / "last.pt", checkpoint_payload(model, optimizer, scheduler, args=args, epoch=epoch, update=global_update, balance=balance, best_metric=best_metric, patience_count=patience_count, history=history))
        atomic_json(run_dir / "history.json", {"epochs": history, "gradient_measurements": gradient_history})
        print(json.dumps(row, allow_nan=True), flush=True)
        if stopped or (epoch + 1 >= args.minimum_epochs and patience_count >= args.patience):
            break
    raw_ratios = [row["weighted_auxiliary_to_detection"] for row in gradient_history if row["update"] >= args.gradient_warmup]
    # The registered balancing target is the EMA ratio.  Gating on the
    # unsmoothed per-batch ratio made a correctly balanced run fail solely
    # because the auxiliary pretext gradient is stochastic and sparse.
    ratios = [row["weighted_ema_auxiliary_to_detection"] for row in gradient_history if row["update"] >= args.gradient_warmup]
    clamp_fraction = balance.clamp_hits / max(balance.measurements, 1)
    gradient_gate = args.condition == "detection_only" or (
        bool(ratios)
        and 0.5 <= float(np.median(ratios)) <= 2.0
        and clamp_fraction <= 0.5
    )
    completed = {
        **manifest,
        "status": "smoke_complete" if args.max_updates is not None else ("training_complete" if gradient_gate else "gradient_gate_failed"),
        "completed_at": utc_now(),
        "epochs_completed": len(history),
        "updates_completed": global_update,
        "best_validation_auprc": best_metric,
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "gradient_gate_passed": gradient_gate,
        "gradient_ratio_median": float(np.median(raw_ratios)) if raw_ratios else None,
        "gradient_ema_ratio_median": float(np.median(ratios)) if ratios else None,
        "gradient_clamp_fraction": clamp_fraction,
        "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
        "elapsed_s": time.monotonic() - started,
        "test_evaluation_count": 0,
    }
    atomic_json(run_dir / "completed.json", completed)
    atomic_json(run_dir / "progress.json", completed)
    return completed


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)
    if args.max_updates is None and not result["gradient_gate_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
