from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from chbmit_groupkfold.data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, load_rows, make_train_loader
from chbmit_groupkfold.meta_model import CHBMetaTTTModel
from chbmit_groupkfold.meta_train import bounded_alpha, differentiable_meta_step
from chbmit_groupkfold.meta_evaluate import _batch_adapted_logits


def sha256_module(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"); os.replace(tmp, path)


def run_one(args: argparse.Namespace, objective: str) -> dict:
    device = torch.device(args.device)
    rows = load_rows(0, "train", args.windows, args.fold_root)
    loader, _ = make_train_loader(rows, batch_size=args.batch_size, steps=args.updates, seed=3407, workers=args.workers, cache_root=args.cache_root)
    model = CHBMetaTTTModel(args.pretrained).to(device)
    raw_alpha = torch.nn.Parameter(torch.tensor(-2.2, device=device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + [raw_alpha], lr=1e-4, weight_decay=5e-2)
    losses: list[float] = []; ssl_losses: list[float] = []; alpha_values: list[float] = []; alpha_grads: list[float] = []
    started = time.monotonic(); model.train()
    for update, (signal, target, sample_ids) in enumerate(loader, start=1):
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True); target = target.to(device=device, dtype=torch.float32, non_blocking=True)
        values = differentiable_meta_step(model, signal, target, list(sample_ids), objective, raw_alpha, training=True)
        if not torch.isfinite(values["classification_loss"]) or not torch.isfinite(values["ssl_loss"]): raise RuntimeError(f"nonfinite loss at {update}")
        optimizer.zero_grad(set_to_none=True); values["classification_loss"].backward();
        if raw_alpha.grad is None or not torch.isfinite(raw_alpha.grad): raise RuntimeError(f"missing/nonfinite alpha gradient at {update}")
        alpha_grads.append(float(raw_alpha.grad.detach().cpu())); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [raw_alpha], 1.0); optimizer.step()
        losses.append(float(values["classification_loss"].detach().cpu())); ssl_losses.append(float(values["ssl_loss"].detach().cpu())); alpha_values.append(float(bounded_alpha(raw_alpha).detach().cpu()))
    model.eval()
    signal, target, sample_ids = next(iter(loader)); keep = min(4, len(signal)); signal = signal[:keep].to(device=device, dtype=torch.float32); target = target[:keep].to(device=device, dtype=torch.float32); ids = list(sample_ids[: len(signal)])
    source_hash = sha256_module(model)
    with torch.inference_mode():
        frozen = model.detect(signal)
    with torch.enable_grad():
        adapted = _batch_adapted_logits(model, signal, ids, objective, torch.tensor(alpha_values[-1], device=device))
        alpha_value = alpha_values[-1]
        alpha_logit = torch.tensor(np.log((alpha_value - 1e-6) / (1e-3 - alpha_value)), device=device)
        scalar = differentiable_meta_step(model, signal[:1], target[:1], ids[:1], objective, alpha_logit, training=False)["adapted_logits"]
    restored_hash = sha256_module(model)
    delta = float((adapted.detach() - frozen.detach()).abs().max().cpu())
    parity_delta = float((adapted[:1].detach() - scalar.detach()).abs().max().cpu())
    result = {
        "release_id": "meta-ttt-chbmit-5fold-v1", "objective": objective, "status": "passed", "updates": args.updates,
        "loss_finite": True, "ssl_loss_finite": True, "alpha_gradient_finite": True,
        "alpha_initial": 0.0001006507, "alpha_final": alpha_values[-1], "alpha_at_boundary": bool(alpha_values[-1] <= 1.1e-6 or alpha_values[-1] >= .9999e-3),
        "alpha_gradient_abs_max": max(abs(x) for x in alpha_grads), "post_update_loss_first": losses[0], "post_update_loss_last": losses[-1],
        "ssl_loss_first": ssl_losses[0], "ssl_loss_last": ssl_losses[-1], "adapted_vs_frozen_max_logit_delta": delta,
        "scalar_vs_vectorized_max_logit_delta": parity_delta, "scalar_vectorized_parity_passed": bool(parity_delta <= 1e-4),
        "adapted_prediction_nonidentical": bool(delta > 1e-7), "source_hash_before_after_ttt_equal": source_hash == restored_hash,
        "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
        "elapsed_s": time.monotonic() - started, "test_partition_read": False, "test_evaluation_count": 0,
    }
    if not result["source_hash_before_after_ttt_equal"] or not result["adapted_prediction_nonidentical"] or not result["scalar_vectorized_parity_passed"]:
        result["status"] = "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / 'outputs/reports/meta-ttt-chbmit-5fold-v1'); parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS); parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS); parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE); parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parent / "pretrained_weights/pretrained_weights.pth"); parser.add_argument("--device", default="cuda"); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--workers", type=int, default=2); parser.add_argument("--updates", type=int, default=25); args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    results = {objective: run_one(args, objective) for objective in ("band", "temporal")}
    payload = {"release_id": "meta-ttt-chbmit-5fold-v1", "status": "passed" if all(value["status"] == "passed" for value in results.values()) else "failed", "results": results, "created_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(args.output_root / "smoke" / "meta_smoke.json", payload); print(json.dumps(payload, indent=2, allow_nan=True))
    if payload["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()
