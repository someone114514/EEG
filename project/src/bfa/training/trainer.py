from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.nn import functional


def capture_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    step: int,
    sampler: Any | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if sampler is not None:
        state["sampler"] = sampler.state_dict()
    return state


def restore_training_state(
    state: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: Any | None = None,
) -> dict[str, int]:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if sampler is not None and "sampler" in state:
        sampler.load_state_dict(state["sampler"])
    return {"epoch": int(state["epoch"]), "step": int(state["step"])}


def train_epoch(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    device: torch.device,
    pos_weight: torch.Tensor,
    precision: str = "bf16",
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    encoder.train()
    head.train()
    total_loss = 0.0
    examples = 0
    use_bf16 = precision == "bf16" and device.type == "cuda"
    use_fp16 = precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    trainable = [
        parameter
        for module in (encoder, head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    for batch in loader:
        inputs = batch["x"].to(device, non_blocking=True)
        labels = batch["y"].to(device, non_blocking=True).float()
        quality = batch.get("quality")
        if quality is not None:
            quality = quality.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=use_bf16 or use_fp16,
        ):
            embeddings = encoder.forward_window_sequence(inputs, quality)
            logits = head(embeddings)
            loss = functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight.to(device)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        count = labels.numel()
        total_loss += float(loss.detach()) * count
        examples += count
    return {"loss": total_loss / max(examples, 1), "examples": float(examples)}


@torch.inference_mode()
def predict_probabilities(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
    precision: str = "bf16",
) -> torch.Tensor:
    encoder.eval()
    head.eval()
    outputs: list[torch.Tensor] = []
    use_bf16 = precision == "bf16" and device.type == "cuda"
    for batch in loader:
        inputs = batch["x"].to(device, non_blocking=True)
        quality = batch.get("quality")
        if quality is not None:
            quality = quality.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            embeddings = encoder.forward_window_sequence(inputs, quality)
            probabilities = torch.sigmoid(head(embeddings))
        outputs.append(probabilities.float().cpu())
    return torch.cat(outputs) if outputs else torch.empty(0)
