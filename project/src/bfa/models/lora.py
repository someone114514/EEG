"""Small dependency-free LoRA parametrizations used by H7 comparisons."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankAdd(nn.Module):
    """Return a frozen weight plus a trainable low-rank update."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("LoRA only supports two-dimensional weights")
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.a = nn.Parameter(weight.new_empty(self.rank, weight.shape[1]))
        self.b = nn.Parameter(weight.new_zeros(weight.shape[0], self.rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5.0))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + self.scale * (self.b @ self.dropout(self.a))


def add_lora(module: nn.Module, name: str = "weight", *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> None:
    """Parametrize a Linear/MHA affine weight and freeze its original value."""
    weight = getattr(module, name)
    if weight.ndim != 2:
        raise ValueError(f"cannot LoRA-parametrize {module}.{name} with shape {tuple(weight.shape)}")
    weight.requires_grad_(False)
    parametrize.register_parametrization(
        module, name, LowRankAdd(weight.detach(), rank=rank, alpha=alpha, dropout=dropout)
    )


def apply_lora(model: nn.Module, model_name: str, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> list[str]:
    """Attach the prespecified target modules for each H7 detector."""
    targets: list[tuple[str, nn.Module, str]] = []
    for full_name, module in model.named_modules():
        if model_name == "singlem":
            if full_name.endswith("self_att_heads.concat_attn") or full_name.endswith("self_att_heads.concat_proj"):
                targets.append((full_name, module, "weight"))
        elif model_name == "cbramod":
            if full_name.endswith("self_attn_s") or full_name.endswith("self_attn_t"):
                targets.append((full_name, module, "in_proj_weight"))
            elif full_name.endswith("self_attn_s.out_proj") or full_name.endswith("self_attn_t.out_proj"):
                targets.append((full_name, module, "weight"))
        elif model_name == "tcn_gat":
            if full_name.startswith("gat_layers.") and (full_name.endswith(".lin_l") or full_name.endswith(".lin_r")):
                targets.append((full_name, module, "weight"))
            elif full_name.startswith("gat_projections."):
                targets.append((full_name, module, "weight"))
    if not targets:
        raise RuntimeError(f"no LoRA targets found for {model_name}")
    names: list[str] = []
    for full_name, module, name in targets:
        add_lora(module, name, rank=rank, alpha=alpha, dropout=dropout)
        # Keep the complete module path in the audit manifest.  Class/name
        # pairs are repeated across layers and cannot prove which affine
        # transforms were adapted.
        names.append(f"{full_name}.{name}")
    return names


def trainable_parameter_report(model: nn.Module) -> dict[str, int | float]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_fraction": trainable / total if total else 0.0}
