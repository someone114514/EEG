from __future__ import annotations

from pathlib import Path

import torch
from einops.layers.torch import Rearrange
from torch import nn

from models.cbramod import CBraMod


class CHBJointModel(nn.Module):
    """Original CHB detector with separately accessible SSL objectives."""

    def __init__(self, pretrained: Path, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        self.detector = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(16 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 1),
            Rearrange("b 1 -> b"),
        )
        self.band_head = nn.Linear(10 * 200, 5)

    def encode(self, signal: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = self.backbone.patch_embedding(signal, mask)
        return self.backbone.encoder(patches)

    def detect_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.detector(features)

    def detect(self, signal: torch.Tensor) -> torch.Tensor:
        return self.detect_from_features(self.encode(signal))

    def band_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.band_head(features.mean(dim=1).flatten(1))

    def reconstruct(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.backbone.proj_out(self.encode(signal, mask))

    def forward(self, signal: torch.Tensor, *, mode: str = "detect", mask: torch.Tensor | None = None) -> torch.Tensor:
        if mode == "detect":
            return self.detect(signal)
        if mode == "band":
            return self.band_logits(self.encode(signal))
        if mode == "mask":
            if mask is None:
                raise ValueError("mask mode requires a mask tensor")
            return self.reconstruct(signal, mask)
        raise ValueError(mode)

    def gradient_reference_parameters(self) -> list[nn.Parameter]:
        layers = self.backbone.encoder.layers
        return [parameter for layer in layers[-2:] for parameter in layer.parameters() if parameter.requires_grad]

    def adaptive_parameters(self, objective: str) -> list[nn.Parameter]:
        if objective not in {"band", "mask"}:
            raise ValueError(objective)
        parameters = list(self.backbone.patch_embedding.parameters()) + list(self.backbone.encoder.parameters())
        if objective == "band":
            parameters += list(self.band_head.parameters())
        else:
            parameters += list(self.backbone.proj_out.parameters())
        return [parameter for parameter in parameters if parameter.requires_grad]
