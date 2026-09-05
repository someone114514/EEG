from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.temporal import TemporalPretext

from .model import CHBJointModel


class CHBMetaTTTModel(CHBJointModel):
    """CBraMod detector with a temporal-order SSL head for Meta-TTT.

    The detector architecture is deliberately inherited unchanged from the
    existing CHB implementation.  Only the temporal pretext head and the
    differentiable inner-update helpers are added in this experiment.
    """

    def __init__(self, pretrained: Path, dropout: float = 0.5) -> None:
        super().__init__(pretrained, dropout=dropout)
        self.temporal_head = TemporalPretext(input_dim=10 * 200, num_chunks=2, p_shuffle=0.5)

    def temporal_logits(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1).flatten(1)
        return self.temporal_head.classifier(pooled)

    def forward(self, signal: torch.Tensor, *, mode: str = "detect", mask: torch.Tensor | None = None) -> torch.Tensor:
        if mode == "temporal":
            return self.temporal_logits(self.encode(signal))
        return super().forward(signal, mode=mode, mask=mask)

    def adaptive_parameters(self, objective: str) -> list[nn.Parameter]:
        if objective not in {"band", "temporal"}:
            raise ValueError(objective)
        # The inner update is intentionally restricted to the last two
        # transformer blocks.  This is the same adaptation subset used by the
        # final TTT evaluator and keeps exact second-order training feasible.
        layers = self.backbone.encoder.layers
        parameters = [parameter for layer in layers[-2:] for parameter in layer.parameters()]
        parameters += list(self.band_head.parameters()) if objective == "band" else list(self.temporal_head.parameters())
        return [parameter for parameter in parameters if parameter.requires_grad]

    def adaptive_named_parameters(self, objective: str) -> dict[str, nn.Parameter]:
        ids = {id(parameter) for parameter in self.adaptive_parameters(objective)}
        return {name: parameter for name, parameter in self.named_parameters() if id(parameter) in ids}
