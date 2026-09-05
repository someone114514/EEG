from __future__ import annotations

import torch

from bfa.models.base import WindowEncoder


class PrecomputedProjectionEncoder(WindowEncoder):
    """Trainable official adapter projection over a frozen backbone feature cache."""

    def __init__(
        self, input_dim: int, projection: torch.nn.Linear | None = None
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection = (
            torch.nn.Linear(input_dim, 128) if projection is None else projection
        )

    def forward_window(
        self, inputs: torch.Tensor, channel_quality: torch.Tensor | None = None
    ) -> torch.Tensor:
        del channel_quality
        if inputs.ndim != 3 or inputs.shape[1:] != (16, self.input_dim):
            raise ValueError(
                f"precomputed features must have shape [batch, 16, {self.input_dim}]"
            )
        return self.projection(inputs)
