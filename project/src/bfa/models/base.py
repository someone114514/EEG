from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class WindowEncoder(torch.nn.Module, ABC):
    output_channels = 16
    output_dim = 128

    @abstractmethod
    def forward_window(
        self, inputs: torch.Tensor, channel_quality: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return one embedding tensor with shape [batch, 16, 128]."""

    def forward_window_sequence(
        self, inputs: torch.Tensor, channel_quality: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, context = inputs.shape[:2]
        flat_inputs = inputs.reshape(batch * context, *inputs.shape[2:])
        flat_quality = None
        if channel_quality is not None:
            flat_quality = channel_quality.reshape(
                batch * context, *channel_quality.shape[2:]
            )
        embeddings = self.forward_window(flat_inputs, flat_quality)
        expected = (batch * context, self.output_channels, self.output_dim)
        if embeddings.shape != expected:
            raise ValueError(f"encoder returned {embeddings.shape}, expected {expected}")
        return embeddings.reshape(batch, context, self.output_channels, self.output_dim)
