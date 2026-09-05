from __future__ import annotations

import torch
from torch.nn import functional


class CausalConv1d(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.left = dilation * (kernel_size - 1)
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(functional.pad(inputs, (self.left, 0)))


class SharedContextHead(torch.nn.Module):
    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        blocks: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not 1 <= blocks <= 4:
            raise ValueError("blocks must be between one and four")
        self.channel_attention = torch.nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        layers: list[torch.nn.Module] = []
        for dilation in (1, 2, 4, 8)[:blocks]:
            layers.extend(
                [
                    CausalConv1d(dim, dim, 3, dilation),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                ]
            )
        self.temporal = torch.nn.Sequential(*layers)
        self.norm = torch.nn.LayerNorm(dim)
        self.output = torch.nn.Linear(dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[2:] != (16, 128):
            raise ValueError("inputs must have shape [batch, context, 16, 128]")
        batch, context, channels, dim = inputs.shape
        tokens = inputs.reshape(batch * context, channels, dim)
        tokens, _ = self.channel_attention(tokens, tokens, tokens, need_weights=False)
        temporal = tokens.mean(dim=1).reshape(batch, context, dim).transpose(1, 2)
        temporal = self.temporal(temporal).transpose(1, 2)[:, -1]
        return self.output(self.norm(temporal)).squeeze(-1)
