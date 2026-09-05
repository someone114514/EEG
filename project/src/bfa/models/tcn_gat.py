from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from bfa.models.base import WindowEncoder
from bfa.preprocessing.channels import CHANNELS


def _endpoints(channel: str) -> frozenset[str]:
    left, right = channel.split("-", maxsplit=1)
    return frozenset((left, right))


def build_bipolar_graph(channels: Sequence[str]) -> torch.Tensor:
    """Build the fixed directed graph induced by shared electrode endpoints."""
    endpoints = [_endpoints(channel) for channel in channels]
    edges = [
        (source, target)
        for source, source_endpoints in enumerate(endpoints)
        for target, target_endpoints in enumerate(endpoints)
        if source == target or source_endpoints.intersection(target_endpoints)
    ]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def has_edge(edge_index: torch.Tensor, source: int, target: int) -> bool:
    matches = (edge_index[0] == source) & (edge_index[1] == target)
    return bool(matches.any().item())


class TemporalBranch(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.GroupNorm(8, 32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).mean(dim=-1)


class MultiScaleTCNGAT(WindowEncoder):
    """Quality-aware temporal-convolutional encoder over the fixed EEG graph."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.branches = nn.ModuleList(TemporalBranch(kernel) for kernel in (15, 31, 63))
        self.node_projection = nn.Linear(96 + 3, 64)
        self.gat_layers = nn.ModuleList(
            GATv2Conv(
                64,
                64,
                heads=4,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )
            for _ in range(2)
        )
        self.gat_projections = nn.ModuleList(nn.Linear(64 * 4, 64) for _ in range(2))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(64, 128)
        self.native_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))
        self.register_buffer("edge_index", build_bipolar_graph(CHANNELS), persistent=True)

    def _batched_edge_index(self, batch_size: int) -> torch.Tensor:
        edges_per_graph = self.edge_index.shape[1]
        offsets = torch.arange(batch_size, device=self.edge_index.device) * len(CHANNELS)
        return (
            self.edge_index[:, None, :]
            .expand(2, batch_size, edges_per_graph)
            .add(offsets[None, :, None])
            .reshape(2, batch_size * edges_per_graph)
        )

    def forward_window(
        self, inputs: torch.Tensor, channel_quality: torch.Tensor | None = None
    ) -> torch.Tensor:
        expected = (len(CHANNELS), 2560)
        if inputs.ndim != 3 or inputs.shape[1:] != expected:
            raise ValueError(f"expected [batch, 16, 2560], got {tuple(inputs.shape)}")
        batch_size = inputs.shape[0]
        if channel_quality is None:
            channel_quality = inputs.new_zeros(batch_size, len(CHANNELS), 3)
        if channel_quality.shape != (batch_size, len(CHANNELS), 3):
            raise ValueError("channel_quality must have shape [batch, 16, 3]")

        temporal_inputs = inputs.reshape(batch_size * len(CHANNELS), 1, 2560)
        temporal = torch.cat([branch(temporal_inputs) for branch in self.branches], dim=-1)
        quality = channel_quality.reshape(batch_size * len(CHANNELS), 3)
        nodes = self.node_projection(torch.cat((temporal, quality), dim=-1))
        edge_index = self._batched_edge_index(batch_size)
        for graph_layer, projection in zip(self.gat_layers, self.gat_projections, strict=True):
            nodes = graph_layer(nodes, edge_index)
            nodes = self.dropout(self.activation(projection(nodes)))
        embeddings = self.output_projection(nodes)
        return embeddings.reshape(batch_size, len(CHANNELS), 128)

    def native_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1:] != (len(CHANNELS), 128):
            raise ValueError("embeddings must have shape [batch, 16, 128]")
        return self.native_head(embeddings.mean(dim=1)).squeeze(-1)
