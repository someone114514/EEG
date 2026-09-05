from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WindowRef:
    patient_id: str
    recording_id: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.patient_id or not self.recording_id:
            raise ValueError("patient_id and recording_id must be non-empty")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("window bounds must satisfy 0 <= start_s < end_s")

    @property
    def available_at_s(self) -> float:
        return self.end_s


@dataclass(frozen=True)
class ModelOutput:
    probability: torch.Tensor
    embedding: torch.Tensor
    channel_quality: torch.Tensor
    window_refs: tuple[WindowRef, ...]
    extras: Mapping[str, torch.Tensor] | None = None

    def validate(self) -> None:
        if self.probability.ndim != 1:
            raise AssertionError("probability must have shape [batch]")
        batch = self.probability.shape[0]
        if self.embedding.shape != (batch, 16, 128):
            raise AssertionError("embedding must have shape [batch, 16, 128]")
        if self.channel_quality.shape != (batch, 16):
            raise AssertionError("channel_quality must have shape [batch, 16]")
        if len(self.window_refs) != batch:
            raise AssertionError("one WindowRef is required per batch item")
        tensors = [self.probability, self.embedding, self.channel_quality]
        if self.extras:
            tensors.extend(self.extras.values())
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            raise AssertionError("ModelOutput tensors must be finite")


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    split_hash: str
    seed: int
    git_commit: str
    config_hash: str
