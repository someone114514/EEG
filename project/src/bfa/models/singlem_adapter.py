from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from bfa.models.base import WindowEncoder


def _load_source(model_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bfa_third_party_singlem", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SingLEM source from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SingLEMAdapter(WindowEncoder):
    def __init__(self, checkpoint: Path, *, train_backbone: bool = False) -> None:
        super().__init__()
        self.train_backbone = bool(train_backbone)
        checkpoint = checkpoint.resolve()
        source_path = checkpoint.parent.parent / "model.py"
        source = _load_source(source_path)
        payload: dict[str, Any] = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        metadata = {
            "sample_rate_hz": 128.0,
            "token_samples": 128,
            "maximum_sequence_tokens": 16,
            "input_unit": "microvolt",
            "input_scale": 0.01,
        }
        for key, expected in metadata.items():
            if payload.get(key) != expected:
                raise ValueError(f"unexpected SingLEM metadata {key}={payload.get(key)!r}")
        config = source.Config()
        for key, value in payload["model_config"].items():
            setattr(config, key, value)
        config.mask_prob = 0.0
        self.encoder = source.EEGEncoder(config)
        self.encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
        self.encoder.requires_grad_(self.train_backbone)
        self.encoder.eval()
        self.projection = torch.nn.Linear(16, 128)

    def train(self, mode: bool = True) -> SingLEMAdapter:
        super().train(mode)
        if not self.train_backbone:
            self.encoder.eval()
        return self

    def forward_window(
        self, inputs: torch.Tensor, channel_quality: torch.Tensor | None = None
    ) -> torch.Tensor:
        pooled = self.backbone_features(inputs, channel_quality)
        return self.projection(pooled)

    def backbone_features(
        self, inputs: torch.Tensor, channel_quality: torch.Tensor | None = None
    ) -> torch.Tensor:
        del channel_quality
        if inputs.ndim != 4 or inputs.shape[1:] != (16, 10, 128):
            raise ValueError("SingLEM inputs must have shape [batch, 16, 10, 128]")
        batch = inputs.shape[0]
        channel_sequences = inputs.reshape(batch * 16, 10, 128)
        if self.train_backbone:
            representations, _, sequence_length = self.encoder(channel_sequences)
        else:
            self.encoder.eval()
            with torch.no_grad():
                representations, _, sequence_length = self.encoder(channel_sequences)
        if sequence_length != 10 or representations.shape != (batch * 16, 10, 16):
            raise ValueError(f"unexpected SingLEM output shape {representations.shape}")
        pooled = representations.mean(dim=1)
        return pooled.reshape(batch, 16, 16)
