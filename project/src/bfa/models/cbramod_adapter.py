from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

from bfa.models.base import WindowEncoder


def _load_source(source_root: Path) -> ModuleType:
    model_path = source_root / "models" / "cbramod.py"
    spec = importlib.util.spec_from_file_location("bfa_third_party_cbramod", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CBraMod source from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class CBraModAdapter(WindowEncoder):
    def __init__(self, checkpoint: Path, *, train_backbone: bool = False) -> None:
        super().__init__()
        self.train_backbone = bool(train_backbone)
        checkpoint = checkpoint.resolve()
        source_root = checkpoint.parent.parent
        source = _load_source(source_root)
        self.backbone = source.CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=10,
            n_layer=12,
            nhead=8,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(self.train_backbone)
        self.backbone.eval()
        self.projection = torch.nn.Linear(200, 128)

    def train(self, mode: bool = True) -> CBraModAdapter:
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
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
        if inputs.ndim != 4 or inputs.shape[1:] != (16, 10, 200):
            raise ValueError("CBraMod inputs must have shape [batch, 16, 10, 200]")
        if self.train_backbone:
            patch_embeddings = self.backbone(inputs)
        else:
            self.backbone.eval()
            with torch.no_grad():
                patch_embeddings = self.backbone(inputs)
        expected = (inputs.shape[0], 16, 10, 200)
        if patch_embeddings.shape != expected:
            raise ValueError(
                f"unexpected CBraMod output shape {patch_embeddings.shape}, expected {expected}"
            )
        return patch_embeddings.mean(dim=2)
