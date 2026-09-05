from __future__ import annotations

import torch


def training_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0:
        raise ValueError("training labels contain no positive examples")
    return negatives / positives
