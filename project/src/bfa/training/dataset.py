from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bfa.preprocessing.signal import MODEL_RATES


class CachedContextDataset(Dataset[dict[str, object]]):
    """Read a causal 31-window context from deterministic continuous caches."""

    def __init__(
        self,
        index: pd.DataFrame,
        cache_root: Path,
        model: str,
        *,
        quality_root: Path | None = None,
        max_open_recordings: int = 16,
        context_windows: int = 31,
    ) -> None:
        if model not in MODEL_RATES:
            raise ValueError(f"unknown model: {model}")
        required = {"relative_path", "start", "label"}
        if not required.issubset(index.columns):
            raise ValueError(f"index missing columns: {sorted(required - set(index.columns))}")
        self.index = index.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.quality_root = self.cache_root if quality_root is None else Path(quality_root)
        self.model = model
        self.rate = MODEL_RATES[model]
        self.max_open_recordings = max_open_recordings
        self.context_windows = int(context_windows)
        if self.context_windows < 1:
            raise ValueError("context_windows must be positive")
        self._arrays: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _array(self, kind: str, relative_path: str) -> np.ndarray:
        key = (kind, relative_path)
        if key in self._arrays:
            self._arrays.move_to_end(key)
            return self._arrays[key]
        root = self.quality_root if kind == "quality" else self.cache_root
        path = root / kind / Path(relative_path).with_suffix(".npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._arrays[key] = array
        while len(self._arrays) > self.max_open_recordings * 2:
            self._arrays.popitem(last=False)
        return array

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.index.iloc[item]
        relative_path = str(row.relative_path)
        signal = self._array(self.model, relative_path)
        quality = self._array("quality", relative_path)
        current_start = int(round(float(row.start) * self.rate))
        context_start = current_start - (self.context_windows - 1) * 2 * self.rate
        window_samples = 10 * self.rate
        segment = signal[:, context_start : current_start + window_samples]
        windows = np.lib.stride_tricks.sliding_window_view(
            segment, window_shape=window_samples, axis=-1
        )[:, :: 2 * self.rate]
        if windows.shape[:2] != (16, self.context_windows):
            raise ValueError(f"invalid cached context shape {windows.shape} for {relative_path}")
        inputs = np.ascontiguousarray(windows.transpose(1, 0, 2))
        if self.model in {"singlem", "cbramod"}:
            inputs = inputs.reshape(self.context_windows, 16, 10, self.rate)
        quality_end = int(round(float(row.start) / 2)) + 1
        quality_context = np.asarray(quality[quality_end - self.context_windows : quality_end])
        label = -1.0 if pd.isna(row.label) else float(row.label)
        return {
            "x": torch.from_numpy(inputs),
            "quality": torch.from_numpy(np.array(quality_context, copy=True)),
            "y": torch.tensor(label, dtype=torch.float32),
            "row_id": int(item),
        }


class CachedFeatureContextDataset(Dataset[dict[str, object]]):
    """Read 31 already-encoded, label-free windows from a frozen feature cache."""

    def __init__(
        self,
        index: pd.DataFrame,
        feature_root: Path,
        model: str,
        *,
        quality_root: Path,
        max_open_recordings: int = 32,
        context_windows: int = 31,
    ) -> None:
        self.index = index.reset_index(drop=True)
        self.feature_root = Path(feature_root)
        self.quality_root = Path(quality_root)
        self.model = model
        self.max_open_recordings = max_open_recordings
        self.context_windows = int(context_windows)
        if self.context_windows < 1:
            raise ValueError("context_windows must be positive")
        self._arrays: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _array(self, kind: str, relative_path: str) -> np.ndarray:
        key = (kind, relative_path)
        if key in self._arrays:
            self._arrays.move_to_end(key)
            return self._arrays[key]
        root = self.feature_root / self.model if kind == "feature" else self.quality_root / "quality"
        path = root / Path(relative_path).with_suffix(".npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._arrays[key] = array
        while len(self._arrays) > self.max_open_recordings * 2:
            self._arrays.popitem(last=False)
        return array

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.index.iloc[item]
        relative_path = str(row.relative_path)
        features = self._array("feature", relative_path)
        quality = self._array("quality", relative_path)
        current = int(round(float(row.start) / 2))
        context = np.asarray(features[current - self.context_windows + 1 : current + 1])
        quality_context = np.asarray(quality[current - self.context_windows + 1 : current + 1])
        if context.shape[:2] != (self.context_windows, 16):
            raise ValueError(f"invalid feature context shape {context.shape} for {relative_path}")
        label = -1.0 if pd.isna(row.label) else float(row.label)
        return {
            "x": torch.from_numpy(np.array(context, dtype=np.float32, copy=True)),
            "quality": torch.from_numpy(np.array(quality_context, copy=True)),
            "y": torch.tensor(label, dtype=torch.float32),
            "row_id": int(item),
        }
