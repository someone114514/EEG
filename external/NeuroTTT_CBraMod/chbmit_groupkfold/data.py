from __future__ import annotations
import os

import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

RATE = 200
WINDOW_SECONDS = 10
WINDOW_SAMPLES = RATE * WINDOW_SECONDS
CHANNELS = 16

DEFAULT_ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
DEFAULT_WINDOWS = DEFAULT_ROOT / "manifests/windows.parquet"
DEFAULT_FOLDS = DEFAULT_ROOT / "manifests/groupkfold_cv_v1"
DEFAULT_CACHE = Path(os.environ.get("BFA_CACHE_ROOT", "/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod"))


def fold_manifest(fold: int, fold_root: Path = DEFAULT_FOLDS) -> dict:
    payload = json.loads((fold_root / f"fold_{fold}.json").read_text())
    sets = {key: set(map(str, payload[key])) for key in ("train", "validation", "test")}
    if any(not value for value in sets.values()):
        raise ValueError(f"fold {fold} has an empty partition")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError(f"patient leakage in fold {fold}")
    return payload


def load_rows(fold: int, split: str, windows_path: Path = DEFAULT_WINDOWS, fold_root: Path = DEFAULT_FOLDS) -> pd.DataFrame:
    if split not in {"train", "validation", "test"}:
        raise ValueError(split)
    manifest = fold_manifest(fold, fold_root)
    patients = set(map(str, manifest[split]))
    columns = ["patient", "recording", "start", "end", "label", "warmup", "cache_key", "relative_path"]
    frame = pd.read_parquet(windows_path, columns=columns)
    frame = frame[
        frame.patient.astype(str).isin(patients)
        & frame.label.isin([0.0, 1.0])
        & ~frame.warmup.astype(bool)
    ].copy()
    frame = frame.sort_values(["patient", "recording", "start"], kind="stable").reset_index(drop=True)
    if frame.empty or set(frame.label.astype(int)) != {0, 1}:
        raise RuntimeError(f"fold={fold} split={split} lacks both labels")
    observed = set(frame.patient.astype(str).unique())
    if observed != patients:
        raise RuntimeError(f"fold={fold} split={split} patient mismatch: {sorted(patients - observed)}")
    frame["sample_id"] = frame.cache_key.astype(str) + ":" + frame.start.map(lambda value: f"{float(value):.3f}")
    return frame


class WindowDataset(Dataset):
    """Slice already-scaled 16-channel, 200-Hz windows from mmap EDF views.

    The cache is already in the original CBraMod input unit (microvolts / 100).
    No additional normalization is allowed here.
    """

    def __init__(self, rows: pd.DataFrame, cache_root: Path = DEFAULT_CACHE, max_open: int = 32) -> None:
        self.rows = rows.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.max_open = int(max_open)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _view(self, relative_path: str) -> np.ndarray:
        if relative_path in self._cache:
            view = self._cache.pop(relative_path)
            self._cache[relative_path] = view
            return view
        path = self.cache_root / Path(relative_path).with_suffix(".npy")
        if not path.is_file():
            raise FileNotFoundError(path)
        view = np.load(path, mmap_mode="r", allow_pickle=False)
        if view.ndim != 2 or view.shape[0] != CHANNELS:
            raise ValueError(f"bad cache shape {view.shape}: {path}")
        self._cache[relative_path] = view
        while len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return view

    def __getitem__(self, index: int):
        row = self.rows.iloc[int(index)]
        view = self._view(str(row.relative_path))
        left = int(round(float(row.start) * RATE))
        right = left + WINDOW_SAMPLES
        if left < 0 or right > view.shape[-1]:
            raise IndexError(f"window outside cache at {row.recording}:{row.start}")
        signal = np.asarray(view[:, left:right], dtype=np.float32)
        if signal.shape != (CHANNELS, WINDOW_SAMPLES) or not np.isfinite(signal).all():
            raise ValueError(f"invalid signal at {row.recording}:{row.start}: {signal.shape}")
        signal = np.ascontiguousarray(signal.reshape(CHANNELS, WINDOW_SECONDS, RATE))
        return torch.from_numpy(signal), torch.tensor(float(row.label), dtype=torch.float32), str(row.sample_id)


class PatientBalancedBatchSampler(BatchSampler):
    """Exactly 50/50 labels, then uniformly sample patients within each class."""

    def __init__(self, rows: pd.DataFrame, batch_size: int, steps: int, seed: int) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be positive and even")
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.seed = int(seed)
        self.epoch = 0
        self.by_label: dict[int, dict[str, np.ndarray]] = {}
        for label in (0, 1):
            selected = rows.index[rows.label.astype(int) == label].to_numpy(dtype=np.int64)
            patients = rows.loc[selected, "patient"].astype(str).to_numpy()
            groups = {patient: selected[patients == patient] for patient in sorted(set(patients))}
            if not groups:
                raise ValueError(f"no rows for label={label}")
            self.by_label[label] = groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + 100_003 * self.epoch)
        half = self.batch_size // 2
        patient_lists = {label: np.asarray(list(groups), dtype=object) for label, groups in self.by_label.items()}
        for _ in range(self.steps):
            batch: list[int] = []
            for label in (1, 0):
                patients = rng.choice(patient_lists[label], size=half, replace=True)
                for patient in patients:
                    choices = self.by_label[label][str(patient)]
                    batch.append(int(choices[rng.integers(0, len(choices))]))
            rng.shuffle(batch)
            yield batch


def make_train_loader(
    rows: pd.DataFrame,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    workers: int,
    cache_root: Path = DEFAULT_CACHE,
) -> tuple[DataLoader, PatientBalancedBatchSampler]:
    dataset = WindowDataset(rows, cache_root=cache_root)
    sampler = PatientBalancedBatchSampler(rows, batch_size=batch_size, steps=steps, seed=seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    return loader, sampler


def make_eval_loader(rows: pd.DataFrame, *, batch_size: int, workers: int, cache_root: Path = DEFAULT_CACHE) -> DataLoader:
    return DataLoader(
        WindowDataset(rows, cache_root=cache_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
