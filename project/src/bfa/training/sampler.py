from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class PatientBalancedSampler(Sampler[int]):
    """Draw a patient first, then a requested class, only from training rows."""

    def __init__(
        self,
        index: pd.DataFrame,
        batch_size: int,
        positive_fraction: float = 0.30,
        seed: int = 17,
        epoch_size: int | None = None,
    ) -> None:
        patient_column = "patient_id" if "patient_id" in index.columns else "patient"
        if patient_column not in index.columns or "label" not in index.columns:
            raise ValueError("index requires patient (or patient_id) and label columns")
        if not 0 < positive_fraction < 1:
            raise ValueError("positive_fraction must be between zero and one")
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        self.epoch_size = len(index) if epoch_size is None else epoch_size
        self._cursor = 0
        patient_values = index[patient_column].astype(str)
        self._patients = tuple(sorted(patient_values.unique()))
        self._positions: dict[tuple[str, int], np.ndarray] = {}
        for patient in self._patients:
            for label in (0, 1):
                mask = (patient_values == patient) & (index.label == label)
                self._positions[(patient, label)] = np.flatnonzero(mask.to_numpy())
        for label in (0, 1):
            if not any(len(self._positions[(patient, label)]) for patient in self._patients):
                raise ValueError(f"training index has no rows for label {label}")
        rng = np.random.default_rng(seed)
        draws = np.empty(self.epoch_size, dtype=np.int64)
        for draw_index in range(self.epoch_size):
            requested_label = int(rng.random() < self.positive_fraction)
            patient = str(rng.choice(self._patients))
            candidates = self._positions[(patient, requested_label)]
            if len(candidates) == 0:
                eligible = [
                    candidate
                    for candidate in self._patients
                    if len(self._positions[(candidate, requested_label)])
                ]
                patient = str(rng.choice(eligible))
                candidates = self._positions[(patient, requested_label)]
            draws[draw_index] = int(rng.choice(candidates))
        self._draws = draws

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self) -> Iterator[int]:
        while self._cursor < self.epoch_size:
            draw = int(self._draws[self._cursor])
            self._cursor += 1
            yield draw
        self._cursor = 0

    def state_dict(self, *, consumed_cursor: int | None = None) -> dict[str, Any]:
        cursor = self._cursor if consumed_cursor is None else consumed_cursor
        if not 0 <= cursor <= self.epoch_size:
            raise ValueError("consumed_cursor is outside the epoch")
        return {
            "cursor": cursor,
            "draws": self._draws.copy(),
            "epoch_size": self.epoch_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["epoch_size"]) != self.epoch_size:
            raise ValueError("sampler epoch_size differs from checkpoint")
        self._cursor = int(state["cursor"])
        draws = np.asarray(state["draws"], dtype=np.int64)
        if draws.shape != (self.epoch_size,):
            raise ValueError("sampler draw schedule differs from checkpoint")
        self._draws = draws.copy()
