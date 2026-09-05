from __future__ import annotations

import math

import pandas as pd


def _distance_to_interval(point: float, start: float, end: float) -> float:
    if start <= point <= end:
        return 0.0
    return min(abs(point - start), abs(point - end))


def build_window_index(
    patient_id: str,
    recording_id: str,
    duration_s: float,
    seizures: list[tuple[float, float]],
    cache_key: str,
    *,
    block_start_s: float = 0.0,
    window_s: float = 10.0,
    stride_s: float = 2.0,
    context_windows: int = 31,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    final_start = math.floor((duration_s - window_s) / stride_s) * stride_s
    starts = [index * stride_s for index in range(int(final_start / stride_s) + 1)]
    history_s = (context_windows - 1) * stride_s
    for start in starts:
        end = start + window_s
        center = (start + end) / 2.0
        distances = [_distance_to_interval(center, left, right) for left, right in seizures]
        minimum_distance = min(distances, default=math.inf)
        label: int | None
        if minimum_distance == 0:
            label = 1
        elif minimum_distance >= 30:
            label = 0
        else:
            label = None
        warmup = start < block_start_s + history_s
        rows.append(
            {
                "patient": patient_id,
                "recording": recording_id,
                "start": start,
                "end": end,
                "context_start": start - history_s,
                "label": label,
                "train_eligible": label is not None and not warmup,
                "warmup": warmup,
                "cache_key": cache_key,
            }
        )
    return pd.DataFrame(rows)
