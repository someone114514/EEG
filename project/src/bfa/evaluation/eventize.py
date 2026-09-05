from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    start_s: float
    end_s: float
    peak_probability: float


def causal_ema(values: np.ndarray, alpha: float = 1 / 3) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    output = np.empty_like(array)
    output[0] = array[0]
    for index in range(1, len(array)):
        output[index] = alpha * array[index] + (1 - alpha) * output[index - 1]
    return output


def eventize(
    times: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    ema_alpha: float = 1 / 3,
    min_duration_s: float = 4,
    merge_gap_s: float = 10,
    refractory_s: float = 30,
) -> list[Event]:
    times_array = np.asarray(times, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)
    if times_array.ndim != 1 or probability_array.shape != times_array.shape:
        raise ValueError("times and probabilities must be aligned one-dimensional arrays")
    if len(times_array) < 2 or not np.all(np.diff(times_array) > 0):
        raise ValueError("times must contain at least two strictly increasing values")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must be finite")
    step_s = float(np.median(np.diff(times_array)))
    smoothed = causal_ema(probability_array, ema_alpha)
    above = smoothed >= threshold

    candidates: list[Event] = []
    index = 0
    while index < len(above):
        if not above[index]:
            index += 1
            continue
        end_index = index
        while end_index + 1 < len(above) and above[end_index + 1]:
            end_index += 1
        start_s = float(times_array[index])
        end_s = float(times_array[end_index] + step_s)
        if end_s - start_s >= min_duration_s:
            candidates.append(
                Event(start_s, end_s, float(np.max(smoothed[index : end_index + 1])))
            )
        index = end_index + 1

    merged: list[Event] = []
    for candidate in candidates:
        if merged and candidate.start_s - merged[-1].end_s <= merge_gap_s:
            previous = merged[-1]
            merged[-1] = Event(
                previous.start_s,
                max(previous.end_s, candidate.end_s),
                max(previous.peak_probability, candidate.peak_probability),
            )
        else:
            merged.append(candidate)

    accepted: list[Event] = []
    for candidate in merged:
        if accepted and candidate.start_s < accepted[-1].end_s + refractory_s:
            continue
        accepted.append(candidate)
    return accepted
