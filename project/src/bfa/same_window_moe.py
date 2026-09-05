from __future__ import annotations

"""Shared primitives for the 10 s same-window, router-first MoE protocol."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax

from bfa.evaluation.eventize import Event, causal_ema
from bfa.evaluation.match import match_events


EXPERTS = ("singlem", "cbramod", "tcn_gat")
ACTIONS = EXPERTS + ("reject",)
TARGET_COLUMNS = tuple(f"policy_target_{name}" for name in ACTIONS)


def threshold_normalize(probability: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    """Map expert-specific probabilities to a common score with threshold 0.5."""
    p = np.asarray(probability, dtype=float)
    t = np.asarray(threshold, dtype=float)
    if np.any((p < 0) | (p > 1)) or np.any((t <= 0) | (t >= 1)):
        raise ValueError("probabilities must be in [0,1] and thresholds in (0,1)")
    return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1.0 - t))


def threshold_relative_confidence(
    probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Signed correctness margin in [-1,1], comparable across expert thresholds."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    t = np.asarray(thresholds, dtype=float).reshape(-1)
    if p.shape != (len(y), len(t)):
        raise ValueError("probability, label and threshold shapes are incompatible")
    positive = y[:, None] == 1.0
    signed = np.where(positive, p - t, t - p)
    correct_denominator = np.where(positive, 1.0 - t, t)
    wrong_denominator = np.where(positive, t, 1.0 - t)
    denominator = np.where(signed >= 0, correct_denominator, wrong_denominator)
    output = signed / np.maximum(denominator, np.finfo(float).eps)
    output[~np.isfinite(y)] = np.nan
    return np.clip(output, -1.0, 1.0)


def build_policy_targets(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    *,
    temperature: float = 0.10,
    multi_correct_uniform_weight: float = 0.35,
) -> dict[str, np.ndarray]:
    """Build four-action targets for one-window expert competence.

    Reject is legal only for background windows on which every expert is a
    false positive.  All-missed seizure windows select the least-wrong expert.
    """
    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float)
    t = np.asarray(thresholds, dtype=float).reshape(-1)
    confidence = threshold_relative_confidence(p, y, t)
    positive = p >= t[None, :]
    known = np.isfinite(y)
    correct = known[:, None] & (positive == (y[:, None] == 1.0))
    target = np.zeros((len(y), len(ACTIONS)), dtype=np.float64)
    outcome = np.full(len(y), "unknown", dtype=object)

    for index in np.flatnonzero(known):
        seizure = bool(y[index] == 1.0)
        correct_experts = correct[index]
        if not seizure and bool(positive[index].all()):
            target[index, 3] = 1.0
            outcome[index] = "background_all_fp"
            continue
        if seizure and not bool(positive[index].any()):
            logits = np.clip(confidence[index] / temperature, -8.0, 8.0)
            target[index, :3] = softmax(logits)
            outcome[index] = "seizure_all_missed"
            continue
        allowed = correct_experts
        logits = np.where(allowed, np.clip(confidence[index] / temperature, -8.0, 8.0), -np.inf)
        expert_target = softmax(logits)
        if int(allowed.sum()) >= 2:
            uniform = allowed.astype(float) / float(allowed.sum())
            expert_target = (1.0 - multi_correct_uniform_weight) * expert_target + multi_correct_uniform_weight * uniform
        target[index, :3] = expert_target
        if seizure:
            outcome[index] = "seizure_selective" if int(allowed.sum()) < 3 else "seizure_all_correct"
        else:
            outcome[index] = "background_selective" if int(allowed.sum()) < 3 else "background_all_safe"

    eligible = known
    if eligible.any() and not np.allclose(target[eligible].sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("eligible policy targets must sum to one")
    if np.any(target[y == 1.0, 3] != 0):
        raise RuntimeError("reject target mass on seizure must be zero")
    hard = np.full(len(y), -1, dtype=np.int8)
    hard[eligible] = target[eligible].argmax(axis=1).astype(np.int8)
    return {
        "target": target,
        "hard_action": hard,
        "outcome_group": outcome,
        "confidence": confidence,
        "expert_positive": positive,
        "expert_correct": correct,
        "eligible": eligible,
    }


def distribution_audit(frame: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    eligible = frame[frame.policy_action_eligible.astype(bool)].copy()
    rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    total = max(len(eligible), 1)
    for action_id, action in enumerate(ACTIONS):
        hard = int((eligible.policy_action_id == action_id).sum())
        soft = float(eligible[f"policy_target_{action}"].sum())
        hard_fraction = hard / total
        soft_fraction = soft / total
        rows.append({"partition": partition, "action": action, "hard_count": hard, "hard_fraction": hard_fraction, "soft_mass": soft, "soft_fraction": soft_fraction})
        if hard_fraction > 0.75:
            warnings.append({"kind": "hard_action_dominance", "action": action, "fraction": hard_fraction})
        if soft_fraction > 0.75:
            warnings.append({"kind": "soft_action_dominance", "action": action, "fraction": soft_fraction})
        if hard_fraction < 0.01:
            warnings.append({"kind": "low_action_support", "action": action, "fraction": hard_fraction})
    seizure = eligible[eligible.label == 1]
    if len(seizure):
        counts = seizure.policy_action_id.value_counts(normalize=True)
        for action_id in range(3):
            fraction = float(counts.get(action_id, 0.0))
            if fraction > 0.80:
                warnings.append({"kind": "seizure_expert_dominance", "action": ACTIONS[action_id], "fraction": fraction})
    return pd.DataFrame(rows), warnings


def eventize_common(
    times: np.ndarray, scores: np.ndarray, *, threshold: float = 0.5
) -> list[Event]:
    """Smooth a continuous score stream before thresholding and eventization."""
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(times) < 2:
        return []
    smoothed = causal_ema(scores, alpha=1 / 3)
    above = smoothed >= float(threshold)
    step = float(np.median(np.diff(times)))
    candidates: list[Event] = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(above) and above[j + 1]:
            j += 1
        start, end = float(times[i]), float(times[j] + step)
        if end - start >= 4.0:
            candidates.append(Event(start, end, float(smoothed[i : j + 1].max())))
        i = j + 1
    merged: list[Event] = []
    for event in candidates:
        if merged and event.start_s - merged[-1].end_s <= 10.0:
            previous = merged[-1]
            merged[-1] = Event(previous.start_s, max(previous.end_s, event.end_s), max(previous.peak_probability, event.peak_probability))
        else:
            merged.append(event)
    accepted: list[Event] = []
    for event in merged:
        if accepted and event.start_s < accepted[-1].end_s + 30.0:
            continue
        accepted.append(event)
    return accepted


def interval_union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return 0.0
    total = 0.0
    left, right = ordered[0]
    for a, b in ordered[1:]:
        if a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + right - left


def overlap_duration(interval: tuple[float, float], truths: list[tuple[float, float]]) -> float:
    left, right = interval
    return interval_union_duration((max(left, a), min(right, b)) for a, b in truths)


@dataclass(frozen=True)
class ScoreComponents:
    true_positive_events: int
    false_alarm_events: int
    truth_events: int
    false_alarm_seconds: float
    alarm_seconds: float
    monitoring_seconds: float
    nonseizure_seconds: float


def score_events(
    events: list[Event], raw_truths: list[tuple[float, float]], monitoring_start: float, monitoring_end: float
) -> ScoreComponents:
    raw = [(max(monitoring_start, a), min(monitoring_end, b)) for a, b in raw_truths if b > monitoring_start and a < monitoring_end]
    expanded = [(max(monitoring_start, a - 30.0), min(monitoring_end, b + 60.0)) for a, b in raw]
    matched = match_events(events, expanded)
    alarm_seconds = interval_union_duration((event.start_s, event.end_s) for event in events)
    false_alarm_seconds = sum(max(0.0, event.end_s - event.start_s - overlap_duration((event.start_s, event.end_s), raw)) for event in events)
    monitoring_seconds = max(0.0, monitoring_end - monitoring_start)
    nonseizure_seconds = max(0.0, monitoring_seconds - interval_union_duration(raw))
    return ScoreComponents(len(matched.pairs), len(matched.unmatched_predictions), len(raw), false_alarm_seconds, alarm_seconds, monitoring_seconds, nonseizure_seconds)


def aggregate_components(parts: Iterable[ScoreComponents]) -> dict[str, float | int]:
    parts = list(parts)
    tp = sum(x.true_positive_events for x in parts)
    fp = sum(x.false_alarm_events for x in parts)
    truth = sum(x.truth_events for x in parts)
    false_seconds = sum(x.false_alarm_seconds for x in parts)
    alarm_seconds = sum(x.alarm_seconds for x in parts)
    monitoring_seconds = sum(x.monitoring_seconds for x in parts)
    nonseizure_seconds = sum(x.nonseizure_seconds for x in parts)
    return {
        "true_positive_events": tp,
        "false_alarm_events": fp,
        "truth_events": truth,
        "sensitivity": tp / truth if truth else float("nan"),
        "fa_per_24h": fp * 86400.0 / nonseizure_seconds if nonseizure_seconds else float("nan"),
        "false_alarm_time_min_per_24h": false_seconds / 60.0 * 86400.0 / monitoring_seconds if monitoring_seconds else float("nan"),
        "alarm_time_pct": 100.0 * alarm_seconds / monitoring_seconds if monitoring_seconds else float("nan"),
    }


def false_positive_overlap(binary: pd.DataFrame) -> dict[str, float | int]:
    """Window-level FP overlap for aligned test rows with label and expert flags."""
    required = {"label", *(f"positive_{name}" for name in EXPERTS)}
    missing = required - set(binary.columns)
    if missing:
        raise ValueError(f"missing overlap columns: {sorted(missing)}")
    bg = binary[binary.label == 0]
    masks = {name: bg[f"positive_{name}"].astype(bool).to_numpy() for name in EXPERTS}
    output: dict[str, float | int] = {"background_windows": int(len(bg))}
    for i, left in enumerate(EXPERTS):
        for right in EXPERTS[i + 1 :]:
            intersection = int(np.count_nonzero(masks[left] & masks[right]))
            union = int(np.count_nonzero(masks[left] | masks[right]))
            output[f"fp_iou_{left}_{right}"] = intersection / union if union else 0.0
    all_three = masks[EXPERTS[0]] & masks[EXPERTS[1]] & masks[EXPERTS[2]]
    any_fp = masks[EXPERTS[0]] | masks[EXPERTS[1]] | masks[EXPERTS[2]]
    output["all_three_fp_fraction_of_union"] = float(all_three.sum() / any_fp.sum()) if any_fp.any() else 0.0
    for name in EXPERTS:
        others = np.zeros(len(bg), dtype=bool)
        for other in EXPERTS:
            if other != name:
                others |= masks[other]
        only = masks[name] & ~others
        output[f"only_{name}_fp_fraction_of_union"] = float(only.sum() / any_fp.sum()) if any_fp.any() else 0.0
    return output
