from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bfa.evaluation.eventize import Event
from bfa.evaluation.match import match_events


@dataclass(frozen=True)
class EventMetrics:
    true_positive_events: int
    false_alarm_events: int
    truth_events: int
    event_sensitivity: float
    fa_per_24h: float


@dataclass(frozen=True)
class ThresholdScore:
    threshold: float
    event_sensitivity: float
    fa_per_24h: float


def event_metrics(
    predictions: list[Event], truths: list[tuple[float, float]], nonseizure_hours: float
) -> EventMetrics:
    if nonseizure_hours <= 0:
        raise ValueError("nonseizure_hours must be positive")
    matched = match_events(predictions, truths)
    true_positives = len(matched.pairs)
    false_alarms = len(matched.unmatched_predictions)
    sensitivity = true_positives / len(truths) if truths else 1.0
    return EventMetrics(
        true_positive_events=true_positives,
        false_alarm_events=false_alarms,
        truth_events=len(truths),
        event_sensitivity=sensitivity,
        fa_per_24h=false_alarms * 24.0 / nonseizure_hours,
    )


def select_threshold(
    scores: Sequence[ThresholdScore], target_sensitivity: float = 0.80
) -> ThresholdScore:
    if not scores:
        raise ValueError("scores must not be empty")
    feasible = [score for score in scores if score.event_sensitivity >= target_sensitivity]
    if feasible:
        return min(feasible, key=lambda score: (score.fa_per_24h, -score.threshold))
    return min(
        scores,
        key=lambda score: (-score.event_sensitivity, score.fa_per_24h, -score.threshold),
    )
