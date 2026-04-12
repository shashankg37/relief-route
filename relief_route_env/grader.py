"""Deterministic scoring for ReliefRoute episodes."""

from __future__ import annotations

from .models import ReliefRouteInfo


STRICT_SCORE_EPSILON = 0.0001


def _strict_unit_interval(value: float) -> float:
    return round(max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, value)), 4)


def score_episode(
    weighted_fulfillment: float,
    on_time_coverage: float,
    efficiency_score: float,
    safety_score: float,
) -> float:
    score = (
        (0.4 * weighted_fulfillment)
        + (0.25 * on_time_coverage)
        + (0.2 * efficiency_score)
        + (0.15 * safety_score)
    )
    return _strict_unit_interval(score)


def grade_observation(info: ReliefRouteInfo) -> float:
    return _strict_unit_interval(info.final_score)
