"""Deterministic scoring for ReliefRoute episodes."""

from __future__ import annotations

from .models import ReliefRouteInfo


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
    return round(max(0.0, min(1.0, score)), 4)


def grade_observation(info: ReliefRouteInfo) -> float:
    return round(max(0.0, min(1.0, info.final_score)), 4)
