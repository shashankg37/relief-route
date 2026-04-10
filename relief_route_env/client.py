"""Typed client wrapper for ReliefRoute."""

from __future__ import annotations

from openenv.core import SyncEnvClient

from .models import ReliefRouteAction, ReliefRouteObservation, ReliefRouteState


class ReliefRouteEnv(SyncEnvClient[ReliefRouteAction, ReliefRouteObservation, ReliefRouteState]):
    """Typed OpenEnv client for ReliefRoute."""
