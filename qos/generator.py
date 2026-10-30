"""Synthetic generator for candidate-service QoS data.

Value ranges loosely follow the QWS web-service dataset (response times from
tens of ms to seconds, reliability close to 1). The generator builds in the
trade-off that makes the problem hard: services that are fast and reliable
also cost more, so no single candidate is best on all three attributes.
"""
from __future__ import annotations

import numpy as np

from .model import ProblemInstance


def generate_qos(n_tasks: int, n_candidates: int, rng: np.random.Generator,
                 correlation: float = 0.6):
    """Return (latency, cost, reliability) matrices of shape (n_tasks, n_candidates).

    Each candidate gets a hidden "tier" q in [0, 1]. Latency quality and
    reliability quality are mixes of q and independent noise; ``correlation``
    sets how much of q they share. Cost rises with q.
    """
    if n_tasks < 1 or n_candidates < 1:
        raise ValueError("n_tasks and n_candidates must be at least 1")
    if not 0 <= correlation <= 1:
        raise ValueError("correlation must be in [0, 1]")
    shape = (n_tasks, n_candidates)
    tier = rng.random(shape)

    def mixed_quality():
        return np.clip(correlation * tier + (1 - correlation) * rng.random(shape), 0.0, 1.0)

    q_latency = mixed_quality()
    q_reliability = mixed_quality()

    # Some tasks are heavier (more work, higher price) than others.
    latency_scale = rng.uniform(0.5, 1.5, size=(n_tasks, 1))
    cost_scale = rng.uniform(0.5, 1.5, size=(n_tasks, 1))

    latency = latency_scale * (40.0 + 960.0 * (1.0 - q_latency)) * rng.lognormal(0.0, 0.1, shape)
    cost = cost_scale * (0.2 + 9.8 * tier ** 1.5) * rng.lognormal(0.0, 0.1, shape)
    reliability = 0.95 + 0.0499 * q_reliability   # in [0.95, 0.9999]
    return latency, cost, reliability


def generate_instance(n_tasks: int, n_candidates: int, tightness=0.5, seed=None,
                      weights=(1 / 3, 1 / 3, 1 / 3), correlation: float = 0.6) -> ProblemInstance:
    """Generate one random problem instance with constraints set by ``tightness``."""
    rng = np.random.default_rng(seed)
    latency, cost, reliability = generate_qos(n_tasks, n_candidates, rng, correlation)
    return ProblemInstance(latency, cost, reliability, weights=weights, tightness=tightness)
