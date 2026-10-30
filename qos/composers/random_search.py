"""Random-search baseline: sample compositions uniformly, keep the best."""
from __future__ import annotations

import time

import numpy as np

from ..model import ProblemInstance
from .base import CompositionResult, build_result, selection_key


def random_search(problem: ProblemInstance, n_samples: int = 30_000, seed=None,
                  batch_size: int = 4096) -> CompositionResult:
    """Best of ``n_samples`` uniform random compositions under Deb's rules.

    The default budget is close to the GA's default worst case
    (100 + 300 x 98 = 29,500 evaluations), so the two compare at similar effort.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    best_assignment, best_key = None, -np.inf
    remaining = n_samples
    while remaining > 0:
        size = min(batch_size, remaining)
        batch = rng.integers(0, problem.n_candidates, size=(size, problem.n_tasks))
        ev = problem.evaluate(batch)
        keys = selection_key(ev.utility, ev.violation)
        idx = int(keys.argmax())
        if keys[idx] > best_key:
            best_assignment, best_key = batch[idx].copy(), float(keys[idx])
        remaining -= size
    return build_result(problem, "Random", best_assignment, start, n_samples)
