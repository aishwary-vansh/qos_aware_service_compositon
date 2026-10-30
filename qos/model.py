"""QoS model for sequential composite cloud services.

A composite service is a chain of N abstract tasks. Each task has M concrete
candidate services. Each candidate has a QoS vector (latency, cost, reliability).
A composition (also called an assignment) picks one candidate per task and is
stored as an integer vector ``a`` of length N with ``a[i]`` in ``[0, M)``.

Aggregation for a sequential workflow:
    latency     L(a) = sum_i lat[i, a_i]
    cost        C(a) = sum_i cost[i, a_i]
    reliability R(a) = prod_i rel[i, a_i]   (handled as sum of logs)

All functions accept either one assignment (shape (N,)) or a batch of
assignments (shape (P, N)) so the composers can evaluate whole populations
with numpy in one call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Violations at or below this value count as feasible. It absorbs float
# rounding, e.g. a MILP solution that sits exactly on a constraint bound.
FEAS_TOL = 1e-7


@dataclass(frozen=True)
class Constraints:
    """Global end-to-end QoS constraints for the composite service."""

    max_latency: float      # milliseconds
    max_cost: float         # currency units per invocation
    min_reliability: float  # probability in (0, 1]


@dataclass(frozen=True)
class Bounds:
    """Best and worst achievable aggregate value of each QoS attribute.

    Each bound is reached by optimising one attribute alone, so the three
    "best" values are usually not reachable by the same composition.
    """

    latency_best: float
    latency_worst: float
    cost_best: float
    cost_worst: float
    log_rel_best: float
    log_rel_worst: float


@dataclass
class Evaluation:
    """Aggregated QoS, utility and constraint violation for a batch."""

    latency: np.ndarray
    cost: np.ndarray
    reliability: np.ndarray
    utility: np.ndarray
    violation: np.ndarray

    @property
    def feasible(self) -> np.ndarray:
        return self.violation <= FEAS_TOL


def constraints_from_tightness(bounds: Bounds, tightness) -> Constraints:
    """Build constraints by interpolating between worst and best aggregates.

    ``tightness`` is a scalar or a (latency, cost, reliability) triple in [0, 1].
    0 puts each bound at the worst aggregate value, so every composition is
    feasible. 1 puts each bound at the best aggregate value of that attribute,
    which usually no single composition meets for all three at once.
    Reliability is interpolated in log space because it aggregates as a product.
    """
    t = np.broadcast_to(np.asarray(tightness, dtype=float), (3,))
    if np.any(t < 0) or np.any(t > 1):
        raise ValueError(f"tightness must be in [0, 1], got {tightness!r}")
    t_lat, t_cost, t_rel = t
    b = bounds
    max_latency = b.latency_best + (1 - t_lat) * (b.latency_worst - b.latency_best)
    max_cost = b.cost_best + (1 - t_cost) * (b.cost_worst - b.cost_best)
    log_rmin = b.log_rel_best - (1 - t_rel) * (b.log_rel_best - b.log_rel_worst)
    return Constraints(float(max_latency), float(max_cost), float(np.exp(log_rmin)))


class ProblemInstance:
    """One multi-constraint QoS-aware service composition problem.

    Objective: maximise the weighted utility U(a) subject to
        L(a) <= max_latency, C(a) <= max_cost, R(a) >= min_reliability.

    Utility is simple additive weighting (SAW) over aggregate QoS, each
    attribute normalised to [0, 1] with the bounds above:
        U = wL (Lw - L)/(Lw - Lb) + wC (Cw - C)/(Cw - Cb)
            + wR (log R - log Rw)/(log Rb - log Rw)
    Because every aggregate is a sum over tasks, U splits exactly into a sum
    of per-service terms ``local_utility[i, a_i]``. The MILP solver and the
    greedy composer rely on this.
    """

    def __init__(self, latency, cost, reliability, constraints: Constraints | None = None,
                 weights=(1 / 3, 1 / 3, 1 / 3), tightness=None):
        self.latency = np.asarray(latency, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.reliability = np.asarray(reliability, dtype=float)
        if self.latency.ndim != 2:
            raise ValueError("QoS matrices must be 2-D with shape (n_tasks, n_candidates)")
        if not (self.latency.shape == self.cost.shape == self.reliability.shape):
            raise ValueError("latency, cost and reliability must have the same shape")
        if np.any(self.latency < 0) or np.any(self.cost < 0):
            raise ValueError("latency and cost must be non-negative")
        if np.any(self.reliability <= 0) or np.any(self.reliability > 1):
            raise ValueError("reliability must be in (0, 1]")

        w = np.asarray(weights, dtype=float)
        if w.shape != (3,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be three non-negative numbers with a positive sum")
        self.weights = w / w.sum()

        self.log_rel = np.log(self.reliability)
        self.bounds = Bounds(
            latency_best=float(self.latency.min(axis=1).sum()),
            latency_worst=float(self.latency.max(axis=1).sum()),
            cost_best=float(self.cost.min(axis=1).sum()),
            cost_worst=float(self.cost.max(axis=1).sum()),
            log_rel_best=float(self.log_rel.max(axis=1).sum()),
            log_rel_worst=float(self.log_rel.min(axis=1).sum()),
        )
        # Normalisation ranges. A zero range (all candidates equal) would
        # divide by zero; that attribute then contributes nothing anyway.
        b = self.bounds
        ranges = np.array([b.latency_worst - b.latency_best,
                           b.cost_worst - b.cost_best,
                           b.log_rel_best - b.log_rel_worst])
        self._ranges = np.where(ranges > 0, ranges, 1.0)
        self.local_utility = self._compute_local_utility()

        if constraints is None:
            if tightness is None:
                raise ValueError("give either constraints or tightness")
            constraints = constraints_from_tightness(self.bounds, tightness)
        self.constraints = constraints
        self.tightness = tightness

    # ------------------------------------------------------------------ shape
    @property
    def n_tasks(self) -> int:
        return self.latency.shape[0]

    @property
    def n_candidates(self) -> int:
        return self.latency.shape[1]

    @property
    def search_space_size(self) -> int:
        return self.n_candidates ** self.n_tasks

    # -------------------------------------------------------------- variants
    def with_constraints(self, constraints: Constraints) -> "ProblemInstance":
        """Same candidate services, different constraints."""
        return ProblemInstance(self.latency, self.cost, self.reliability,
                               constraints=constraints, weights=self.weights)

    def with_tightness(self, tightness) -> "ProblemInstance":
        """Same candidate services, constraints rebuilt for a new tightness."""
        return ProblemInstance(self.latency, self.cost, self.reliability,
                               weights=self.weights, tightness=tightness)

    # ------------------------------------------------------------ evaluation
    def _compute_local_utility(self) -> np.ndarray:
        w_lat, w_cost, w_rel = self.weights
        r_lat, r_cost, r_rel = self._ranges
        lat_term = (self.latency.max(axis=1, keepdims=True) - self.latency) / r_lat
        cost_term = (self.cost.max(axis=1, keepdims=True) - self.cost) / r_cost
        rel_term = (self.log_rel - self.log_rel.min(axis=1, keepdims=True)) / r_rel
        return w_lat * lat_term + w_cost * cost_term + w_rel * rel_term

    def violation_of(self, latency, cost, log_rel):
        """Normalised total constraint violation; 0 means feasible.

        Each term is the amount by which one constraint is exceeded, divided
        by that attribute's range, so the three terms are comparable.
        Works element-wise on arrays of any shape.
        """
        c = self.constraints
        r_lat, r_cost, r_rel = self._ranges
        return (np.maximum(0.0, latency - c.max_latency) / r_lat
                + np.maximum(0.0, cost - c.max_cost) / r_cost
                + np.maximum(0.0, np.log(c.min_reliability) - log_rel) / r_rel)

    def evaluate(self, assignments) -> Evaluation:
        """Aggregate QoS for one assignment (N,) or a batch (P, N).

        Always returns 1-D arrays of length P (P = 1 for a single assignment).
        """
        a = np.atleast_2d(np.asarray(assignments, dtype=np.intp))
        if a.shape[1] != self.n_tasks:
            raise ValueError(f"assignment length {a.shape[1]} != n_tasks {self.n_tasks}")
        if a.min() < 0 or a.max() >= self.n_candidates:
            raise ValueError("assignment contains an invalid candidate index")
        rows = np.arange(self.n_tasks)
        latency = self.latency[rows, a].sum(axis=1)
        cost = self.cost[rows, a].sum(axis=1)
        log_rel = self.log_rel[rows, a].sum(axis=1)
        utility = self.local_utility[rows, a].sum(axis=1)
        return Evaluation(latency=latency, cost=cost, reliability=np.exp(log_rel),
                          utility=utility,
                          violation=self.violation_of(latency, cost, log_rel))

    def __repr__(self) -> str:
        c = self.constraints
        return (f"ProblemInstance(n_tasks={self.n_tasks}, n_candidates={self.n_candidates}, "
                f"tightness={self.tightness}, max_latency={c.max_latency:.1f}, "
                f"max_cost={c.max_cost:.2f}, min_reliability={c.min_reliability:.4f})")
