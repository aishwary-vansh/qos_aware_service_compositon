"""Shared result type and selection rule for all composers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..model import FEAS_TOL, ProblemInstance


@dataclass
class CompositionResult:
    """Outcome of one composer run on one problem instance."""

    algorithm: str
    assignment: np.ndarray
    latency: float
    cost: float
    reliability: float
    utility: float
    violation: float
    feasible: bool
    runtime_s: float
    evaluations: int   # number of compositions whose QoS was computed
    status: str = ""
    history: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Flat dict of scalar fields, ready for a CSV row."""
        return {
            "algorithm": self.algorithm, "feasible": self.feasible,
            "utility": self.utility, "latency": self.latency, "cost": self.cost,
            "reliability": self.reliability, "violation": self.violation,
            "runtime_s": self.runtime_s, "evaluations": self.evaluations,
            "status": self.status,
        }


def selection_key(utility, violation):
    """Scalar ranking that implements Deb's feasibility rules.

    Feasible beats infeasible; two feasible compositions compare by utility;
    two infeasible ones compare by smaller violation. Utility lies in [0, 1],
    so mapping infeasible ones to ``-1 - violation`` keeps them below all
    feasible ones. Higher key is better.
    """
    utility = np.asarray(utility, dtype=float)
    violation = np.asarray(violation, dtype=float)
    return np.where(violation <= FEAS_TOL, utility, -1.0 - violation)


def build_result(problem: ProblemInstance, algorithm: str, assignment, start_time: float,
                 evaluations: int, status: str = "", history: dict | None = None) -> CompositionResult:
    """Evaluate the final assignment and pack it into a CompositionResult."""
    assignment = np.asarray(assignment, dtype=np.intp).copy()
    ev = problem.evaluate(assignment)
    feasible = bool(ev.feasible[0])
    if not status:
        status = "feasible" if feasible else "infeasible"
    return CompositionResult(
        algorithm=algorithm, assignment=assignment,
        latency=float(ev.latency[0]), cost=float(ev.cost[0]),
        reliability=float(ev.reliability[0]), utility=float(ev.utility[0]),
        violation=float(ev.violation[0]), feasible=feasible,
        runtime_s=time.perf_counter() - start_time, evaluations=int(evaluations),
        status=status, history=history or {},
    )
