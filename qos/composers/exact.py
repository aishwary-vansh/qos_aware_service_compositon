"""Exact solvers that give the true optimum for measuring solution quality.

``solve_milp``  : 0-1 integer linear program solved with HiGHS (scipy.optimize.milp).
                  Exact because utility and every aggregate (with reliability
                  in log form) are linear in the selection variables.
``brute_force`` : full enumeration; only for tiny instances, used to check the MILP.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds as VarBounds
from scipy.optimize import LinearConstraint, milp

from ..model import ProblemInstance
from .base import CompositionResult, build_result, selection_key


def solve_milp(problem: ProblemInstance, time_limit: float | None = 60.0,
               feasibility_only: bool = False) -> CompositionResult:
    """Optimal composition via MILP.

    Variables x[i, j] in {0, 1}: task i uses candidate j.
        max  sum_ij u[i, j] x[i, j]
        s.t. sum_j x[i, j] = 1                       for every task i
             sum_ij lat[i, j]  x[i, j] <= max_latency
             sum_ij cost[i, j] x[i, j] <= max_cost
             sum_ij log rel[i, j] x[i, j] >= log min_reliability
    If the problem is infeasible the result has ``feasible=False`` and
    status "infeasible"; its assignment is the unconstrained optimum.

    ``feasibility_only`` drops the objective, so the solver stops at the first
    feasible composition; that is much faster and enough for bisection.
    The result history records HiGHS's proven upper bound on utility
    ("upper_bound") and its node count. When the time limit is hit, the
    composition found so far may not be optimal, and the bound shows how far
    from optimal it can be at most.
    """
    start = time.perf_counter()
    n, m = problem.n_tasks, problem.n_candidates
    c = problem.constraints
    one_per_task = LinearConstraint(sparse.kron(sparse.identity(n), np.ones((1, m)), format="csr"),
                                    1, 1)
    qos = LinearConstraint(
        np.vstack([problem.latency.ravel(), problem.cost.ravel(), -problem.log_rel.ravel()]),
        -np.inf,
        [c.max_latency, c.max_cost, -np.log(c.min_reliability)],
    )
    options = {"time_limit": time_limit} if time_limit is not None else {}
    objective = np.zeros(n * m) if feasibility_only else -problem.local_utility.ravel()
    res = milp(objective, constraints=[one_per_task, qos],
               integrality=np.ones(n * m), bounds=VarBounds(0, 1), options=options)

    if res.x is None:
        status = "infeasible" if res.status == 2 else f"no_solution(status={res.status})"
        assignment = problem.local_utility.argmax(axis=1)
    else:
        status = "optimal" if res.status == 0 else f"time_limit(status={res.status})"
        assignment = res.x.reshape(n, m).argmax(axis=1)
    history = {}
    bound = getattr(res, "mip_dual_bound", None)
    if not feasibility_only and bound is not None and np.isfinite(bound):
        history = {"upper_bound": float(-bound),
                   "node_count": int(getattr(res, "mip_node_count", 0) or 0)}
    return build_result(problem, "Optimal-MILP", assignment, start, evaluations=0,
                        status=status, history=history)


def critical_tightness(problem: ProblemInstance, tol: float = 1e-3,
                       time_limit: float | None = 60.0,
                       mask=(1.0, 1.0, 1.0)) -> float:
    """Largest scalar tightness at which the instance still has a feasible composition.

    ``mask`` selects which attributes are tightened; the others stay at
    tau = 0 (unconstrained). A single-attribute mask always returns 1.0:
    picking each task's own best service reaches that attribute's best
    aggregate. Pairs and the triple can be infeasible before 1.

    Feasible sets are nested (a higher tightness only shrinks them), so
    feasibility is monotone in tightness and bisection with the MILP finds the
    threshold to within ``tol``. Experiments use it to express tightness
    relative to each instance's own limit: rho = tau / tau_star.
    """
    mask = np.asarray(mask, dtype=float)

    def feasible(tightness: float) -> bool:
        # A time-out without a solution counts as infeasible, so tau_star can
        # be slightly underestimated on very large instances.
        return solve_milp(problem.with_tightness(tightness * mask), time_limit,
                          feasibility_only=True).feasible

    if feasible(1.0):
        return 1.0
    lo, hi = 0.0, 1.0   # tightness 0 is always feasible
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def brute_force(problem: ProblemInstance, max_space: int = 2_000_000,
                batch_size: int = 200_000) -> CompositionResult:
    """Enumerate every composition. Refuses search spaces above ``max_space``."""
    start = time.perf_counter()
    n, m = problem.n_tasks, problem.n_candidates
    space = m ** n
    if space > max_space:
        raise ValueError(f"search space {space} exceeds max_space {max_space}")
    weights = m ** np.arange(n - 1, -1, -1)
    best_assignment, best_key = None, -np.inf
    for lo in range(0, space, batch_size):
        codes = np.arange(lo, min(lo + batch_size, space))
        batch = (codes[:, None] // weights) % m   # base-m digits = assignments
        ev = problem.evaluate(batch)
        keys = selection_key(ev.utility, ev.violation)
        idx = int(keys.argmax())
        if keys[idx] > best_key:
            best_assignment, best_key = batch[idx].copy(), float(keys[idx])
    status = "optimal" if best_key >= 0 else "infeasible"
    return build_result(problem, "BruteForce", best_assignment, start, space, status)
