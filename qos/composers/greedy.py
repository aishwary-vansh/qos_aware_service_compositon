"""Greedy composers.

``greedy_local``  : classic local selection. Each task takes the candidate with
                    the best local utility. Fast, but blind to global constraints.
``greedy_repair`` : local selection, then a constraint-aware repair phase that
                    swaps single services until all constraints hold, then a
                    hill-climbing phase that raises utility while staying feasible.
"""
from __future__ import annotations

import time

import numpy as np

from ..model import FEAS_TOL, ProblemInstance
from .base import CompositionResult, build_result

_EPS = 1e-12


def greedy_local(problem: ProblemInstance) -> CompositionResult:
    """Pick the highest local-utility candidate for every task independently."""
    start = time.perf_counter()
    assignment = problem.local_utility.argmax(axis=1)
    return build_result(problem, "Greedy-Local", assignment, start, evaluations=1)


def _single_swap_table(problem: ProblemInstance, assignment: np.ndarray):
    """QoS of every composition that differs from ``assignment`` in one task.

    Entry [i, j] describes the composition with task i switched to candidate j.
    Returns (utility, violation) arrays of shape (N, M).
    """
    rows = np.arange(problem.n_tasks)
    lat_now = problem.latency[rows, assignment]
    logr_now = problem.log_rel[rows, assignment]
    util_now = problem.local_utility[rows, assignment]

    latency = lat_now.sum() + (problem.latency - lat_now[:, None])
    log_rel = logr_now.sum() + (problem.log_rel - logr_now[:, None])
    utility = util_now.sum() + (problem.local_utility - util_now[:, None])
    return utility, problem.violation_of(latency, log_rel)


def greedy_repair(problem: ProblemInstance, improve: bool = True,
                  max_steps: int | None = None) -> CompositionResult:
    """Greedy local selection followed by repair and (optionally) improvement.

    Repair: while constraints are violated, apply the single swap with the
    largest ratio (violation reduction) / (utility loss). Swaps that reduce
    violation without losing utility are taken first. The phase stops when the
    composition is feasible or no swap reduces violation (a local dead end).

    Improve: while some single swap keeps the composition feasible and raises
    utility, apply the one with the largest gain (steepest ascent).
    """
    start = time.perf_counter()
    n, m = problem.n_tasks, problem.n_candidates
    max_steps = max_steps if max_steps is not None else 10 * n * m
    assignment = problem.local_utility.argmax(axis=1)
    ev = problem.evaluate(assignment)
    utility, violation = float(ev.utility[0]), float(ev.violation[0])
    evaluations = 1
    repair_steps = improve_steps = 0

    while violation > FEAS_TOL and repair_steps < max_steps:
        new_util, new_viol = _single_swap_table(problem, assignment)
        evaluations += n * m
        viol_gain = violation - new_viol
        util_loss = np.maximum(utility - new_util, 0.0)
        score = np.where(viol_gain > _EPS, viol_gain / (util_loss + 1e-9), -np.inf)
        best = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[best]):
            break
        assignment[best[0]] = best[1]
        utility, violation = float(new_util[best]), float(new_viol[best])
        repair_steps += 1

    feasible = violation <= FEAS_TOL
    if improve and feasible:
        while repair_steps + improve_steps < max_steps:
            new_util, new_viol = _single_swap_table(problem, assignment)
            evaluations += n * m
            gain = np.where(new_viol <= FEAS_TOL, new_util - utility, -np.inf)
            best = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[best] <= _EPS:
                break
            assignment[best[0]] = best[1]
            utility = float(new_util[best])
            improve_steps += 1

    status = "feasible" if feasible else "repair_failed"
    history = {"repair_steps": repair_steps, "improve_steps": improve_steps}
    name = "Greedy-Repair" if improve else "Greedy-RepairOnly"
    return build_result(problem, name, assignment, start, evaluations, status, history)
