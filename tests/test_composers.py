import numpy as np
import pytest

from qos import Constraints
from helpers import instance
from qos.composers import (ALGORITHMS, GAConfig, brute_force, critical_tightness,
                           genetic_algorithm, greedy_local, greedy_repair, random_search,
                           solve_milp)

SMALL = [(seed, t) for seed in range(4) for t in (0.5, 0.8, 0.9, 0.97)]


@pytest.mark.parametrize("seed,tightness", SMALL)
def test_milp_matches_brute_force(seed, tightness):
    p = instance(5, 5, tightness=tightness, seed=seed)
    exact, brute = solve_milp(p), brute_force(p)
    assert exact.feasible == brute.feasible
    if brute.feasible:
        assert exact.utility == pytest.approx(brute.utility, abs=1e-7)


@pytest.mark.parametrize("seed,tightness", SMALL)
def test_heuristics_never_beat_optimum_and_report_feasibility_honestly(seed, tightness):
    p = instance(5, 5, tightness=tightness, seed=seed)
    opt = brute_force(p)
    for result in (greedy_repair(p), genetic_algorithm(p, seed=seed), random_search(p, seed=seed)):
        assert result.feasible == bool(p.evaluate(result.assignment).feasible[0])
        if result.feasible:
            assert opt.feasible
            assert result.utility <= opt.utility + 1e-9


def test_unconstrained_greedy_local_is_optimal():
    p = instance(8, 10, tightness=0.0, seed=0)
    assert greedy_local(p).utility == pytest.approx(solve_milp(p).utility)


def test_greedy_repair_fixes_violated_local_choice():
    found = 0
    for seed in range(20):
        p = instance(10, 15, tightness=0.86, seed=seed)
        local, repaired = greedy_local(p), greedy_repair(p)
        if not local.feasible and repaired.feasible:
            found += 1
            assert repaired.utility <= local.utility + 1e-9
    assert found > 0


def test_improve_phase_never_lowers_utility():
    for seed in range(10):
        p = instance(10, 15, tightness=0.86, seed=seed)
        only, full = greedy_repair(p, improve=False), greedy_repair(p)
        if only.feasible:
            assert full.feasible and full.utility >= only.utility - 1e-12


def test_ga_is_reproducible_with_seed():
    p = instance(10, 15, tightness=0.86, seed=1)
    a, b = genetic_algorithm(p, seed=42), genetic_algorithm(p, seed=42)
    np.testing.assert_array_equal(a.assignment, b.assignment)
    assert a.evaluations == b.evaluations


def test_ga_history_and_budget():
    p = instance(10, 15, tightness=0.86, seed=1)
    cfg = GAConfig(population_size=40, generations=50, stagnation_limit=None, seed=0)
    r = genetic_algorithm(p, cfg)
    h = r.history
    assert h["generations_run"] == 50
    assert len(h["best_key"]) == 51
    assert all(x <= y + 1e-12 for x, y in zip(h["best_key"], h["best_key"][1:]))  # elitist
    assert r.evaluations == 40 + 50 * (40 - cfg.elite_size)


def test_ga_penalty_mode_runs():
    p = instance(10, 15, tightness=0.86, seed=2)
    r = genetic_algorithm(p, seed=0, constraint_handling="penalty")
    assert r.feasible == bool(p.evaluate(r.assignment).feasible[0])
    with pytest.raises(ValueError):
        genetic_algorithm(p, seed=0, constraint_handling="nope")


def test_hybrid_ga_at_least_as_good_as_greedy_seed():
    for seed in range(5):
        p = instance(10, 15, tightness=0.88, seed=seed)
        g, h = greedy_repair(p), genetic_algorithm(p, seed=seed, seed_with_greedy=True)
        if g.feasible:
            assert h.feasible and h.utility >= g.utility - 1e-12


def test_ga_close_to_optimum_on_medium_instance():
    gaps = []
    for seed in range(5):
        p = instance(10, 20, tightness=0.86, seed=seed)
        opt = solve_milp(p)
        if opt.feasible:
            r = genetic_algorithm(p, seed=seed)
            assert r.feasible
            gaps.append((opt.utility - r.utility) / opt.utility)
    assert gaps and np.mean(gaps) < 0.05


def test_infeasible_problem_reported_by_everyone():
    p = instance(6, 5, seed=0, tightness=0.0)
    p = p.with_constraints(Constraints(max_latency=1.0, max_cost=0.01, min_reliability=0.999))
    for name, run in ALGORITHMS.items():
        assert not run(p, seed=0).feasible, name
    assert solve_milp(p).status == "infeasible"


def test_registry_runs_all_algorithms():
    p = instance(6, 8, tightness=0.5, seed=3)
    for name, run in ALGORITHMS.items():
        r = run(p, seed=0)
        assert r.assignment.shape == (6,)
        assert r.runtime_s >= 0


@pytest.mark.parametrize("seed", range(3))
def test_critical_tightness_is_feasibility_threshold(seed):
    p = instance(5, 5, tightness=0.0, seed=seed)
    tau_star = critical_tightness(p, tol=1e-3)
    assert brute_force(p.with_tightness(tau_star)).feasible
    if tau_star < 1.0:
        assert not brute_force(p.with_tightness(min(1.0, tau_star + 1e-3))).feasible


def test_brute_force_refuses_huge_space():
    with pytest.raises(ValueError):
        brute_force(instance(20, 10, tightness=0.5, seed=0))
