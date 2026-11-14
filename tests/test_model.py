import numpy as np
import pytest

from qos import Constraints, ProblemInstance, constraints_from_tightness
from helpers import instance


def tiny_problem(constraints=None, tightness=None):
    # 2 tasks x 2 candidates with hand-checkable numbers.
    latency = [[100, 300], [200, 50]]
    cost = [[5, 1], [2, 8]]
    reliability = [[0.99, 0.90], [0.95, 0.98]]
    if constraints is None and tightness is None:
        constraints = Constraints(max_latency=1e9, max_cost=1e9, min_reliability=1e-9)
    return ProblemInstance(latency, cost, reliability, constraints=constraints, tightness=tightness)


def test_aggregation_matches_hand_computation():
    ev = tiny_problem().evaluate([0, 1])
    assert ev.latency[0] == pytest.approx(150)
    assert ev.cost[0] == pytest.approx(13)
    assert ev.reliability[0] == pytest.approx(0.99 * 0.98)


def test_batch_and_single_evaluation_agree():
    p = instance(6, 5, tightness=0.5, seed=1)
    batch = np.random.default_rng(0).integers(0, 5, size=(20, 6))
    ev = p.evaluate(batch)
    for k, a in enumerate(batch):
        single = p.evaluate(a)
        assert single.utility[0] == pytest.approx(ev.utility[k])
        assert single.violation[0] == pytest.approx(ev.violation[k])


def test_bounds():
    b = tiny_problem().bounds
    assert b.latency_best == pytest.approx(150) and b.latency_worst == pytest.approx(500)
    assert b.cost_best == pytest.approx(3) and b.cost_worst == pytest.approx(13)
    assert b.log_rel_best == pytest.approx(np.log(0.99 * 0.98))


def test_utility_equals_saw_formula_and_stays_in_unit_interval():
    p = instance(5, 4, tightness=0.0, seed=3)
    b, w = p.bounds, p.weights
    batch = np.random.default_rng(1).integers(0, 4, size=(200, 5))
    ev = p.evaluate(batch)
    expected = (w[0] * (b.latency_worst - ev.latency) / (b.latency_worst - b.latency_best)
                + w[1] * (b.cost_worst - ev.cost) / (b.cost_worst - b.cost_best)
                + w[2] * (np.log(ev.reliability) - b.log_rel_worst)
                / (b.log_rel_best - b.log_rel_worst))
    np.testing.assert_allclose(ev.utility, expected)
    assert np.all(ev.utility >= -1e-12) and np.all(ev.utility <= 1 + 1e-12)


def test_violation_zero_iff_constraints_hold():
    c = Constraints(max_latency=200, max_cost=10, min_reliability=0.9)
    p = tiny_problem(constraints=c)
    ev = p.evaluate([[0, 1], [1, 0], [0, 0]])
    # [0,1]: lat 150 ok, cost 13 > 10      -> violated
    # [1,0]: lat 500 > 200                  -> violated
    # [0,0]: lat 300 > 200                  -> violated
    assert list(ev.feasible) == [False, False, False]
    loose = p.with_constraints(Constraints(max_latency=200, max_cost=13, min_reliability=0.9))
    assert loose.evaluate([0, 1]).feasible[0]


def test_tightness_zero_makes_everything_feasible():
    p = instance(6, 5, tightness=0.0, seed=2)
    batch = np.random.default_rng(2).integers(0, 5, size=(500, 6))
    assert p.evaluate(batch).feasible.all()


def test_tightness_is_monotone():
    p = instance(8, 10, tightness=0.0, seed=4)
    previous = None
    for t in np.linspace(0, 1, 11):
        c = constraints_from_tightness(p.bounds, t)
        if previous is not None:
            assert c.max_latency <= previous.max_latency + 1e-9
            assert c.max_cost <= previous.max_cost + 1e-9
            assert c.min_reliability >= previous.min_reliability - 1e-12
        previous = c


def test_tightness_one_bounds_equal_best_aggregates():
    p = instance(5, 6, tightness=1.0, seed=5)
    c, b = p.constraints, p.bounds
    assert c.max_latency == pytest.approx(b.latency_best)
    assert c.max_cost == pytest.approx(b.cost_best)
    assert np.log(c.min_reliability) == pytest.approx(b.log_rel_best)


def test_per_attribute_tightness():
    p = instance(5, 6, tightness=(0.0, 1.0, 0.0), seed=6)
    assert p.constraints.max_latency == pytest.approx(p.bounds.latency_worst)
    assert p.constraints.max_cost == pytest.approx(p.bounds.cost_best)


@pytest.mark.parametrize("kwargs", [
    dict(latency=[[1, 2]], cost=[[1]], reliability=[[0.9, 0.9]]),
    dict(latency=[[1, 2]], cost=[[1, 1]], reliability=[[0.9, 1.5]]),
    dict(latency=[[-1, 2]], cost=[[1, 1]], reliability=[[0.9, 0.9]]),
])
def test_invalid_input_rejected(kwargs):
    with pytest.raises(ValueError):
        ProblemInstance(**kwargs, tightness=0.5)


def test_invalid_assignment_rejected():
    with pytest.raises(ValueError):
        tiny_problem().evaluate([0, 2])
    with pytest.raises(ValueError):
        constraints_from_tightness(tiny_problem().bounds, 1.5)
