"""Genetic-algorithm composer.

Encoding: integer chromosome of length N; gene i is the candidate index chosen
for task i, so every chromosome is a valid composition (no decoding step).

Operators: tournament selection, uniform crossover, per-gene random-reset
mutation, elitism. Constraints are handled with Deb's feasibility rules
(default) or a static penalty. The population is evaluated as one numpy batch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from ..model import FEAS_TOL, ProblemInstance
from .base import CompositionResult, build_result, selection_key
from .greedy import greedy_repair


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    generations: int = 300
    crossover_rate: float = 0.9
    mutation_rate: float | None = None    # per gene; None means 1 / n_tasks
    tournament_size: int = 3
    elite_size: int = 2
    stagnation_limit: int | None = 80     # stop after this many generations without improvement
    constraint_handling: str = "deb"      # "deb" or "penalty"
    penalty_coef: float = 10.0
    seed_with_greedy: bool = False        # hybrid: inject the Greedy-Repair result
    seed: int | None = None


def _fitness(problem: ProblemInstance, population: np.ndarray, cfg: GAConfig):
    ev = problem.evaluate(population)
    if cfg.constraint_handling == "deb":
        fitness = selection_key(ev.utility, ev.violation)
    elif cfg.constraint_handling == "penalty":
        fitness = ev.utility - cfg.penalty_coef * ev.violation
    else:
        raise ValueError(f"unknown constraint_handling {cfg.constraint_handling!r}")
    return fitness, ev


def _tournament(rng: np.random.Generator, fitness: np.ndarray, n_winners: int, k: int) -> np.ndarray:
    """Indices of ``n_winners`` tournament winners, each from ``k`` random entrants."""
    entrants = rng.integers(0, fitness.size, size=(n_winners, k))
    return entrants[np.arange(n_winners), fitness[entrants].argmax(axis=1)]


def genetic_algorithm(problem: ProblemInstance, config: GAConfig | None = None,
                      **overrides) -> CompositionResult:
    """Run the GA and return the best composition found.

    The best composition is tracked with Deb's rules whatever the fitness
    mode, so the result is the best feasible composition if one was seen,
    otherwise the least-violating one.
    """
    cfg = replace(config or GAConfig(), **overrides)
    if cfg.population_size < 2 or cfg.elite_size >= cfg.population_size:
        raise ValueError("need population_size >= 2 and elite_size < population_size")
    start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n, m = problem.n_tasks, problem.n_candidates
    pop_size = cfg.population_size
    mutation_rate = cfg.mutation_rate if cfg.mutation_rate is not None else 1.0 / n
    evaluations = 0

    population = rng.integers(0, m, size=(pop_size, n))
    name = "GA"
    if cfg.seed_with_greedy:
        name = "GA+Greedy"
        seed_result = greedy_repair(problem)
        evaluations += seed_result.evaluations
        population[0] = seed_result.assignment
        population[1] = problem.local_utility.argmax(axis=1)

    fitness, ev = _fitness(problem, population, cfg)
    evaluations += pop_size
    keys = selection_key(ev.utility, ev.violation)
    best_idx = int(keys.argmax())
    best_assignment, best_key = population[best_idx].copy(), float(keys[best_idx])

    history = {"best_utility": [], "best_key": [], "mean_fitness": [], "feasible_ratio": []}

    def record():
        history["best_utility"].append(best_key if best_key >= 0 else np.nan)
        history["best_key"].append(best_key)
        history["mean_fitness"].append(float(fitness.mean()))
        history["feasible_ratio"].append(float((ev.violation <= FEAS_TOL).mean()))

    record()
    n_children = pop_size - cfg.elite_size
    n_pairs = (n_children + 1) // 2
    stagnant = 0
    generation = 0
    for generation in range(1, cfg.generations + 1):
        elites = population[np.argsort(-fitness)[:cfg.elite_size]]

        parent_a = population[_tournament(rng, fitness, n_pairs, cfg.tournament_size)]
        parent_b = population[_tournament(rng, fitness, n_pairs, cfg.tournament_size)]
        do_cross = rng.random((n_pairs, 1)) < cfg.crossover_rate
        take_a = rng.random((n_pairs, n)) < 0.5
        mask = np.where(do_cross, take_a, True)   # no crossover: copy parents as they are
        child_a = np.where(mask, parent_a, parent_b)
        child_b = np.where(mask, parent_b, parent_a)
        children = np.vstack([child_a, child_b])[:n_children]

        # Random-reset mutation to a different candidate of the same task.
        mutate = rng.random(children.shape) < mutation_rate
        if m > 1:
            shift = rng.integers(1, m, size=children.shape)
            children = np.where(mutate, (children + shift) % m, children)

        population = np.vstack([elites, children])
        fitness, ev = _fitness(problem, population, cfg)
        evaluations += n_children

        keys = selection_key(ev.utility, ev.violation)
        idx = int(keys.argmax())
        if keys[idx] > best_key + 1e-12:
            best_assignment, best_key = population[idx].copy(), float(keys[idx])
            stagnant = 0
        else:
            stagnant += 1
        record()
        if cfg.stagnation_limit is not None and stagnant >= cfg.stagnation_limit:
            break

    history["generations_run"] = generation
    status = "feasible" if best_key >= 0 else "no_feasible_found"
    return build_result(problem, name, best_assignment, start, evaluations, status, history)
