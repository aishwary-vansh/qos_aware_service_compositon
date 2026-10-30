"""Composition algorithms. Every composer takes a ProblemInstance and returns a CompositionResult."""
from .base import CompositionResult, build_result, selection_key
from .exact import brute_force, critical_tightness, solve_milp
from .ga import GAConfig, genetic_algorithm
from .greedy import greedy_local, greedy_repair
from .random_search import random_search

# Name -> callable(problem, seed). Phase 3 experiments iterate over this.
ALGORITHMS = {
    "Greedy-Local": lambda p, seed=None: greedy_local(p),
    "Greedy-Repair": lambda p, seed=None: greedy_repair(p),
    "GA": lambda p, seed=None: genetic_algorithm(p, seed=seed),
    "GA+Greedy": lambda p, seed=None: genetic_algorithm(p, seed=seed, seed_with_greedy=True),
    "Random": lambda p, seed=None: random_search(p, seed=seed),
    "Optimal-MILP": lambda p, seed=None: solve_milp(p),
}

__all__ = [
    "ALGORITHMS", "CompositionResult", "GAConfig", "brute_force", "build_result",
    "critical_tightness",
    "genetic_algorithm", "greedy_local", "greedy_repair", "random_search",
    "selection_key", "solve_milp",
]
