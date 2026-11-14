"""Phase 3: run all experiments and write raw per-run results.

Experiments (each over --runs random instances, fixed seeds):
  E1_tightness   success rate / quality / runtime vs absolute tightness tau
  E1_relative    the same vs relative tightness rho = tau / tau_star
  E2_tasks       scalability in the number of tasks N
  E3_candidates  scalability in the number of candidates per task M
  E4_attribute   single constraints and pairs vs all three together
  E4_critical    per-subset critical tightness tau*_S (MILP bisection only)
  E5_handling    GA constraint handling: Deb's rules vs static penalty
  E6_convergence GA and GA+Greedy convergence (full 300 generations)
  E8_limit       150+ instances at rho in {0.98, 0.99, 1.0} (--limit-runs)
  E9_weights     skewed utility weights on the E1 instances

Every instance is also solved exactly with the MILP, which gives the optimum
and tells whether the instance has any feasible composition.

tau_star is the instance's critical tightness: the largest tau that still has
a feasible composition. Absolute tau mixes two effects: some instances become
unsolvable while others stay easy. Relative tightness rho keeps every instance
solvable (rho <= 1) and puts rho -> 1 at the hard edge of feasibility.

Instances come from real QoS data: QWS 2.0 or WS-DREAM (see qos/datasets.py).

Usage:
  python experiments/run_experiments.py --dataset qws           # 30 runs
  python experiments/run_experiments.py --dataset wsdream --runs 5 --only E1_relative
Output: results/<dataset>/raw/<experiment>.csv and E6_convergence.npz
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# One thread per worker process; parallelism comes from the process pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qos.datasets import make_instance  # noqa: E402
from qos.composers import (critical_tightness, genetic_algorithm, greedy_local,  # noqa: E402
                           greedy_repair, random_search, solve_milp)

MILP_TIME_LIMIT = 60.0
TIGHTNESS_GRID = [0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
RELATIVE_GRID = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
SCALE_RHO = 0.95         # near the feasibility edge, still solvable
MAIN_ALGORITHMS = ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy", "Random"]
# E1 additionally runs the penalty variant, so Deb's rules (plain "GA") and
# lambda=1 can be compared on the same paired instances in the Wilcoxon test.
E1_ALGORITHMS = MAIN_ALGORITHMS + ["GA (penalty λ=1)"]
HANDLING_VARIANTS = ["GA (Deb)", "GA (penalty λ=1)", "GA (penalty λ=10)",
                     "GA (penalty λ=100)", "GA+Greedy"]

# Name -> callable(problem, seed). Variant names are what appear in the CSVs.
VARIANTS = {
    "Greedy-Local": lambda p, s: greedy_local(p),
    "Greedy-Repair": lambda p, s: greedy_repair(p),
    "GA": lambda p, s: genetic_algorithm(p, seed=s),
    "GA+Greedy": lambda p, s: genetic_algorithm(p, seed=s, seed_with_greedy=True),
    "Random": lambda p, s: random_search(p, seed=s),
    "GA (Deb)": lambda p, s: genetic_algorithm(p, seed=s),
    "GA (penalty λ=1)": lambda p, s: genetic_algorithm(
        p, seed=s, constraint_handling="penalty", penalty_coef=1.0),
    "GA (penalty λ=10)": lambda p, s: genetic_algorithm(
        p, seed=s, constraint_handling="penalty", penalty_coef=10.0),
    "GA (penalty λ=100)": lambda p, s: genetic_algorithm(
        p, seed=s, constraint_handling="penalty", penalty_coef=100.0),
}

ATTRIBUTE_MODES = {
    "latency only": lambda t: (t, 0.0, 0.0),
    "cost only": lambda t: (0.0, t, 0.0),
    "reliability only": lambda t: (0.0, 0.0, t),
    "latency + cost": lambda t: (t, t, 0.0),
    "latency + reliability": lambda t: (t, 0.0, t),
    "cost + reliability": lambda t: (0.0, t, t),
    "all three": lambda t: (t, t, t),
}
# E4 levels, relative to the instance's all-three critical tightness tau_star.
# Every level is multiplied by tau_star and clamped to [0, 1] (the valid range
# of tightness). Since tau* is ~0.9, level 1.2 always clamps to tau = 1.0, the
# attribute's own best aggregate, so a SINGLE constraint is feasible there by
# construction: that row cannot show that a single constraint is hard. The
# measured hardness of each subset comes from the E4_critical bisection.
ATTRIBUTE_LEVELS = (0.9, 1.0, 1.2)


def build_tasks(runs: int, dataset: str, limit_runs: int = 150) -> list[dict]:
    """One task = one generated instance plus the algorithms to run on it."""
    tasks = []

    def add(experiment, seed, n, m, label, level, algorithms, tightness=None, rho=None, **extra):
        # Exactly one of: absolute ``tightness`` or relative ``rho``.
        tasks.append(dict(experiment=experiment, dataset=dataset, seed=seed, n=n, m=m,
                          tightness=tightness, rho=rho, label=label, level=level,
                          algorithms=algorithms, **extra))

    for k in range(runs):
        # E1: the same 30 instances at every tightness (paired design).
        for t in TIGHTNESS_GRID:
            add("E1_tightness", k, 10, 20, t, t, E1_ALGORITHMS, tightness=t)
        for rho in RELATIVE_GRID:
            add("E1_relative", k, 10, 20, rho, rho, E1_ALGORITHMS, rho=rho)
        for n in (5, 10, 20, 40):
            add("E2_tasks", 1000 + k, n, 20, n, SCALE_RHO, MAIN_ALGORITHMS, rho=SCALE_RHO)
        for m in (10, 20, 50, 100):
            add("E3_candidates", 2000 + k, 10, m, m, SCALE_RHO, MAIN_ALGORITHMS, rho=SCALE_RHO)
        # E4/E5 share the E1 instances (seed k), so every section measures the
        # same problems and their tau_star values agree.
        for level in ATTRIBUTE_LEVELS:
            for mode in ATTRIBUTE_MODES:
                add("E4_attribute", k, 10, 20, mode, level,
                    ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy"], mode=mode)
        # Per-subset critical tightness: bisect each mode's own limit with the
        # MILP (no heuristics needed). One task per seed covers all 7 modes.
        add("E4_critical", k, 10, 20, "critical", None, [], critical_modes=list(ATTRIBUTE_MODES))
        for rho in (0.9, 0.95, 1.0):
            add("E5_handling", k, 10, 20, rho, rho, HANDLING_VARIANTS, rho=rho)
        add("E6_convergence", 5000 + k, 20, 30, SCALE_RHO, SCALE_RHO, ["Greedy-Repair"],
            rho=SCALE_RHO, convergence=True)
    # E8: many more instances near the feasibility edge, where 30 runs give
    # success rates with ~10pp confidence intervals. Seeds < runs overlap E1.
    for k in range(limit_runs):
        for rho in (0.98, 0.99, 1.0):
            add("E8_limit", k, 10, 20, rho, rho, E1_ALGORITHMS, rho=rho)
    # E9: non-equal utility weights on the same instances as E1.
    for w in ((0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
        for rho in (0.95, 1.0):
            for k in range(runs):
                add("E9_weights", k, 10, 20, f"w={w[0]:g}/{w[1]:g}/{w[2]:g}", rho,
                    MAIN_ALGORITHMS, rho=rho, weights=w)
    return tasks


def _row(task: dict, problem, tightness, tau_star, algorithm: str, result) -> dict:
    t = np.broadcast_to(np.asarray(tightness, dtype=float), (3,))
    return {
        "experiment": task["experiment"], "seed": task["seed"],
        "n_tasks": task["n"], "n_candidates": task["m"],
        "label": task["label"], "level": task["level"],
        "rho": task["rho"], "tau_star": tau_star,
        "tau_latency": t[0], "tau_cost": t[1], "tau_reliability": t[2],
        "max_latency": problem.constraints.max_latency,
        "max_cost": problem.constraints.max_cost,
        "min_reliability": problem.constraints.min_reliability,
        **result.summary(), "algorithm": algorithm,
        "upper_bound": result.history.get("upper_bound", np.nan),
        "w_latency": problem.weights[0], "w_cost": problem.weights[1],
        "w_reliability": problem.weights[2],
    }


def run_task(task: dict):
    """Solve one instance with the MILP and every requested algorithm."""
    weights = task.get("weights")
    base = make_instance(task["dataset"], task["n"], task["m"], seed=task["seed"],
                         weights=weights) if weights is not None else \
        make_instance(task["dataset"], task["n"], task["m"], seed=task["seed"])
    if task.get("critical_modes"):
        # E4_critical: one row per constraint subset, no heuristics. The mask
        # for a mode is its tightness vector at tau = 1.
        rows = []
        for mode in task["critical_modes"]:
            tau = critical_tightness(base, mask=ATTRIBUTE_MODES[mode](1.0),
                                     time_limit=MILP_TIME_LIMIT)
            rows.append(dict(experiment=task["experiment"], seed=task["seed"],
                             n_tasks=task["n"], n_candidates=task["m"], label=mode,
                             tau_critical=tau))
        return rows, None
    tau_star = np.nan
    if task["rho"] is not None:
        tau_star = critical_tightness(base, time_limit=MILP_TIME_LIMIT)
        tightness = task["rho"] * tau_star
    elif task.get("mode") is not None:
        tau_star = critical_tightness(base, time_limit=MILP_TIME_LIMIT)
        tightness = ATTRIBUTE_MODES[task["mode"]](min(1.0, task["level"] * tau_star))
    else:
        tightness = task["tightness"]
    problem = base.with_tightness(tightness)
    opt = solve_milp(problem, time_limit=MILP_TIME_LIMIT)
    rows = [_row(task, problem, tightness, tau_star, "Optimal-MILP", opt)]
    for name in task["algorithms"]:
        result = VARIANTS[name](problem, task["seed"])
        rows.append(_row(task, problem, tightness, tau_star, name, result))

    for row in rows:
        row["opt_feasible"] = opt.feasible
        row["opt_status"] = opt.status
        row["opt_utility"] = opt.utility if opt.feasible else np.nan

    histories = None
    if task.get("convergence"):
        # Full-length runs (no early stop) so all curves have 301 points.
        histories = {"seed": task["seed"], "opt_utility": rows[0]["opt_utility"],
                     "greedy_utility": rows[1]["utility"] if rows[1]["feasible"] else np.nan}
        for key, hybrid in (("GA", False), ("GA+Greedy", True)):
            r = genetic_algorithm(problem, seed=task["seed"], seed_with_greedy=hybrid,
                                  stagnation_limit=None)
            histories[key] = {name: np.asarray(r.history[name], dtype=float)
                              for name in ("best_utility", "feasible_ratio")}
    return rows, histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 experiments.")
    parser.add_argument("--runs", type=int, default=30, help="instances per setting")
    parser.add_argument("--limit-runs", type=int, default=150,
                        help="instances for the near-limit E8 sweep")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--only", nargs="+", help="experiment names to run")
    parser.add_argument("--dataset", choices=("qws", "wsdream"), default="qws")
    parser.add_argument("--out", help="output folder (default results/<dataset>/raw)")
    args = parser.parse_args()

    tasks = build_tasks(args.runs, args.dataset, args.limit_runs)
    if args.only:
        tasks = [t for t in tasks if t["experiment"] in args.only]
    out = pathlib.Path(args.out or ROOT / "results" / args.dataset / "raw")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} instances, {args.jobs} worker processes")

    rows, histories = [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (task_rows, hist) in enumerate(pool.map(run_task, tasks, chunksize=4), 1):
            rows.extend(task_rows)
            if hist is not None:
                histories.append(hist)
            if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done ({time.perf_counter() - start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    for name, group in df.groupby("experiment", sort=False):
        group.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / f'{name}.csv'} ({len(group)} rows)")

    if histories:
        histories.sort(key=lambda h: h["seed"])
        np.savez(out / "E6_convergence.npz",
                 seed=np.array([h["seed"] for h in histories]),
                 opt_utility=np.array([h["opt_utility"] for h in histories]),
                 greedy_utility=np.array([h["greedy_utility"] for h in histories]),
                 **{f"{alg}_{name}": np.vstack([h[alg][name] for h in histories])
                    for alg in ("GA", "GA+Greedy") for name in ("best_utility", "feasible_ratio")})
        print(f"wrote {out / 'E6_convergence.npz'} ({len(histories)} runs)")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
