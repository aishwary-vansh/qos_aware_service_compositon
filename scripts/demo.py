"""Run every composer on one real-data instance at a few tightness levels.

Usage: python scripts/demo.py [--dataset qws] [--tasks 10] [--candidates 20] [--seed 7]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qos.datasets import make_instance  # noqa: E402
from qos.composers import ALGORITHMS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=("qws", "wsdream"), default="qws")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tightness", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    args = parser.parse_args()

    base = make_instance(args.dataset, args.tasks, args.candidates, seed=args.seed)
    print(f"Instance: {args.dataset}, {args.tasks} tasks x {args.candidates} candidates "
          f"(search space {base.search_space_size:.2e})")
    for tightness in args.tightness:
        problem = base.with_tightness(tightness)
        c = problem.constraints
        print(f"\ntightness={tightness:.2f}  Lmax={c.max_latency:.0f} ms  "
              f"Cmax={c.max_cost:.2f}  Rmin={c.min_reliability:.4f}")
        print(f"  {'algorithm':<14}{'feasible':>9}{'utility':>9}{'latency':>9}"
              f"{'cost':>8}{'rel':>8}{'time(ms)':>10}{'evals':>8}")
        for name, run in ALGORITHMS.items():
            r = run(problem, seed=args.seed)
            print(f"  {name:<14}{str(r.feasible):>9}{r.utility:>9.4f}{r.latency:>9.0f}"
                  f"{r.cost:>8.2f}{r.reliability:>8.4f}{r.runtime_s * 1e3:>10.1f}{r.evaluations:>8}")


if __name__ == "__main__":
    main()
