"""E7: whole-dataset instances.

Instead of sampling a few hundred services per instance, each instance puts
(almost) the entire dataset into one composition problem:
  QWS 2.0   : 2,500 of 2,507 services, as 10 x 250 or 50 x 50
  WS-DREAM  : 5,500 of 5,577 services, as 10 x 550 or 50 x 110
              (one user per instance; 303 of 339 users observed that many)
Each of the 30 seeds gives a different random split of services into tasks
(and, for WS-DREAM, a different user).

Usage: python experiments/run_full_dataset.py --dataset qws
Output: results/<dataset>/raw/E7_full.csv, results/<dataset>/tables/E7_full_summary.{csv,md}
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from make_figures import FMT, add_derived, summarise, to_markdown  # noqa: E402
from run_experiments import MAIN_ALGORITHMS, ROOT, run_task  # noqa: E402

SIZES = {"qws": [(10, 250), (50, 50)], "wsdream": [(10, 550), (50, 110)]}
RHO_LEVELS = (0.90, 0.95, 0.98, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run whole-dataset instances (E7).")
    parser.add_argument("--dataset", choices=tuple(SIZES), default="qws")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    tasks = [dict(experiment="E7_full", dataset=args.dataset, seed=6000 + k, n=n, m=m,
                  tightness=None, rho=rho, label=f"{n}x{m}", level=rho,
                  algorithms=MAIN_ALGORITHMS)
             for (n, m) in SIZES[args.dataset] for rho in RHO_LEVELS for k in range(args.runs)]
    print(f"{len(tasks)} whole-dataset instances on {args.dataset}, {args.jobs} workers")
    rows, start = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (task_rows, _) in enumerate(pool.map(run_task, tasks), 1):
            rows.extend(task_rows)
            if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done ({time.perf_counter() - start:.0f}s)", flush=True)

    out = ROOT / "results" / args.dataset
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw" / "E7_full.csv", index=False)
    summary = summarise(add_derived(df), ["label", "level"])
    summary.to_csv(out / "tables" / "E7_full_summary.csv", index=False)
    to_markdown(summary, out / "tables" / "E7_full_summary.md", FMT)
    print(f"wrote {out / 'raw' / 'E7_full.csv'} and E7_full_summary; "
          f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
