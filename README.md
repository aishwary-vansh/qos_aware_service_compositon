# QoS-Aware Service Composition under Latency, Cost and Reliability Constraints

Topic 27. Picks one cloud service per workflow task so that end-to-end latency,
cost and reliability constraints all hold, and maximises overall utility.
Compares a greedy composer and a genetic-algorithm composer against an exact
MILP optimum and a random-search baseline.

## Layout

```
qos/
  model.py              problem model: aggregation, utility, violation, tightness
  generator.py          synthetic QoS data generator
  composers/
    greedy.py           Greedy-Local, Greedy-Repair
    ga.py               Genetic algorithm (+ GA+Greedy hybrid)
    exact.py            Optimal-MILP (HiGHS), brute force, critical tightness
    random_search.py    Random-search baseline
    base.py             CompositionResult, Deb's ranking rule
experiments/
  run_experiments.py    runs E1-E6, writes results/raw/
  make_figures.py       writes results/tables/ and results/figures/
results/
  ANALYSIS.md           results and discussion
  figures/              fig1-fig9 (PNG, 200 dpi)
  tables/               summary tables (CSV + Markdown), Wilcoxon tests
  raw/                  per-run results
report/
  report.tex            IEEE-format technical report
  figures/, tables/     built by experiments/make_report_figures.py
scripts/demo.py         run all composers on one instance
tests/                  unit tests (pytest)
docs/design.md          formulation, algorithms, metrics
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

Run the tests:

```bash
python -m pytest -q
```

Run the demo:

```bash
python scripts/demo.py --tasks 10 --candidates 20 --seed 7
```

Run all experiments (about 2 minutes):

```bash
python experiments/run_experiments.py --runs 30
```

Build the figures and tables:

```bash
python experiments/make_figures.py
```

Build the report figures and tables:

```bash
python experiments/make_report_figures.py
```

## Key results (N=10, M=20, 30 instances)

| Composer | Success at ρ = 0.95 | Success at ρ = 1.0 | Gap % at ρ = 0.95 | Runtime |
|---|---|---|---|---|
| Greedy-Local | 0 % | 0 % | – | 0.08 ms |
| Greedy-Repair | 100 % | 17 % | 0.15 | ~1 ms |
| GA | 100 % | 37 % | 0.34 | ~45 ms |
| GA+Greedy | 100 % | 43 % | 0.10 | ~30 ms |
| Random (30k samples) | 3 % | 0 % | – | ~14 ms |

ρ = τ / τ\* is tightness relative to the instance's feasibility limit. See
`results/ANALYSIS.md`.

## Status

- Phase 1, problem model and design: done
- Phase 2, composers and tests: done
- Phase 3, experiments, figures and analysis: done
- Phase 4, report: drafted (`report/report.tex`); needs compiling and author details
- Phase 5, slides and packaging: pending
