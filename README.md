# QoS-Aware Service Composition under Latency and Reliability Constraints

**Authors:** Aishwary Vansh and Soham Biswas, Dept. of Computer Science and Engineering,
Indian Institute of Information Technology Sri City, Sri City, India

This repository holds the code, experiments, results and IEEE-format report for Topic 27,
evaluated on two real QoS datasets: **QWS 2.0** and **WS-DREAM**. The report is
[`report/report.pdf`](report/report.pdf).

## The problem (Topic 27)

- **Research problem:** composite cloud services must satisfy multiple QoS constraints
  simultaneously.
- **Research gap:** student implementations of multi-constraint composition
  (latency + reliability) are still limited, often relying on synthetic data or unvalidated heuristics.
- **Proposed solution:** model a set of candidate services with QoS vectors, implement a
  greedy and a genetic-algorithm composer, and compare success rate and solution quality
  under different constraint tightness without using synthetic data.

A composite service is a chain of tasks. Each task can be served by several candidate
services that differ in latency and reliability. We pick one service per task so
that the whole chain meets two limits at once:
- end-to-end latency under a maximum;
- overall reliability above a minimum.

10 tasks with 20 candidates each already give about 10^13 possible compositions.

## What we did

1. **Built the problem model** (`qos/model.py`). Latency adds up across tasks,
   and reliability multiplies, computed as a sum of logs. The model also defines a
   weighted utility in [0, 1] and a normalised constraint violation.
2. **Loaded two real QoS datasets** (`qos/datasets.py`):
   - **QWS 2.0:** 2,507 real web services, with measured response time and reliability.
   - **WS-DREAM:** 339 users calling 5,825 real web services. It gives measured response
     time per user, and reliability as the share of users whose call succeeded.
   - **No synthetic data:** strictly uses measured attributes from both datasets. Neither
     dataset records pricing data, so cost is omitted to adhere to the strict no-synthetic-data requirement.
3. **Implemented the two required composers:**
   - **Greedy-Repair:** starts from the best service per task, swaps services until all
     limits hold, then improves utility while staying feasible.
   - **GA:** a genetic algorithm with integer chromosomes, tournament selection, uniform
     crossover, mutation and elitism. Constraints are handled by Deb's rules or a penalty.
4. **Added comparison methods:**
   - **Greedy-Local:** naive best service per task.
   - **GA+Greedy:** a hybrid that seeds the GA with the greedy result.
   - **Random:** random search with 30,000 samples.
   - **Optimal-MILP:** an exact integer program that gives the true optimum, used to
     measure every heuristic.
5. **Controlled the tightness.** Each instance's feasibility limit τ\* is found exactly.
   Experiments sweep the relative tightness ρ = τ/τ\* up to 1, the edge of feasibility.
6. **Ran the main experiments on each dataset** (E1–E6, E8, E9; 30 instances per
   setting, 150 in E8; 2,220 instances per dataset, 2,460 with E7): tightness, scalability in tasks and candidates, single
   constraints against combined constraints (with each subset's own critical
   tightness bisected by the exact solver), GA constraint handling, GA convergence,
   a 150-instance sweep at ρ ∈ {0.98, 0.99, 1.0}, and skewed utility weights. Paired
   Wilcoxon tests check the main differences.
7. **Ran a whole-dataset experiment** (E7, 240 instances per dataset). Each instance uses
   almost the entire dataset as candidates: 2,500 QWS services, or 5,500 WS-DREAM
   services for one user.
8. **Validated the code** with 71 unit tests.
9. **Wrote the IEEE report** (`report/`). It is 7 pages, follows Topic 27 point by point,
   covers E1–E9 on both datasets, and has 15 references checked against their sources.

## Key results

Success rate, N = 10 tasks and M = 20 candidates, 30 instances per value:

| Composer | QWS, ρ = 0.98 | QWS, ρ = 1.0 | WS-DREAM, ρ = 0.98 | WS-DREAM, ρ = 1.0 | Runtime |
|---|---|---|---|---|---|
| Greedy-Local | 100 % | 3 % | 100 % | 27 % | 0.1 ms |
| Greedy-Repair | 100 % | 90 % | 100 % | 100 % | under 1 ms |
| GA | 100 % | 87 % | 100 % | 100 % | 26–32 ms |
| GA+Greedy | 100 % | 97 % | 100 % | 100 % | 18–21 ms |
| Random | 0 % | 0 % | 60 % | 0 % | about 8 ms |

- **Combining constraints is what makes it hard.** No single constraint binds (its
  critical tightness is 1 by construction), and the combined limit (0.963 on QWS,
  0.993 on WS-DREAM) lies below every individual limit.
- **Greedy-Repair is fastest.** It is optimal up to ρ = 0.98, running in under 1 ms.
- **The GA and the hybrid are robust at the limit.** Over the 150 paired instances
  at ρ = 1, GA+Greedy and GA achieve high success rates at the limit.
- **Naive methods fail.** Greedy-Local and Random break down as constraints tighten.

**Whole datasets (E7).** 10 or 50 tasks over almost all services, up to about 10^102
possible compositions:

| | QWS 2.0 (2,500 services) | WS-DREAM (5,500 services) |
|---|---|---|
| Exact MILP | Optimal in all 240 instances, slowest solve 0.33 s | Optimal in all 240, slowest solve 0.23 s |
| Greedy-Repair at ρ ≤ 0.98 | 100 % success, gap ≤ 0.004 %, 0.3–1.3 ms | 100 % success, gap 0 %, 0.3–0.5 ms |
| Greedy-Repair at ρ = 1 | 63–67 % | 90 % |
| GA at ρ = 1 | 0–13 % | 0–23 % |

- **The exact solver stays fast on whole datasets.** With QoS constraints
  linking the tasks, the LP relaxation leaves very few tasks fractional, so the MILP
  barely needs to search.
- **Greedy-Repair scales best:** near-optimal in about 1 ms.
- **A GA with a fixed budget does not scale** to these search spaces.

Full results: [`results/ANALYSIS.md`](results/ANALYSIS.md).

## Repository layout

```
qos/
  model.py              aggregation, utility, violation, tightness
  datasets.py           QWS 2.0 and WS-DREAM loaders, instances
  composers/
    greedy.py           Greedy-Local, Greedy-Repair
    ga.py               Genetic algorithm and GA+Greedy hybrid
    exact.py            Optimal-MILP, brute force, critical tightness
    random_search.py    Random-search baseline
    base.py             CompositionResult, Deb's ranking rule
experiments/
  run_experiments.py    runs E1-E9 on one dataset (E7 lives in run_full_dataset.py)
  run_full_dataset.py   runs E7 (whole-dataset instances) on one dataset
  make_figures.py       builds tables and figures for one dataset
  make_report_figures.py  builds the report's figures and tables from both datasets
results/
  ANALYSIS.md           results and discussion for both datasets
  qws/, wsdream/        raw/, tables/, figures/ for each dataset
scripts/
  download_data.py      downloads both datasets from Zenodo and checks md5 sums
  demo.py               runs all composers on one instance
tests/                  unit tests
docs/design.md          formulation, data, algorithms and metrics in detail
report/                 IEEE report: report.tex, report.pdf, figures/, tables/
```

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the datasets. They are not committed, because the QWS licence forbids
redistribution:

```bash
python scripts/download_data.py
```

Run the tests:

```bash
python -m pytest -q
```

Run the demo:

```bash
python scripts/demo.py --dataset qws --tightness 0.85 0.9
```

Run all experiments on one dataset (about 1 minute each):

```bash
python experiments/run_experiments.py --dataset qws
```

```bash
python experiments/run_experiments.py --dataset wsdream
```

Run the whole-dataset experiment (E7, a few minutes each):

```bash
python experiments/run_full_dataset.py --dataset qws
```

```bash
python experiments/run_full_dataset.py --dataset wsdream
```

Build the tables and figures:

```bash
python experiments/make_figures.py --dataset qws
```

```bash
python experiments/make_figures.py --dataset wsdream
```

Build the report's figures and tables, then compile it (run the last command from `report/`, twice):

```bash
python experiments/make_report_figures.py
```

```bash
pdflatex report.tex
```

## Data sources

- **QWS 2.0:** Al-Masri and Mahmoud, Zenodo record 3557008. Free for educational and
  non-commercial use. Cite their ICCCN 2007 and WWW 2008 papers.
- **WS-DREAM dataset 1:** Zheng, Zhang and Lyu, Zenodo record 1133476. MIT licence. Cite
  Zheng et al., IEEE Transactions on Services Computing, 2014.
- **Zero Synthetic Data:** To strictly comply with non-synthetic requirements, only real
  empirically measured attributes (response time and reliability) from both datasets are used.

## Tool choice

We used a custom Python model instead of CloudSim, which the problem statement allows.
CloudSim simulates data-center resources, while this problem is a selection over known
QoS values. A custom model also lets us run an exact solver and thousands of repeated
runs.

## Limitations

- Both datasets record measured latency and reliability; cost is not included to avoid synthetic pricing.
- WS-DREAM reliability is an observed success ratio, used as a proxy.
- Services are grouped into tasks at random, because the datasets do not label their
  functions.
- Only sequential workflows are tested.
- In the largest instances (E7, and sizes above the dataset), a service can appear in
  several tasks, but never twice in one task.
- The MILP stops within HiGHS's default 0.01 % relative gap. Reported optimality gaps are
  therefore accurate to about ±0.01 %.

## Publication outlook

| Venue | Current version | After upgrades |
|---|---|---|
| Regional IEEE conference (e.g. INDICON) | Realistic | Good |
| ICSOC workshops | Moderate | Good |
| IEEE SCC 2027 | Low–moderate | Moderate–good (best topic fit) |
| IEEE EDGE 2027 / ICEdge 2027 | Low (needs an edge angle) | Moderate |

**Upgrades that matter most:**
- published baselines (PSO, Grey Wolf Optimizer, a GA with 10× the budget);
- a proof that the MILP relaxation leaves at most K = 3 tasks fractional;
- more QoS constraints or parallel workflows, to show where exact solving slows down;
- for edge venues, an edge-versus-cloud setting with energy as an attribute.

## Submission status

| Deliverable | Status |
|---|---|
| Source code and experiment scripts | Done |
| Results, graphs and tables (both datasets) | Done |
| Technical report | Done: `report/report.pdf`, 7 pages, IEEE format, covers E1–E9 including E7, with the GitHub link in Code Availability |
| Presentation slides | Not started |
| Individual viva | Scheduled by the instructor |
| Peer evaluation form | If requested by the instructor |
