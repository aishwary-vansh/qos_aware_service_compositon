# Results on Real QoS Data: QWS 2.0 and WS-DREAM

All numbers come from `results/qws/tables/` and `results/wsdream/tables/`. Those tables
are built from the per-run files in `results/<dataset>/raw/`. Every setting uses 30
instances with fixed seeds, and every instance is also solved exactly with the MILP.

To reproduce everything (about 2 minutes per dataset):

```bash
python scripts/download_data.py
```

```bash
python experiments/run_experiments.py --dataset qws
```

```bash
python experiments/run_experiments.py --dataset wsdream
```

```bash
python experiments/make_figures.py --dataset qws
```

```bash
python experiments/make_figures.py --dataset wsdream
```

The whole-dataset experiment (E7) has its own script, which takes a few minutes per dataset:

```bash
python experiments/run_full_dataset.py --dataset qws
```

```bash
python experiments/run_full_dataset.py --dataset wsdream
```

## 1. Data

| | QWS 2.0 | WS-DREAM (dataset 1) |
|---|---|---|
| Services used | 2,507 (none dropped) | 5,577 (248 dropped: valid for under 50 % of users) |
| Latency | Response Time, measured (ms) | Response time seen by one user, measured (ms) |
| Reliability | Reliability column, measured (0.33–0.89) | Share of 339 users with a valid call (0.50–1.00) |
| Cost | AWS EC2 t3 hourly price, by throughput tier | AWS EC2 t3 hourly price, by throughput tier |

**Cost.** Neither dataset has a price. Each service gets the hourly price of the AWS EC2
t3 size (nano to xlarge, us-east-1, $0.0052–$0.1664 per hour) that matches its measured
throughput rank. This produces a real trade-off in both datasets: higher-priced services
have lower latency (Spearman ρ = −0.35 on QWS, −0.54 on WS-DREAM).

**Instances.** N × M distinct services are drawn without replacement and split at random
into N tasks. Neither dataset labels services by function, so the split cannot follow
functions. On WS-DREAM, each instance is built for one random user, using only the
services that user observed successfully.

**Feasibility limit.** For N = 10 and M = 20, the critical tightness τ\* averaged 0.904
on QWS (range 0.876–0.926) and 0.962 on WS-DREAM (range 0.922–0.976). The constraints
therefore bind only at high tightness, which is why all main experiments use relative
tightness ρ = τ / τ\*.

## 2. Did we solve Problem Statement 27?

| Requirement in the problem statement | Where it is met |
|---|---|
| Composite services must satisfy multiple QoS constraints simultaneously | Latency, cost and reliability are hard end-to-end limits enforced together |
| Model candidate services with QoS vectors | Every candidate is a real service with a (latency, cost, reliability) vector from QWS 2.0 or WS-DREAM |
| Implement a greedy composer | Greedy-Local (baseline) and Greedy-Repair (`qos/composers/greedy.py`) |
| Implement a GA-based composer | GA with Deb's rules or penalty, and a GA+Greedy hybrid (`qos/composers/ga.py`) |
| Compare success rate under different constraint tightness | E1 (absolute τ and relative ρ), E4, E5 below |
| Compare solution quality under different constraint tightness | Gap to the exact MILP optimum, E1 to E3 below |
| Tool: custom Java/Python model | Python model, 67 unit tests |

**Yes.** Every element of the proposed solution is implemented and evaluated on two real
QoS datasets. Two caveats must be stated in the report:
- Cost is derived from a real price list, not measured per service.
- WS-DREAM reliability is an observed success ratio, used as a proxy.

## 3. Results

### E1: Success rate vs relative tightness (N = 10, M = 20)

| ρ | QWS: Greedy-Local / Greedy-Repair / GA / GA+Greedy / Random | WS-DREAM: same order |
|---|---|---|
| 0.50–0.85 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| 0.90 | 1.00 / 1.00 / 1.00 / 1.00 / 0.93 | 1.00 / 1.00 / 1.00 / 1.00 / 0.97 |
| 0.95 | 0.73 / 1.00 / 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 1.00 / 1.00 / 0.70 |
| 0.98 | 0.23 / 1.00 / 1.00 / 1.00 / 0.00 | 0.80 / 1.00 / 1.00 / 1.00 / 0.03 |
| **1.00** | 0.00 / **0.63** / **0.70** / **0.90** / 0.00 | 0.00 / **0.63** / **0.83** / **0.80** / 0.00 |

- **Greedy-Local** (picking the best service per task) breaks the limits from ρ = 0.95 on
  QWS and from ρ = 0.98 on WS-DREAM.
- **Random search** collapses from ρ = 0.90–0.95. The search space (about 10^13
  compositions) is far too large to sample blindly.
- **Greedy-Repair, the GA and GA+Greedy** stay at 100 % up to ρ = 0.98 on both datasets.
- **At the feasibility limit (ρ = 1)** the population-based methods are more robust than
  greedy repair:
  - QWS: 63 % (Greedy-Repair), 70 % (GA), 90 % (GA+Greedy).
  - WS-DREAM: 63 %, 83 % and 80 %.

### E1: Solution quality

Mean optimality gap (%) over successful runs:

| ρ | QWS: Greedy-Repair / GA / GA+Greedy / Random | WS-DREAM |
|---|---|---|
| ≤ 0.90 | 0.000 / 0.000 / 0.000 / 5.4–6.2 | 0.000 / 0.000 / 0.000 / 3.5–3.6 |
| 0.95 | 0.003 / 0.004 / 0.003 / – | 0.000 / 0.000 / 0.000 / 2.7 |
| 0.98 | 0.023 / 0.012 / 0.023 / – | 0.000 / 0.006 / 0.000 / 2.3 |

On real data, the three main composers almost always return the exact optimum when they
succeed. The largest mean gap is 0.023 %. Random search stays 2–6 % from the optimum.

Paired Wilcoxon signed-rank tests on normalised quality at ρ = 1 (ties dropped, not
corrected for multiple comparisons):

| Comparison | QWS | WS-DREAM |
|---|---|---|
| GA+Greedy vs Greedy-Repair | better on 8, worse on 0; p = 0.0078 | better on 5, worse on 0; p = 0.0625 |
| GA vs Greedy-Repair | 7 vs 5; p = 0.77 | 8 vs 4; p = 0.14 |

### E1: Runtime per instance (N = 10, M = 20)

| Composer | QWS | WS-DREAM |
|---|---|---|
| Greedy-Local | 0.1–0.3 ms | 0.1 ms |
| Greedy-Repair | 0.3–0.7 ms | 0.3–0.6 ms |
| Random (30,000 samples) | 15–19 ms | 15–16 ms |
| MILP (exact) | 16–22 ms | 16–19 ms |
| GA+Greedy | 30–36 ms | 29–33 ms |
| GA | 44–56 ms | 43–48 ms |

### E2/E3: Scalability at ρ = 0.95

- **Success.** Greedy-Repair, the GA and GA+Greedy keep 100 % success for
  N ∈ {5, 10, 20, 40} and M ∈ {10, 20, 50, 100} on both datasets.
  - On QWS, Greedy-Local falls from 87 % at N = 5 to 43 % at N = 40.
  - Random fails for N ≥ 10 on QWS. On WS-DREAM it falls to 10 % at N = 20 and 0 % at N = 40.
- **Quality.** Greedy-Repair and GA+Greedy stay within 0.006 % of the optimum. The GA's gap
  rises to at most 0.030 % (QWS, M = 100).
- **Runtime at N = 40:**

| Composer | QWS | WS-DREAM |
|---|---|---|
| Greedy-Repair | 0.5 ms | 0.3 ms |
| GA+Greedy | 47 ms | 44 ms |
| MILP | 49 ms | 45 ms |
| GA | 160 ms | 143 ms |

### E4: Single constraints, pairs and all three together

Each constraint group is tightened to a level relative to the instance's all-three
limit τ\*.

| Level | Single constraints | Any pair | All three |
|---|---|---|---|
| 0.9 × τ\* | 30/30 solvable | 30/30 solvable | 30/30 solvable |
| 1.0 × τ\* | 30/30 | 30/30 | 30/30 |
| 1.2 × τ\* | **30/30** | **0/30** | 0/30 |

Results are identical on both datasets. **Every single constraint stays solvable beyond
the all-three limit, while every pair is already infeasible.** The conflict appears as
soon as two QoS attributes are constrained together, which is direct evidence for the
multi-constraint gap in the problem statement.

At 1.0 × τ\*, all three together is the hardest case for every composer:

| | Greedy-Repair | GA | GA+Greedy |
|---|---|---|---|
| QWS, all three | 57 % | 70 % | 70 % |
| WS-DREAM, all three | 77 % | 73 % | 87 % |
| Any pair (both datasets) | ≥ 90 % | ≥ 93 % | ≥ 97 % |

Greedy-Local fails on different attributes in each dataset:
- QWS: reliability (3 % with reliability alone or latency + reliability).
- WS-DREAM: latency (10 % with latency alone, 0 % with latency + cost).

### E5: GA constraint handling at ρ = 1

| | Deb's rules | Penalty λ = 1 | λ = 10 | λ = 100 | GA+Greedy |
|---|---|---|---|---|---|
| QWS success | 0.73 | **0.93** | 0.77 | 0.77 | 0.80 |
| WS-DREAM success | 0.70 | **0.90** | 0.80 | 0.63 | 0.83 |

At ρ = 0.90 and 0.95, all variants reach 100 % with gaps ≤ 0.017 %. At the limit, a
**mild penalty (λ = 1) is the most reliable choice on both datasets**. It lets the
population cross slightly infeasible regions on its way to a feasible one. This was not
tested for significance.

### E6: GA convergence (N = 20, M = 30, ρ = 0.95)

| Generation | 10 | 25 | 50 | 100 | 300 |
|---|---|---|---|---|---|
| QWS: runs with a feasible solution | 3 % | 100 % | 100 % | 100 % | 100 % |
| QWS: mean U/U\* | 0.03 | 0.972 | 0.990 | 0.9985 | 0.9999 |
| WS-DREAM: runs with a feasible solution | 93 % | 100 % | 100 % | 100 % | 100 % |
| WS-DREAM: mean U/U\* | 0.907 | 0.988 | 0.996 | 0.9995 | 1.0000 |
| GA+Greedy (both): mean U/U\* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### E7: Whole-dataset instances

Each instance puts almost the whole dataset into one composition problem: 2,500 of the
2,507 QWS services, or 5,500 of the 5,577 WS-DREAM services observed by one user. There
are 30 random task splits per shape at ρ ∈ {0.90, 0.95, 0.98, 1.0}, which is 240
instances per dataset. Tables: `results/<dataset>/tables/E7_full_summary.md`.

| Shape | Compositions | τ\* mean (range) |
|---|---|---|
| QWS 10 × 250 | about 10^24 | 0.947 (0.938–0.958) |
| QWS 50 × 50 | about 10^85 | 0.935 (0.930–0.940) |
| WS-DREAM 10 × 550 | about 10^27 | 0.995 (0.989–0.997) |
| WS-DREAM 50 × 110 | about 10^102 | 0.989 (0.978–0.994) |

Success rate at the feasibility limit (ρ = 1):

| Shape | Greedy-Repair | GA | GA+Greedy | Greedy-Local | Random |
|---|---|---|---|---|---|
| QWS 10 × 250 | 0.67 | 0.13 | 0.70 | 0.00 | 0.00 |
| QWS 50 × 50 | 0.63 | 0.00 | 0.63 | 0.00 | 0.00 |
| WS-DREAM 10 × 550 | 0.90 | 0.23 | 0.90 | 0.00 | 0.00 |
| WS-DREAM 50 × 110 | 0.90 | 0.00 | 0.90 | 0.00 | 0.00 |

**At ρ ≤ 0.98:**
- Greedy-Repair and GA+Greedy succeed in 100 % of instances, within 0.004 % of the
  optimum.
- The GA succeeds in 97–100 %, but its gap is 0.06–0.49 %.
- Greedy-Local already fails at ρ = 0.98 on QWS: 50 % for 10 × 250, 0 % for 50 × 50.
- Random fails almost everywhere.

**Mean runtime per instance:**

| Composer | QWS | WS-DREAM |
|---|---|---|
| Greedy-Repair | 0.3–2.4 ms | 0.3–3.4 ms |
| GA+Greedy | 22–38 ms | 21–36 ms |
| GA | 61–135 ms | 70–131 ms |
| MILP (exact) | 48–130 ms; slowest single solve 0.33 s | 82–152 ms; slowest 0.23 s |

**The MILP solved all 480 instances to optimality,** within HiGHS's default 0.01 %
gap. There are two reasons:
- **Only 3 QoS constraints link the tasks.** A basic solution of the LP relaxation then
  leaves at most 3 tasks split between candidates, so the solver needs very little
  branching.
- **Its "optimum" is within 0.01 % of the true optimum.** In 2 QWS and 8 WS-DREAM
  instances, Greedy-Repair found a marginally better composition. The largest gap
  between the MILP's solution and its proven bound was 0.01 %.

## 4. Main findings

1. **Combining constraints is the source of difficulty.** Every single constraint is
   solvable past the point where every pair is infeasible, on both real datasets (E4).
2. **Greedy-Repair is the fastest near-optimal composer.** It takes under 1 ms and is
   optimal or within 0.023 % up to ρ = 0.98. It is the weakest of the three at the exact
   feasibility limit (63 %).
3. **The GA is more robust at the limit, but only on small instances.** At N = 10 and
   M = 20 it reaches 70–83 % at ρ = 1, and mild penalty handling (λ = 1) raises this to
   90–93 %. On whole datasets (E7) its fixed budget of about 30,000 evaluations is
   too small: 0–23 % at ρ = 1, and gaps up to 0.49 %.
4. **GA+Greedy is the best default.** It has 80–90 % success at the limit on small
   instances, is never worse than Greedy-Repair, and is faster than the plain GA. On
   whole datasets its results come almost entirely from the greedy seed.
5. **Naive methods fail on real data.** Greedy-Local and Random break down as constraints
   approach the limit and as the number of tasks grows.
6. **Exact solving is practical for three QoS constraints, even on whole datasets.**
   The MILP was optimal in every instance, at most 0.33 s with up to 5,500 binary
   variables (E7). The case for heuristics is speed, about 1 ms for Greedy-Repair, and
   settings a linear model cannot handle. It is not that the exact solver is too slow.

## 5. Limitations

- **Cost is derived** from a public price list by throughput tier, not measured per
  service.
- **WS-DREAM reliability is a proxy:** the share of users whose call returned a valid
  measurement.
- **Task grouping is random,** because the datasets do not label service functions.
- **Only sequential workflows** were tested.
- **Runtimes carry some noise:** they were measured with 10 parallel worker processes.
- **MILP tolerance:** "optimal" means within HiGHS's default 0.01 % relative gap, so
  reported optimality gaps are accurate to about ±0.01 %.
- **Only three QoS constraints:** with more, the MILP may slow down. This was not tested.
- **The report covers E1–E6 only.** E7 is in this file and the README, but not yet in
  `report/report.tex`.
