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
| Tool: custom Java/Python model | Python model, 73 unit tests |

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

Mean wall-clock range over ρ ∈ {0.95, 0.98, 1.0}, measured with 10 parallel workers
(see Limitations):

| Composer | QWS | WS-DREAM |
|---|---|---|
| Greedy-Local | 0.1 ms | 0.1 ms |
| Greedy-Repair | 0.2–0.5 ms | 0.2–0.4 ms |
| Random (30,000 samples) | 11–12 ms | 13–16 ms |
| MILP (exact) | 6–11 ms | 9–14 ms |
| GA+Greedy | 21–28 ms | 27–31 ms |
| GA | 28–37 ms | 36–45 ms |

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
| Greedy-Repair | 0.4 ms | 0.2 ms |
| GA+Greedy | 32 ms | 43 ms |
| MILP | 25 ms | 29 ms |
| GA | 124 ms | 136 ms |

### E4: Single constraints, pairs and all three together

Each constraint group is tightened to a level relative to the instance's all-three
limit τ\*. At the 1.2 level the tightness vector clamps to τ = 1 (the valid range is
[0, 1]), where a single constraint is feasible by construction, so the hardness of each
subset is measured by its own critical tightness instead (`E4_critical`):

| Subset | QWS τ\*_S mean | WS-DREAM τ\*_S mean |
|---|---|---|
| Latency only | 1.000 | 1.000 |
| Cost only | 1.000 | 1.000 |
| Reliability only | 1.000 | 1.000 |
| Latency + cost | 0.971 | 0.970 |
| Latency + reliability | 0.963 | 0.993 |
| Cost + reliability | 0.921 | 0.988 |
| All three | **0.904** | **0.962** |

A single attribute never binds: picking each task's own best service attains that
attribute's best aggregate. Each pair costs 2.9–7.9 pp of headroom on QWS and
0.7–3.0 pp on WS-DREAM, and the all-three limit is below every pair's.

At the all-three limit (1.0 × τ\*), all three together is the hardest case:

| | Greedy-Repair | GA | GA+Greedy |
|---|---|---|---|
| QWS, all three | 63 % | 70 % | 90 % |
| WS-DREAM, all three | 63 % | 83 % | 80 % |
| Any pair, QWS | ≥ 90 % | 100 % | 100 % |
| Any pair, WS-DREAM | 100 % | 100 % | 100 % |

Greedy-Local fails on different attributes in each dataset:
- QWS: reliability (10 % with reliability alone, 7 % with latency + reliability).
- WS-DREAM: latency (37 % with latency alone, 3 % with latency + cost).

### E5: GA constraint handling at ρ = 1

| | Deb's rules | Penalty λ = 1 | λ = 10 | λ = 100 | GA+Greedy |
|---|---|---|---|---|---|
| QWS success | 0.70 | 0.80 | 0.83 | 0.87 | **0.90** |
| WS-DREAM success | 0.83 | 0.87 | 0.77 | **0.90** | 0.80 |

At ρ = 0.90 and 0.95, all variants reach 100 %. At the limit, the differences lie within
the ±16 pp interval of 30 runs. The paired test of λ = 1 against Deb's rules on the same
30 instances is not significant (QWS: 5 against 3 better instances, p = 0.45; WS-DREAM:
6 against 3, p = 0.51). Over the 150 instances of E8 at ρ = 1, λ = 1 leads Deb's rules on
QWS (79 % against 67 %, p = 0.005) but not on WS-DREAM (84 % against 82 %, p = 0.13).
Deb's rules therefore remain the default.

### E8: The 150-instance sweep at the feasibility edge

The band ρ ∈ {0.98, 0.99, 1.0} repeated over 150 instances (`--limit-runs 150`,
`E8_limit`):

| ρ | QWS: Greedy-Local / Greedy-Repair / GA / λ=1 / GA+Greedy / Random | WS-DREAM: same order |
|---|---|---|
| 0.98 | 0.18 / 1.00 / 1.00 / 1.00 / 1.00 / 0.00 | 0.78 / 1.00 / 1.00 / 1.00 / 1.00 / 0.02 |
| 0.99 | 0.08 / 0.97 / 0.99 / 1.00 / 0.99 / 0.00 | 0.47 / 1.00 / 1.00 / 1.00 / 1.00 / 0.00 |
| **1.00** | 0.03 / **0.60** / **0.67** / **0.79** / **0.79** / 0.00 | 0.01 / **0.72** / **0.82** / **0.84** / **0.84** / 0.00 |

Paired Wilcoxon tests on normalised quality at ρ = 1 (150 paired instances):

| Comparison | QWS | WS-DREAM |
|---|---|---|
| λ=1 vs GA | 30 vs 14 better; p = 0.005 | 31 vs 15; p = 0.13 |
| GA vs Greedy-Repair | 27 vs 16; p = 0.14 | 40 vs 31; p = 0.15 |
| GA+Greedy vs Greedy-Repair | 30 vs 0; p < 10⁻⁴ | 20 vs 0; p < 10⁻⁴ |

### E9: Skewed utility weights (ρ ∈ {0.95, 1.0})

One weight raised to 0.6, the others 0.2, on the same 30 instances as E1:

| Weights | QWS: GR / GA / G+G | WS-DREAM: GR / GA / G+G |
|---|---|---|
| w_L = .6 | 0.43 / 0.70 / 0.80 | 0.70 / 0.83 / 0.77 |
| w_C = .6 | 0.63 / 0.70 / 0.80 | 0.80 / 0.83 / 0.80 |
| w_R = .6 | 0.77 / 0.70 / 0.87 | 0.60 / 0.83 / 0.83 |

Greedy-Repair's success at the limit moves 20 pp below and 14 pp above its equal-weight
value of 63 % on QWS, because repair follows the utility, not the constraints.
GA+Greedy stays within 77–87 % and the GA at 70–83 %, and all gaps remain near zero
(maximum mean gap 0.037 %), so the quality conclusions are unchanged.

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

1. **Combining constraints is the source of difficulty.** No single constraint binds
   (its critical tightness is 1 by construction), and the all-three limit (0.904 on QWS,
   0.962 on WS-DREAM) lies below every pair's, on both real datasets (E4).
2. **Greedy-Repair is the fastest near-optimal composer.** It takes under 1 ms and is
   optimal or within 0.023 % up to ρ = 0.98. It is the weakest of the three at the exact
   feasibility limit (60–63 % over 150 instances).
3. **The GA is more robust at the limit on small instances, but the paired test is not
   significant.** At ρ = 1 it reaches 67–82 % over 150 instances against 60–72 % for
   Greedy-Repair (p = 0.14 and 0.15). Mild penalty handling (λ = 1) beats Deb's rules on
   QWS (p = 0.005) but not on WS-DREAM. On whole datasets (E7) the GA's fixed budget of
   about 30,000 evaluations is too small: 0–23 % at ρ = 1, and gaps up to 0.49 %.
4. **GA+Greedy is the best default.** It has 79–90 % success at the limit on small
   instances, beats Greedy-Repair on every paired instance at ρ = 1 (p < 10⁻⁴), and is
   faster than the plain GA. On whole datasets its results come almost entirely from the
   greedy seed.
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
- **Runtimes carry some noise:** they were measured with 10 parallel worker processes,
  which contend for CPU; a single-worker re-run of the ρ ≥ 0.95 sweep gave 0.3–0.6 times
  the parallel runtimes. Relative comparisons are unaffected; absolute runtimes are
  upper bounds.
- **MILP tolerance:** "optimal" means within HiGHS's default 0.01 % relative gap, so
  reported optimality gaps are accurate to about ±0.01 %.
- **Only three QoS constraints:** with more, the MILP may slow down. This was not tested.
- **The report covers E1–E9 on both datasets** (7 pages, IEEE format).
