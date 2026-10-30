# Phase 3 Results and Analysis

All numbers come from `results/tables/*.csv`. Those tables are built from the raw per-run
files in `results/raw/` by `experiments/make_figures.py`. Every setting uses 30
random instances with fixed seeds, and every instance is also solved exactly with
the MILP.

To reproduce everything (about 2 minutes on 10 worker processes):

```bash
python experiments/run_experiments.py --runs 30
python experiments/make_figures.py
```

## 1. Experimental setup

| Item | Value |
|---|---|
| Workflow | Sequential, N tasks, M candidates per task |
| Default size | N = 10, M = 20 (search space 20^10 ≈ 10^13) |
| QoS data | Synthetic, QWS-like ranges, built-in cost-vs-quality trade-off (`qos/generator.py`) |
| Utility | SAW over normalised aggregates, equal weights |
| Composers | Greedy-Local, Greedy-Repair, GA, GA+Greedy (hybrid), Random (30,000 samples) |
| Reference | Optimal-MILP (HiGHS via `scipy.optimize.milp`) |
| GA settings | P = 100, ≤ 300 generations, tournament k = 3, uniform crossover 0.9, mutation 1/N, elitism 2, stop after 80 stagnant generations, Deb's rules |
| Runs | 30 instances per setting; 1,320 instances and 7,440 result rows in total, MILP included |
| Machine | Python 3.13, NumPy 2.4, SciPy 1.17; 10 parallel worker processes, single-threaded each |

### Metrics

- **Success rate.** The share of solvable instances where the composer returns a
  feasible composition. "Solvable" means the MILP found a feasible composition.
- **Optimality gap.** `(U* − U) / U*` in %, computed over successful runs only.
- **Optimal hits.** The share of successful runs that reach the exact optimum.
- **Runtime.** Wall-clock milliseconds per instance.

### Two ways to set constraint tightness

**Absolute tightness τ** interpolates each bound between its worst and best aggregate
value (`docs/design.md` §2.6). Every instance has a **critical tightness τ\***: the
largest τ that still has a feasible composition. The MILP finds τ\* by bisection to
within 10^-3.

For N=10 and M=20, τ\* averaged 0.680, ranging from 0.661 to 0.703. Above τ ≈ 0.70
almost no instance is solvable, so the hard region in absolute τ is a narrow band.

The main experiments therefore use **relative tightness ρ = τ / τ\***. With ρ ≤ 1,
every instance is solvable, and ρ → 1 is the edge of feasibility, where the feasible
set is smallest.

## 2. Results

### E1a: Absolute tightness (Fig. 1)

| τ | Solvable | Greedy-Local solved | Greedy-Repair / GA / GA+Greedy solved | Random solved |
|---|---|---|---|---|
| 0.0–0.2 | 30/30 | 30 | 30 / 30 / 30 | 30 |
| 0.4 | 30/30 | 21 | 30 / 30 / 30 | 30 |
| 0.5 | 30/30 | 1 | 30 / 30 / 30 | 30 |
| 0.6 | 30/30 | 0 | 30 / 30 / 30 | 30 |
| 0.65 | 30/30 | 0 | 30 / 30 / 30 | 1 |
| 0.7 | 3/30 | 0 | 2 / 0 / 2 | 0 |
| ≥ 0.75 | 0/30 | 0 | 0 | 0 |

When constraints are loose, the unconstrained choice (Greedy-Local) is optimal.
Its success collapses between τ = 0.4 and 0.5, well before the instances become hard.
No composer ever claimed feasibility on an unsolvable instance (sanity check = 0).

### E1b: Relative tightness (Figs. 2–5): main result

Success rate (30 solvable instances per row):

| ρ | Greedy-Local | Greedy-Repair | GA | GA+Greedy | Random |
|---|---|---|---|---|---|
| 0.50 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.70 | 0.033 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.80–0.90 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.95 | 0.000 | 1.000 | 1.000 | 1.000 | 0.033 |
| 0.98 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **1.00** | 0.000 | **0.167** | **0.367** | **0.433** | 0.000 |

Optimality gap (%, mean over successful runs) and optimal-hit rate:

| ρ | Greedy-Repair | GA | GA+Greedy | Random |
|---|---|---|---|---|
| 0.80 | 0.053 (67 %) | 0.088 (57 %) | 0.048 (70 %) | 8.07 |
| 0.85 | 0.091 (63 %) | 0.194 (33 %) | 0.074 (80 %) | 8.07 |
| 0.90 | 0.083 (70 %) | 0.321 (30 %) | 0.062 (77 %) | 8.91 |
| 0.95 | 0.146 (63 %) | 0.336 (40 %) | 0.100 (67 %) | – (1 success) |
| 0.98 | 0.213 (57 %) | 0.348 (37 %) | 0.156 (67 %) | – |

Mean runtime per instance at N=10, M=20:

| Composer | Runtime |
|---|---|
| Greedy-Local | 0.08 ms |
| Greedy-Repair | 0.3–1.1 ms |
| Random | ≈ 14 ms |
| GA+Greedy | 26–35 ms |
| GA | 38–49 ms |
| MILP | 14–74 ms (the slowest point is ρ = 1) |

Paired Wilcoxon signed-rank tests on normalised quality (`E1_wilcoxon.md`). Ties are
dropped, and p-values are not corrected for multiple comparisons:

- **GA is worse than Greedy-Repair** at ρ = 0.85 (p = 0.036) and ρ = 0.90 (p = 0.0046).
  The other levels show no significant difference.
- **GA+Greedy is better than GA** at ρ = 0.85, 0.90, 0.95 and 0.98
  (p = 0.0062, 3.8·10^-4, 0.046, 0.035).
- **GA+Greedy against Greedy-Repair.** The hybrid is never worse, by construction.
  It is strictly better on 1–5 instances at each tested ρ from 0.80 to 0.98. At ρ = 1 it solves 8
  instances that Greedy-Repair cannot, and is beaten on none (p = 0.0078).
- **GA against Greedy-Repair at ρ = 1.** GA wins 9 instances and loses 3. That
  difference is not significant (p = 0.17).

### E2/E3: Scalability at ρ = 0.95 (Fig. 6)

Greedy-Repair, GA and GA+Greedy keep a 100 % success rate at every N and M tested.
Greedy-Local never exceeds 3 %, and Random fails for N ≥ 10. The quality trends diverge:

| | N = 5 | N = 10 | N = 20 | N = 40 |
|---|---|---|---|---|
| GA gap % (optimal hits) | 0.12 (83 %) | 0.28 (50 %) | 0.31 (7 %) | 0.50 (0 %) |
| Greedy-Repair gap % | 0.21 | 0.085 | 0.061 | 0.022 |
| GA+Greedy gap % | 0.004 | 0.067 | 0.043 | 0.022 |
| GA runtime (ms) | 34 | 44 | 74 | 131 |
| Greedy-Repair runtime (ms) | 0.6 | 0.8 | 1.8 | 2.9 |
| MILP runtime (ms) | 16 | 22 | 82 | 92 |

| | M = 10 | M = 20 | M = 50 | M = 100 |
|---|---|---|---|---|
| GA gap % | 0.15 | 0.18 | 0.46 | 0.55 |
| Greedy-Repair gap % | 0.20 | 0.088 | 0.051 | 0.047 |
| GA+Greedy gap % | 0.12 | 0.081 | 0.042 | 0.047 |

The GA has a fixed budget, so its gap grows with the search space (M^N). Greedy-Repair
does better as instances grow. With more tasks and more candidates, a single swap
changes the totals only slightly, so fine-grained repair rarely overshoots.

### E4: One constraint against three together (Fig. 7)

| Absolute τ | Latency only | Cost only | Reliability only | All three |
|---|---|---|---|---|
| Solvable at 0.8 | 30/30 | 30/30 | 30/30 | **0/30** |
| Greedy-Local solved at 0.65 | 30/30 | 0/30 | 30/30 | 0/30 |
| Greedy-Repair solved at 0.65 | 30/30 | 30/30 | 30/30 | 28/30 |

Each constraint alone is easy: it can be met even at τ = 0.8. The three together
cannot be, because they pull in opposite directions. Low cost needs cheap services,
which are slower and less reliable. This is the multi-constraint gap named in the
problem statement.

Cost is the constraint that the utility-greedy choice breaks first, because a
balanced utility still favours premium services. The only setting where
Greedy-Repair failed to find an existing solution was the three-constraint case at
τ = 0.65.

### E5: GA constraint handling (Fig. 8)

| ρ | Deb's rules | Penalty λ=1 | λ=10 | λ=100 | GA+Greedy |
|---|---|---|---|---|---|
| Success at 1.0 | 0.367 | 0.333 | 0.333 | 0.300 | 0.400 |
| Gap % at 0.90 | 0.12 | 0.07 | 0.16 | 0.20 | 0.06 |
| Gap % at 0.95 | 0.24 | 0.16 | 0.29 | 0.19 | 0.05 |

All 95 % confidence intervals overlap. No constraint-handling rule wins clearly.
Deb's rules are marginally the most reliable at ρ = 1, and they need no λ tuning.
How the population is seeded matters much more than which rule ranks infeasible
individuals.

### E6: Convergence at N=20, M=30, ρ=0.95 (Fig. 9)

| Generation | 10 | 25 | 50 | 100 | 300 |
|---|---|---|---|---|---|
| GA: runs with a feasible solution | 0 % | 90 % | 100 % | 100 % | 100 % |
| GA: mean U/U\* (0 while infeasible) | 0.000 | 0.864 | 0.979 | 0.991 | 0.996 |
| GA+Greedy: mean U/U\* | 0.9997 | 0.9997 | 0.9997 | 0.9997 | 0.9997 |

A random population starts with no feasible member. Deb's rules first push it
towards feasibility, which takes about 25 generations. Only then does it optimise
utility. After that, about half of the population stays feasible.

The hybrid starts where the plain GA ends after 300 generations. That is why its
stagnation stop fires earlier, and why it is about 35 % faster than the plain GA.

## 3. Discussion

1. **The difficulty comes from combining constraints, not from any single one.**
   Every single-attribute problem was solvable at τ = 0.8, and the combined problem
   never was (E4). Local selection, the strategy most simple implementations use,
   fails as soon as the constraints start to bind (E1).
2. **Greedy-Repair is the best quality-per-millisecond composer.** It stays within
   0.25 % of the optimum on average up to ρ = 0.98, at 1–3 ms, roughly 40× faster
   than the GA. Its weakness is the very edge of feasibility: at ρ = 1 it finds a
   solution in only 17 % of instances, because single swaps get stuck in local dead
   ends.
3. **The GA is more robust at the edge but less precise.** At ρ = 1 it succeeds about
   twice as often as Greedy-Repair. With a fixed budget, though, its optimality gap
   grows with N and M, and it is significantly worse than Greedy-Repair at
   ρ = 0.85–0.90.
4. **The hybrid GA+Greedy combines both strengths.** It has the highest success rate
   at ρ = 1 (43 %), and the lowest or equal-lowest gap for 0.8 ≤ ρ ≤ 0.98 and at every
   N and M at ρ = 0.95. It is also faster than the plain GA. At loose settings
   (ρ ≤ 0.7) the plain GA has a marginally lower gap, within the confidence intervals.
5. **Random search is not competitive.** Its gap is 7–9 %, and it fails from ρ = 0.95,
   or from N ≥ 10. This shows that the search space is far too large to sample blindly.
6. **The exact MILP is fast at these sizes** (≤ 0.1 s). It exists here as the quality
   reference. The heuristics still matter where an exact linear model is not
   available or not practical:
   - non-linear aggregation, such as parallel branches (max latency) and loops;
   - dynamic or online re-composition, which needs millisecond answers;
   - much larger instances.

   Greedy-Repair keeps a millisecond budget throughout.

## 4. Threats to validity

- **Synthetic QoS data.** The data is synthetic, with a deliberate cost/quality
  trade-off. Real QWS records may correlate differently.
- **Workflow shape.** Only sequential workflows are tested. With parallel and choice
  patterns, aggregation is no longer additive, which affects both the MILP and the
  swap table.
- **Runtime measurement.** Runtimes were measured with 10 worker processes running in
  parallel, so absolute values carry some noise. The relative ordering is stable
  across settings.
- **Fixed GA budget.** The GA's budget is fixed (P = 100, ≤ 300 generations). A larger
  budget would close part of its quality gap, at a proportional cost in time.

## 5. Files

| File | Content |
|---|---|
| `figures/fig1_solved_vs_absolute_tightness.png` | E1a |
| `figures/fig2_success_vs_relative_tightness.png` | E1b success rate |
| `figures/fig3_gap_vs_relative_tightness.png` | E1b optimality gap |
| `figures/fig4_runtime_vs_relative_tightness.png` | E1b runtime |
| `figures/fig5_gap_distribution.png` | E1b gap distribution, ρ = 0.8 / 0.9 / 0.95 |
| `figures/fig6_scalability.png` | E2 + E3 |
| `figures/fig7_single_vs_multi_constraint.png` | E4 |
| `figures/fig8_constraint_handling.png` | E5 |
| `figures/fig9_convergence.png` | E6 |
| `tables/*_summary.csv / .md` | Per-setting summaries |
| `tables/E1_wilcoxon.*` | Significance tests |
| `raw/*.csv`, `raw/E6_convergence.npz` | Per-run results |
