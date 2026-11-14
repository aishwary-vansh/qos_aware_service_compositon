# Design: QoS-Aware Service Composition under Latency, Cost and Reliability Constraints

This document covers Phase 1 (problem model) and Phase 2 (composition algorithms).
Each section names the source file that implements it.

## 1. Problem

A composite cloud application is a workflow of **N abstract tasks** that run one after
another, for example *authenticate → fetch data → process → store → notify*. For each
task, **M concrete candidate services** can do the work. They differ in QoS.

A **composition** picks one candidate for each task. The composite service must meet
three end-to-end constraints **at the same time**:

| Attribute   | Constraint            | Better when |
|-------------|-----------------------|-------------|
| Latency     | `L(a) ≤ L_max` (ms)   | lower       |
| Cost        | `C(a) ≤ C_max`        | lower       |
| Reliability | `R(a) ≥ R_min`        | higher      |

Among the feasible compositions, we want the one with the highest overall utility.

## 2. Formal model (`qos/model.py`)

### 2.1 Notation

- Tasks `i = 1..N`; candidates `j = 1..M` for each task.
- Candidate QoS vector `q_ij = (lat_ij, cost_ij, rel_ij)`.
- Composition `a = (a_1, …, a_N)` with `a_i ∈ {1..M}`. Implemented as an integer vector.
  The search space has `M^N` compositions; for N = 10 and M = 20 that is about 10^13.

### 2.2 Aggregation (sequential workflow)

```
L(a) = Σ_i lat[i, a_i]
C(a) = Σ_i cost[i, a_i]
R(a) = Π_i rel[i, a_i]   ⇔   log R(a) = Σ_i log rel[i, a_i]
```

Reliability is a product, so the code stores `log rel` and adds it like the other
attributes. This keeps every aggregate linear in the choices, which matters for
the exact MILP solver (§4.4).

### 2.3 Bounds and normalisation

For each attribute, the best and worst aggregate values are:

```
L_best = Σ_i min_j lat_ij        L_worst = Σ_i max_j lat_ij
C_best = Σ_i min_j cost_ij       C_worst = Σ_i max_j cost_ij
logR_best = Σ_i max_j log rel_ij logR_worst = Σ_i min_j log rel_ij
```

Each best bound comes from optimising one attribute alone. The data has a built-in
trade-off (§3), so no single composition reaches all three best bounds at once.

### 2.4 Utility (objective)

Utility uses Simple Additive Weighting (SAW) over normalised aggregates, with weights
`w = (w_L, w_C, w_R)` that sum to 1 (default 1/3 each):

```
U(a) = w_L (L_worst − L)/(L_worst − L_best)
     + w_C (C_worst − C)/(C_worst − C_best)
     + w_R (log R − logR_worst)/(logR_best − logR_worst)        ∈ [0, 1]
```

All three aggregates are sums over tasks, so U splits exactly into per-service terms:

```
U(a) = Σ_i u[i, a_i],   u[i, j] = w_L (max_k lat_ik − lat_ij)/ΔL
                                  + w_C (max_k cost_ik − cost_ij)/ΔC
                                  + w_R (log rel_ij − min_k log rel_ik)/ΔlogR
```

`ProblemInstance.local_utility` holds the matrix `u`. Without constraints, the best
composition takes `argmax_j u[i, j]` for each task. The constraints are what make
the problem hard.

### 2.5 Constraint violation

```
V(a) = max(0, L − L_max)/ΔL + max(0, C − C_max)/ΔC + max(0, log R_min − log R)/ΔlogR
```

`V(a) = 0` means the composition is feasible. A tolerance `FEAS_TOL = 1e-7` absorbs
float rounding. Each term is scaled by its attribute's range, so a large latency
overrun does not drown out a small reliability shortfall. All composers use `V` to
rank infeasible compositions.

### 2.6 Constraint tightness τ

Experiments need constraints that run from loose to tight in a controlled way. One
tightness value `τ ∈ [0, 1]` places each bound between the worst and best aggregate:

```
L_max     = L_best    + (1 − τ)(L_worst    − L_best)
C_max     = C_best    + (1 − τ)(C_worst    − C_best)
log R_min = logR_best − (1 − τ)(logR_best  − logR_worst)
```

- `τ = 0`: every composition is feasible. This is the unconstrained problem.
- `τ = 1`: each bound equals the best value of its attribute alone. In practice no
  composition meets all three.
- τ can also be a triple, to tighten one attribute only (an ablation in Phase 3).

`ProblemInstance.with_tightness(τ)` reuses the same candidate services with new
bounds. Experiments therefore change only the constraints, not the data.

### 2.7 Complexity

This is a **Multi-choice Multi-dimensional Knapsack Problem (MMKP)**: pick one item
from each of N classes, maximise total value, and respect three resource bounds.
MMKP is NP-hard. So is our problem, because with a single cost constraint it reduces
to the multiple-choice knapsack problem. Exact methods do not scale, which motivates
the greedy and GA heuristics.

## 3. Real QoS data (`qos/datasets.py`)

All candidate services come from two real QoS datasets. `scripts/download_data.py`
fetches them from Zenodo and checks their md5 sums. The raw files are not committed,
because the QWS licence does not allow redistribution.

| | QWS 2.0 (2,507 services) | WS-DREAM dataset 1 (339 users × 5,825 services) |
|---|---|---|
| Latency | Response Time (ms), measured | Response time seen by one user (ms), measured |
| Reliability | Reliability (%) / 100, measured | Share of users whose call returned a valid value (the dataset marks the rest `-1`) |
| Cost | AWS EC2 t3 hourly price, by throughput tier | Same |
| Cleaning | No rows dropped | Services valid for under 50 % of users dropped (248) |

**Cost.** Neither dataset records a price. Each service is ranked by its measured
throughput, which indicates provisioned capacity. The ranking is split into six equal
groups, mapped to the AWS EC2 t3 sizes nano, micro, small, medium, large and xlarge
(on-demand, Linux, us-east-1: $0.0052, $0.0104, $0.0208, $0.0416, $0.0832 and $0.1664
per hour). A composite's cost is therefore its hourly hosting cost. This gives a real
trade-off: higher-priced services have lower latency (Spearman ρ = −0.35 on QWS,
−0.54 on WS-DREAM).

**Instances** (`make_instance`):
- N × M distinct services are drawn without replacement and split at random into N
  tasks. Neither dataset labels service functions, so the split is random.
- On WS-DREAM, an instance is built for one random user, using only the services that
  user observed successfully and that user's measured latency.
- Every instance is reproducible from its seed.
- **Very large instances** (`needs_reuse`): when N × M exceeds the services available,
  each task draws its own M distinct services. A service can then appear in several
  tasks, but never twice in one task. This happens only above about 2,500 services on
  QWS and 5,500 on WS-DREAM.

## 4. Composition algorithms (`qos/composers/`)

Every composer takes a `ProblemInstance` and returns a `CompositionResult`. The
result holds the assignment, aggregated QoS, utility, violation, a feasibility flag,
runtime, the number of compositions evaluated, and an optional history.

### 4.0 Common ranking rule: Deb's feasibility rules (`base.selection_key`)

1. A feasible composition beats an infeasible one.
2. Between two feasible compositions, higher utility wins.
3. Between two infeasible compositions, lower violation wins.

The code turns this into one scalar: `key = U` if feasible, else `−1 − V`. Since
`U ∈ [0, 1]`, every feasible key ranks above every infeasible key. Random search,
brute force and GA selection all use this key. So does the "best so far" tracking.

### 4.1 Greedy-Local (`greedy.greedy_local`), baseline

```
for each task i:  a_i ← argmax_j u[i, j]
```

This is the classic *local selection* strategy (Zeng et al.). It costs O(NM) and
ignores the global constraints. It is optimal when τ = 0 and becomes infeasible once
the constraints bind.

### 4.2 Greedy-Repair (`greedy.greedy_repair`), proposed greedy composer

```
a ← Greedy-Local
REPAIR:  while V(a) > 0:
             for every single swap (i, j):  compute ΔV = V(a) − V(a[i→j]),  ΔU = U(a) − U(a[i→j])
             keep swaps with ΔV > 0
             if none: stop  → "repair_failed"
             apply the swap that maximises  ΔV / (max(ΔU, 0) + ε)
IMPROVE: while some single swap keeps a feasible and raises U:
             apply the swap with the largest utility gain (steepest ascent)
```

- The ratio rule is the classic knapsack-greedy idea: buy the most constraint
  satisfaction per unit of utility given up.
- All N·M swaps are scored in one numpy step. Each swap changes one term of each
  sum, so its new aggregates are `total − old_i + new_ij` (`_single_swap_table`).
- Each step costs O(NM). Both loops are capped at `10·N·M` steps, so the method
  takes polynomial time.
- It can stop at a local dead end and report infeasible when a feasible composition
  exists. The experiments measure how often this happens.

### 4.3 Genetic algorithm (`ga.genetic_algorithm`), proposed GA composer

| Component              | Choice                                                                  |
|------------------------|-------------------------------------------------------------------------|
| Encoding               | Integer chromosome of length N; gene i = candidate for task i. Every chromosome is a valid composition. |
| Initial population     | Uniform random, P = 100                                                 |
| Fitness / constraints  | Deb's rules (`key`, default) or static penalty `U − λ·V` with λ = 10     |
| Selection              | Tournament, k = 3                                                       |
| Crossover              | Uniform, rate p_c = 0.9                                                 |
| Mutation               | Random reset to a *different* candidate, per gene rate p_m = 1/N        |
| Elitism                | Best 2 individuals copied unchanged                                     |
| Stopping               | 300 generations, or 80 generations without improvement                  |
| Evaluation             | Whole population evaluated as one numpy batch                           |

```
P ← random population (optionally inject Greedy-Repair and Greedy-Local results)
repeat for G generations:
    E ← top-e individuals by fitness
    parents ← tournament selection
    children ← uniform crossover + random-reset mutation
    P ← E ∪ children;  evaluate P
    update best-so-far with Deb's rules; record history
    stop early on stagnation
return best feasible composition seen (else the least-violating one)
```

**GA+Greedy (hybrid).** This variant seeds the initial population with the
Greedy-Repair and Greedy-Local results. Elitism keeps the best individual, so the
hybrid's result is never worse than Greedy-Repair. The GA then searches beyond the
greedy local optimum. Phase 3 tests whether the extra time pays off.

The history (best utility, mean fitness, share of feasible individuals in each
generation) feeds the convergence plots.

### 4.4 Reference solvers (`exact.py`, `random_search.py`)

- **Optimal-MILP.** A 0-1 ILP solved with HiGHS through `scipy.optimize.milp`:
  - Variables: `x_ij ∈ {0, 1}`.
  - Objective: maximise `Σ u_ij x_ij`.
  - Choice constraint: `Σ_j x_ij = 1` for every task.
  - QoS constraints: three linear rows, with reliability in log form.

  The model is exact because everything is linear (§2.2, §2.4). It gives the true
  optimum, which defines the **optimality gap**. It also proves infeasibility, so
  success rate is measured against instances that actually have a feasible solution.

  Implementation details:
  - The one-per-task rows use a sparse matrix, so instances with tens of thousands of
    variables fit in memory.
  - The result records HiGHS's proven upper bound on utility (`upper_bound`) and its
    node count. If the time limit is hit, the bound shows how far from optimal the
    returned composition can be at most.
  - `feasibility_only=True` drops the objective, so the solver stops at the first
    feasible composition. The τ\* bisection uses this, and it is much faster.
  - HiGHS stops within its default 0.01 % relative gap. So "optimal" means within
    0.01 % of the true optimum, and on rare instances a heuristic finds a marginally
    better composition.

  **Why the MILP is fast here.** Besides the N one-per-task rows, only K = 3 QoS rows
  link the tasks. A basic optimal solution of the LP relaxation has at most N + 3
  nonzero variables, and every task needs at least one. So at most 3 tasks are split
  between candidates, and the rest are already integral. The solver therefore needs
  very little branching, even on whole datasets.
- **Brute force.** Full enumeration for M^N ≤ 2·10^6. It is used only to check the
  MILP in the unit tests.
- **Random search.** The best of 30,000 uniform samples under Deb's rules. That
  is close to the GA's worst-case budget of 100 + 300 × 98 = 29,500 evaluations,
  so the comparison uses similar effort.

## 5. Evaluation metrics (used in Phase 3)

| Metric                   | Definition                                                          |
|--------------------------|---------------------------------------------------------------------|
| Success rate             | Share of *feasible* instances (by MILP) where the composer returns a feasible composition |
| Solution quality         | Utility U(a) of the returned composition (feasible runs only)       |
| Optimality gap           | `(U* − U(a)) / U*`, with U* from the MILP                            |
| Runtime                  | Wall-clock seconds                                                  |
| Evaluations              | Number of compositions whose QoS was computed                       |
| Convergence              | GA best utility per generation                                      |

**Relative tightness.** Absolute τ is a poor experimental axis, because the feasibility
limit differs between instances and datasets. For N=10 and M=20, the limit averages
τ ≈ 0.90 on QWS and τ ≈ 0.96 on WS-DREAM. Below it the constraints barely bind; above
it nothing is solvable.

Each instance's **critical tightness τ\*** is the largest τ that still has a feasible
composition. Feasible sets are nested in τ, so the MILP finds τ\* by bisection
(`exact.critical_tightness`). The experiments then sweep **ρ = τ / τ\***. With ρ ≤ 1
every instance is solvable, and ρ = 1 is the edge of feasibility.

Sweeps (`experiments/run_experiments.py`; results in `results/ANALYSIS.md`):
1. **E1_tightness:** absolute τ from 0.0 to 0.9.
2. **E1_relative:** ρ from 0.5 to 1.0.
3. **E2_tasks:** N ∈ {5, 10, 20, 40} at ρ = 0.95.
4. **E3_candidates:** M ∈ {10, 20, 50, 100} at ρ = 0.95.
5. **E4_attribute:** each single constraint, each pair, and all three together, at
   0.9 and 1.0 × the all-three limit τ\* (a 1.2 level would exceed the valid range
   [0,1] and clamps to τ = 1, where a single constraint is feasible by construction).
5b. **E4_critical:** bisects each subset's own critical tightness τ\*_S with the
   exact solver (`critical_tightness` accepts a mask); singles always reach 1.0.
6. **E5_handling:** GA with Deb's rules against static penalties λ ∈ {1, 10, 100}.
7. **E6_convergence:** GA and GA+Greedy curves at N=20, M=30.
7b. **E8_limit:** 150 instances at ρ ∈ {0.98, 0.99, 1.0} (`--limit-runs`).
7c. **E9_weights:** skewed utility weights (one attribute at 0.6) at ρ ∈ {0.95, 1.0}.
8. **E7_full** (`experiments/run_full_dataset.py`): whole-dataset instances, using
   2,500 QWS services as 10 × 250 or 50 × 50, and 5,500 WS-DREAM services (one user) as
   10 × 550 or 50 × 110, at ρ ∈ {0.90, 0.95, 0.98, 1.0}.

Each setting runs 30 instances with fixed seeds, on each dataset
(`--dataset qws` or `--dataset wsdream`).

## 6. Validation (`tests/`)

73 unit tests run on real QWS instances, plus a hand-computed 2×2 example. They check:
- aggregation against hand computation;
- the SAW utility against its closed form, staying within [0, 1];
- the violation definition, and the τ edge cases (τ = 0 makes everything feasible;
  τ = 1 bounds equal the best aggregates; tightness is monotone);
- MILP against brute force on 16 random small instances;
- that heuristics never beat the optimum and report feasibility honestly;
- GA reproducibility, elitist monotone history and exact evaluation count;
- the hybrid GA never doing worse than its greedy seed;
- the GA staying within 5 % of the optimum on medium instances;
- every composer reporting infeasible on an impossible instance;
- that critical tightness is the exact feasibility threshold, checked against brute force;
- the dataset loaders: valid value ranges, price tiers, reproducible instances with
  distinct services, and services never repeated within one task in very large
  instances.

## 7. Assumptions and limits

- The workflow is sequential only. Parallel (AND), choice (XOR) and loop patterns
  would need different aggregation, for example max for parallel latency. The GA
  handles any aggregation unchanged. The MILP and the greedy swap table rely on
  additive aggregation.
- QoS values are static, known in advance and deterministic.
- Cost is derived from a public price list by throughput tier, not measured per service.
- WS-DREAM reliability is an observed success ratio, used as a proxy.
- Services are grouped into tasks at random, because the datasets do not label their
  functions.
- Only three QoS constraints are modelled. With more constraints, the LP relaxation can
  leave more tasks fractional, and the MILP may slow down. This has not been tested.
