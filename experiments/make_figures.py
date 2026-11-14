"""Phase 3: turn raw experiment CSVs into summary tables and figures.

Usage:
  python experiments/make_figures.py [--raw results/raw] [--out results]
Output: results/tables/*.csv, results/tables/*.md, results/figures/*.png
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- style
# Categorical slots from the validated reference palette, in fixed order.
# Each algorithm keeps one colour and marker in every figure; markers and
# line styles are the secondary encoding, so identity never relies on colour.
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
STYLE = {
    "Greedy-Repair":      dict(color="#2a78d6", marker="o", ls="-"),
    "GA":                 dict(color="#eb6834", marker="s", ls="-"),
    "GA+Greedy":          dict(color="#1baf7a", marker="^", ls="-"),
    "Random":             dict(color="#eda100", marker="D", ls="--"),
    "Greedy-Local":       dict(color="#e87ba4", marker="v", ls=":"),
    "Optimal-MILP":       dict(color=INK, marker="x", ls="-."),
    "GA (Deb)":           dict(color="#eb6834", marker="s", ls="-"),
    # Penalty weight λ is ordered, so its variants share one violet ramp, light to dark.
    "GA (penalty λ=1)":   dict(color="#b3aaf0", marker="P", ls="--"),
    "GA (penalty λ=10)":  dict(color="#6250d6", marker="X", ls="--"),
    "GA (penalty λ=100)": dict(color="#33278a", marker="*", ls="--"),
}
FEASIBLE_BAR = "#dcdbd6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.labelsize": 9, "axes.labelcolor": INK_2,
    "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5, "text.color": INK,
})


def line(ax, x, y, name, **kw):
    s = STYLE[name]
    return ax.plot(x, y, color=s["color"], marker=s["marker"], ls=s["ls"], label=name,
                   markeredgecolor=SURFACE, markeredgewidth=0.8, **kw)[0]


DATASET_LABEL = ""   # set from --dataset; stamped on every figure


def save(fig, out: pathlib.Path, name: str):
    if DATASET_LABEL:
        fig.text(0.99, 0.995, f"Dataset: {DATASET_LABEL}", ha="right", va="top", fontsize=7,
                 color=INK_MUTED)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {name}.png")


# ---------------------------------------------------------------- summary
def ci95(x: pd.Series) -> float:
    x = x.dropna()
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ok = df["feasible"] & df["opt_feasible"]
    df["gap_pct"] = np.where(ok, 100 * (df["opt_utility"] - df["utility"]) / df["opt_utility"], np.nan)
    df["gap_pct"] = df["gap_pct"].clip(lower=0)   # remove -1e-14 float noise
    # Normalised quality: U/U* on success, 0 on failure; only solvable instances.
    df["norm_quality"] = np.where(df["opt_feasible"],
                                  np.where(df["feasible"], df["utility"] / df["opt_utility"], 0.0),
                                  np.nan)
    df["runtime_ms"] = 1e3 * df["runtime_s"]
    return df


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    solvable = df[df["opt_feasible"]]
    g_all = df.groupby(keys + ["algorithm"], sort=False)
    g_sol = solvable.groupby(keys + ["algorithm"], sort=False)
    out = pd.DataFrame({
        "instances": g_all.size(),
        "solvable": g_sol.size(),
        "success_rate": g_sol["feasible"].mean(),
        "successes": g_sol["feasible"].sum(),
        "gap_pct_mean": g_sol["gap_pct"].mean(),
        "gap_pct_ci95": g_sol["gap_pct"].apply(ci95),
        "optimal_hits": g_sol["gap_pct"].apply(lambda x: float((x.dropna() < 1e-6).mean())
                                               if x.notna().any() else np.nan),
        "norm_quality": g_sol["norm_quality"].mean(),
        "runtime_ms_mean": g_all["runtime_ms"].mean(),
        "runtime_ms_ci95": g_all["runtime_ms"].apply(ci95),
        "evaluations_mean": g_all["evaluations"].mean(),
        # Sanity: a composer must never claim feasibility on an unsolvable instance.
        "false_feasible": g_all.apply(lambda x: int((x["feasible"] & ~x["opt_feasible"]).sum()),
                                      include_groups=False),
    }).reset_index()
    return out


def to_markdown(df: pd.DataFrame, path: pathlib.Path, floatfmt: dict | None = None):
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("–" if np.isnan(v) else format(v, floatfmt.get(c, ".3f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


FMT = {"success_rate": ".3f", "gap_pct_mean": ".2f", "gap_pct_ci95": ".2f",
       "optimal_hits": ".2f", "norm_quality": ".4f", "runtime_ms_mean": ".2f",
       "runtime_ms_ci95": ".2f", "evaluations_mean": ".0f", "level": ".2f"}


def write_table(summary: pd.DataFrame, tables: pathlib.Path, name: str):
    summary.to_csv(tables / f"{name}.csv", index=False)
    to_markdown(summary, tables / f"{name}.md", FMT)
    print(f"  table  {name}.csv/.md")


# ---------------------------------------------------------------- figures
HEURISTICS = ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy", "Random"]


def fig_absolute(df: pd.DataFrame, figs: pathlib.Path):
    """Share of ALL instances solved vs absolute tau, against the share that is solvable.

    Success among solvable instances is misleading here: past tau ~ 0.7 only a
    handful of instances are solvable, so the rate rests on 0-3 instances.
    """
    solved = df.groupby(["label", "algorithm"])["feasible"].mean().unstack()
    feas = df[df["algorithm"] == "Optimal-MILP"].groupby("label")["opt_feasible"].mean()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(feas.index, feas.values, width=0.04, color=FEASIBLE_BAR, zorder=0,
           label="Solvable instances (MILP)")
    for name in HEURISTICS:
        line(ax, solved.index, solved[name], name, zorder=3)
    ax.set_xlabel("Absolute constraint tightness τ")
    ax.set_ylabel("Share of instances solved")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Instances solved vs absolute tightness (N=10, M=20, 30 instances)", loc="left")
    ax.legend(loc="lower left", ncol=2)
    save(fig, figs, "fig1_solved_vs_absolute_tightness")


RHO_LABEL = "Relative tightness ρ = τ / τ*"
MIN_SUCCESSES = 5


def fig_relative(s: pd.DataFrame, figs: pathlib.Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name in HEURISTICS:
        d = s[s["algorithm"] == name]
        line(ax, d["label"], d["success_rate"], name)
    ax.set_xlabel(RHO_LABEL)
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Success rate vs relative tightness (N=10, M=20, 30 instances, all solvable)",
                 loc="left")
    ax.legend(loc="center left", ncol=1)
    save(fig, figs, "fig2_success_vs_relative_tightness")

    # Optimality gap: full view and zoom on the three strong composers.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, names, title in ((axes[0], ["Greedy-Repair", "GA", "GA+Greedy", "Random"], "(a) All composers"),
                             (axes[1], ["Greedy-Repair", "GA", "GA+Greedy"], "(b) Zoom: greedy vs GA")):
        for name in names:
            # A mean over fewer than MIN_SUCCESSES successful runs is not shown.
            d = s[(s["algorithm"] == name) & (s["successes"] >= MIN_SUCCESSES)]
            line(ax, d["label"], d["gap_pct_mean"], name)
            ax.fill_between(d["label"], (d["gap_pct_mean"] - d["gap_pct_ci95"]).clip(lower=0),
                            d["gap_pct_mean"] + d["gap_pct_ci95"], color=STYLE[name]["color"],
                            alpha=0.12, lw=0)
        ax.set_xlabel(RHO_LABEL)
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(f"Optimality gap (%), successful runs\n(points need ≥ {MIN_SUCCESSES} successes)")
    axes[0].legend(loc="center left")
    fig.suptitle("Solution quality vs relative tightness (mean ± 95% CI)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, figs, "fig3_gap_vs_relative_tightness")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name in HEURISTICS + ["Optimal-MILP"]:
        d = s[s["algorithm"] == name]
        line(ax, d["label"], d["runtime_ms_mean"], name)
    ax.set_yscale("log")
    ax.set_xlabel(RHO_LABEL)
    ax.set_ylabel("Runtime per instance (ms, log scale)")
    ax.set_title("Runtime vs relative tightness", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, figs, "fig4_runtime_vs_relative_tightness")


def fig_gap_box(df: pd.DataFrame, figs: pathlib.Path, taus=(0.8, 0.9, 0.95)):
    names = ["Greedy-Repair", "GA", "GA+Greedy", "Random"]
    short = {"Greedy-Repair": "Greedy-\nRepair", "GA": "GA", "GA+Greedy": "GA+\nGreedy",
             "Random": "Random"}
    fig, axes = plt.subplots(1, len(taus), figsize=(7.6, 3.4), sharey=True)
    for ax, tau in zip(axes, taus):
        data = [df[(df["label"] == tau) & (df["algorithm"] == n)]["gap_pct"].dropna() for n in names]
        # Boxes over fewer than MIN_SUCCESSES successful runs are left empty.
        shown = [d if len(d) >= MIN_SUCCESSES else [] for d in data]
        bp = ax.boxplot(shown, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, lw=1.2),
                        flierprops=dict(marker="o", ms=3, mfc=INK_MUTED, mec="none"))
        for patch, n in zip(bp["boxes"], names):
            patch.set(facecolor=STYLE[n]["color"], alpha=0.75, edgecolor=SURFACE, lw=1.5)
        ax.set_xticks(range(1, len(names) + 1),
                      [f"{short[n]}\nn={len(d)}" for n, d in zip(names, data)])
        ax.set_title(f"ρ = {tau}", loc="left")
        ax.set_yscale("symlog", linthresh=0.1)
    axes[0].set_ylabel("Optimality gap (%), symlog scale")
    fig.suptitle(f"Distribution of optimality gap (n = successful runs of 30; "
                 f"no box when n < {MIN_SUCCESSES})", x=0.01, ha="left", fontweight="bold",
                 fontsize=10)
    fig.tight_layout()
    save(fig, figs, "fig5_gap_distribution")


def fig_scalability(s_n: pd.DataFrame, s_m: pd.DataFrame, figs: pathlib.Path):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
    for col, (s, xlabel) in enumerate(((s_n, "Number of tasks N (M=20)"),
                                       (s_m, "Candidates per task M (N=10)"))):
        for name in HEURISTICS + ["Optimal-MILP"]:
            d = s[s["algorithm"] == name]
            line(axes[0, col], d["label"], d["runtime_ms_mean"], name)
            if name in ("Greedy-Repair", "GA", "GA+Greedy", "Random"):
                d = d[d["successes"] >= MIN_SUCCESSES]
                line(axes[1, col], d["label"], d["gap_pct_mean"], name)
        for row in (0, 1):
            axes[row, col].set_xscale("log", base=2)
            axes[row, col].set_xticks(sorted(s["label"].unique()),
                                      [str(v) for v in sorted(s["label"].unique())])
            axes[row, col].minorticks_off()
        axes[0, col].set_yscale("log")
        # Gaps are often exactly 0 on real data; symlog shows 0 where log cannot.
        axes[1, col].set_yscale("symlog", linthresh=1e-3)
        axes[1, col].set_ylim(bottom=0)
        axes[1, col].set_xlabel(xlabel)
    axes[0, 0].set_ylabel("Runtime (ms, log scale)")
    axes[1, 0].set_ylabel("Optimality gap (%), symlog scale\n(successful runs)")
    axes[0, 0].set_title("(a) Runtime vs N", loc="left")
    axes[0, 1].set_title("(b) Runtime vs M", loc="left")
    axes[1, 0].set_title("(c) Optimality gap vs N", loc="left")
    axes[1, 1].set_title("(d) Optimality gap vs M", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Scalability at ρ = 0.95 (30 instances per point)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, figs, "fig6_scalability")


ATTRIBUTE_MODES = ["latency only", "cost only", "reliability only", "latency + cost",
                   "latency + reliability", "cost + reliability", "all three"]
ATTRIBUTE_TICKS = ["Lat.", "Cost", "Rel.", "Lat.+\nCost", "Lat.+\nRel.", "Cost+\nRel.", "All 3"]


def fig_attribute(s: pd.DataFrame, df: pd.DataFrame, figs: pathlib.Path):
    """Single constraints, pairs and all three, at levels relative to the all-three tau_star."""
    names = ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy"]
    levels = sorted(s["level"].unique())
    fig, axes = plt.subplots(len(levels), 1, figsize=(7.6, 2.2 * len(levels) + 0.6), sharey=True)
    width = 0.18
    x = np.arange(len(ATTRIBUTE_MODES))
    for ax, level in zip(np.atleast_1d(axes), levels):
        d = s[s["level"] == level]
        solv = df[(df["level"] == level) & (df["algorithm"] == "Optimal-MILP")] \
            .groupby("label")["opt_feasible"].mean().reindex(ATTRIBUTE_MODES)
        ax.bar(x, solv.values, width=0.9, color=FEASIBLE_BAR, zorder=0,
               label="Solvable instances (MILP)")
        for k, name in enumerate(names):
            vals = d[d["algorithm"] == name].set_index("label")["success_rate"] \
                .reindex(ATTRIBUTE_MODES)
            ax.bar(x + (k - 1.5) * width, vals.values, width=width - 0.03,
                   color=STYLE[name]["color"], label=name, zorder=2)
        ax.set_xticks(x, ATTRIBUTE_TICKS)
        for xi, frac in zip(x, solv.values):
            if frac == 0:
                ax.text(xi, 0.5, "none\nsolvable", ha="center", va="center", fontsize=7,
                        color=INK_2)
        ax.set_title(f"Tightened attributes at τ = {level} × τ* (τ* = all-three limit)",
                     loc="left")
        ax.set_ylabel("Success rate")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Single constraints, pairs and all three together", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, figs, "fig7_single_vs_multi_constraint")


def fig_handling(s: pd.DataFrame, figs: pathlib.Path):
    names = list(dict.fromkeys(s["algorithm"]))
    names = [n for n in names if n != "Optimal-MILP"]
    taus = sorted(s["label"].unique())
    x = np.arange(len(taus))
    width = 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for k, name in enumerate(names):
        d = s[s["algorithm"] == name].set_index("label").reindex(taus)
        off = (k - (len(names) - 1) / 2) * width
        axes[0].bar(x + off, d["success_rate"], width=width - 0.02, color=STYLE[name]["color"],
                    label=name)
        axes[1].bar(x + off, d["gap_pct_mean"], width=width - 0.02, color=STYLE[name]["color"],
                    yerr=[np.minimum(d["gap_pct_ci95"], d["gap_pct_mean"]), d["gap_pct_ci95"]],
                    error_kw=dict(ecolor=INK_2, lw=0.8, capsize=2))   # gap >= 0: clip CI at 0
    for ax in axes:
        ax.set_xticks(x, [f"ρ = {t}" for t in taus])
    axes[0].set_ylabel("Success rate (solvable instances)")
    axes[1].set_ylabel("Optimality gap (%), mean ± 95% CI")
    axes[0].set_title("(a) Success rate", loc="left")
    axes[1].set_title("(b) Solution quality", loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("GA constraint handling: Deb's rules vs static penalty", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, figs, "fig8_constraint_handling")


def fig_convergence(npz_path: pathlib.Path, figs: pathlib.Path) -> pd.DataFrame:
    z = np.load(npz_path)
    opt = z["opt_utility"]
    solvable = ~np.isnan(opt)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    rows = []
    for name in ("GA", "GA+Greedy"):
        best = z[f"{name}_best_utility"][solvable]
        norm = np.nan_to_num(best / opt[solvable, None], nan=0.0)   # no feasible yet -> 0
        gens = np.arange(norm.shape[1])
        mean, ci = norm.mean(0), 1.96 * norm.std(0, ddof=1) / np.sqrt(len(norm))
        line(axes[0], gens, mean, name, markevery=30)
        axes[0].fill_between(gens, mean - ci, np.minimum(mean + ci, 1), color=STYLE[name]["color"],
                             alpha=0.15, lw=0)
        feas = z[f"{name}_feasible_ratio"][solvable].mean(0)
        line(axes[1], gens, feas, name, markevery=30)
        for g in (0, 10, 25, 50, 100, 200, 300):
            rows.append({"algorithm": name, "generation": g, "mean_U_over_Ustar": mean[g],
                         "share_feasible_runs": float((norm[:, g] > 0).mean()),
                         "population_feasible_ratio": feas[g]})
    greedy = np.nanmean(np.nan_to_num(z["greedy_utility"][solvable] / opt[solvable], nan=0.0))
    axes[0].axhline(greedy, color=STYLE["Greedy-Repair"]["color"], ls="--", lw=1.2,
                    label="Greedy-Repair (final)")
    axes[0].axhline(1.0, color=INK_MUTED, ls=":", lw=1.0, label="Optimum (MILP)")
    axes[0].set_ylim(max(0.0, min(0.8, greedy - 0.1)), 1.005)
    axes[0].set_xlabel("Generation")
    axes[1].set_xlabel("Generation")
    axes[0].set_ylabel("Best-so-far U / U*  (0 until feasible)")
    axes[1].set_ylabel("Feasible share of population")
    axes[0].set_title("(a) Convergence of best solution", loc="left")
    axes[1].set_title("(b) Population feasibility", loc="left")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"GA convergence, N=20, M=30, ρ=0.95 ({int(solvable.sum())} solvable instances)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, figs, "fig9_convergence")
    return pd.DataFrame(rows)


def significance(df: pd.DataFrame, taus) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests on normalised quality (E1)."""
    pairs = [("GA", "Greedy-Repair"), ("GA+Greedy", "Greedy-Repair"), ("GA+Greedy", "GA")]
    rows = []
    for tau in taus:
        d = df[(df["label"] == tau) & df["opt_feasible"]]
        piv = d.pivot_table(index="seed", columns="algorithm", values="norm_quality")
        for a, b in pairs:
            if a not in piv or b not in piv:
                continue
            diff = (piv[a] - piv[b]).dropna()
            nonzero = diff[np.abs(diff) > 1e-12]
            p = wilcoxon(nonzero).pvalue if len(nonzero) >= 5 else np.nan
            rows.append({"rho": tau, "A": a, "B": b, "n": len(diff),
                         "mean_diff": diff.mean(), "A_better": int((nonzero > 0).sum()),
                         "B_better": int((nonzero < 0).sum()), "ties": int(len(diff) - len(nonzero)),
                         "p_value": p})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Phase 3 tables and figures.")
    parser.add_argument("--dataset", choices=("qws", "wsdream"), default="qws")
    parser.add_argument("--raw", help="raw results folder (default results/<dataset>/raw)")
    parser.add_argument("--out", help="output folder (default results/<dataset>)")
    args = parser.parse_args()
    global DATASET_LABEL
    DATASET_LABEL = {"qws": "QWS 2.0", "wsdream": "WS-DREAM"}[args.dataset]
    out = pathlib.Path(args.out or ROOT / "results" / args.dataset)
    raw = pathlib.Path(args.raw or out / "raw")
    figs, tables = out / "figures", out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    load = lambda name: add_derived(pd.read_csv(raw / f"{name}.csv"))  # noqa: E731

    e1 = load("E1_tightness")
    s1 = summarise(e1, ["label"])
    write_table(s1, tables, "E1_tightness_summary")
    fig_absolute(e1, figs)

    e1r = load("E1_relative")
    s1r = summarise(e1r, ["label"])
    write_table(s1r, tables, "E1_relative_summary")
    fig_relative(s1r, figs)
    fig_gap_box(e1r, figs)
    sig = significance(e1r, [0.8, 0.85, 0.9, 0.95, 0.98, 1.0])
    sig.to_csv(tables / "E1_wilcoxon.csv", index=False)
    to_markdown(sig, tables / "E1_wilcoxon.md", {"rho": ".2f", "mean_diff": ".5f", "p_value": ".2e"})
    print("  table  E1_wilcoxon.csv/.md")
    tau_star = e1r.drop_duplicates("seed")["tau_star"]
    print(f"  tau_star over E1 instances: mean {tau_star.mean():.3f}, "
          f"min {tau_star.min():.3f}, max {tau_star.max():.3f}")

    e2, e3 = load("E2_tasks"), load("E3_candidates")
    for d in (e2, e3):
        d["label"] = d["label"].astype(int)
    s2, s3 = summarise(e2, ["label"]), summarise(e3, ["label"])
    write_table(s2, tables, "E2_tasks_summary")
    write_table(s3, tables, "E3_candidates_summary")
    fig_scalability(s2, s3, figs)

    e4 = load("E4_attribute")
    s4 = summarise(e4, ["level", "label"])
    write_table(s4, tables, "E4_attribute_summary")
    fig_attribute(s4, e4, figs)

    e5 = load("E5_handling")
    s5 = summarise(e5, ["label"])
    write_table(s5, tables, "E5_handling_summary")
    fig_handling(s5, figs)

    conv = fig_convergence(raw / "E6_convergence.npz", figs)
    conv.to_csv(tables / "E6_convergence_summary.csv", index=False)
    to_markdown(conv, tables / "E6_convergence_summary.md",
                {"mean_U_over_Ustar": ".4f", "share_feasible_runs": ".2f",
                 "population_feasible_ratio": ".3f"})
    print("  table  E6_convergence_summary.csv/.md")

    bad = sum(int(s["false_feasible"].sum()) for s in (s1, s1r, s2, s3, s4, s5))
    print(f"sanity: {bad} runs claimed feasibility on an unsolvable instance (must be 0)")


if __name__ == "__main__":
    main()
