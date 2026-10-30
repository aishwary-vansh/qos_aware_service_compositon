"""Build the column-width figures and LaTeX tables used by report/report.tex.

Reads results/raw and results/tables, so run make_figures.py first.

Usage: python experiments/make_report_figures.py
Output: report/figures/*.pdf, report/tables/*.tex
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import FEASIBLE_BAR, INK_MUTED, SURFACE, STYLE, line

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"

# IEEE two-column widths in inches: one column, then the full text width.
COLUMN, FULL = 3.5, 7.16

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "lines.linewidth": 1.1,
    "lines.markersize": 3.5, "axes.linewidth": 0.6, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,   # TrueType, so fonts stay embedded and selectable
})
MAIN = ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy", "Random"]
RHO = r"Relative tightness $\rho = \tau / \tau^*$"
MIN_SUCCESSES = 5


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {name}.pdf")


def fig_absolute() -> None:
    df = pd.read_csv(RESULTS / "raw" / "E1_tightness.csv")
    solved = df.groupby(["label", "algorithm"])["feasible"].mean().unstack()
    solvable = df[df["algorithm"] == "Optimal-MILP"].groupby("label")["opt_feasible"].mean()
    fig, ax = plt.subplots(figsize=(COLUMN, 1.95))
    ax.bar(solvable.index, solvable.values, width=0.04, color=FEASIBLE_BAR, zorder=0,
           label="Solvable (MILP)")
    for name in MAIN:
        line(ax, solved.index, solved[name], name, zorder=3)
    ax.set_xlabel(r"Absolute tightness $\tau$")
    ax.set_ylabel("Share of instances solved")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left", ncol=2)
    save(fig, "absolute_tightness")


def fig_success(s: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN, 1.95))
    for name in MAIN:
        d = s[s["algorithm"] == name]
        line(ax, d["label"], d["success_rate"], name)
    ax.set_xlabel(RHO)
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center left")
    save(fig, "success_rho")


def fig_gap(s: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN, 1.95))
    for name in ["Greedy-Repair", "GA", "GA+Greedy"]:
        d = s[(s["algorithm"] == name) & (s["successes"] >= MIN_SUCCESSES)]
        line(ax, d["label"], d["gap_pct_mean"], name)
        ax.fill_between(d["label"], (d["gap_pct_mean"] - d["gap_pct_ci95"]).clip(lower=0),
                        d["gap_pct_mean"] + d["gap_pct_ci95"], color=STYLE[name]["color"],
                        alpha=0.12, lw=0)
    ax.set_xlabel(RHO)
    ax.set_ylabel("Optimality gap (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    save(fig, "gap_rho")


def fig_scalability(s_n: pd.DataFrame, s_m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.75))
    panels = ((0, s_n, "runtime_ms_mean", "(a) Runtime vs $N$", "Tasks $N$"),
              (1, s_n, "gap_pct_mean", "(b) Gap vs $N$", "Tasks $N$"),
              (2, s_m, "runtime_ms_mean", "(c) Runtime vs $M$", "Candidates $M$"),
              (3, s_m, "gap_pct_mean", "(d) Gap vs $M$", "Candidates $M$"))
    for i, s, column, title, xlabel in panels:
        ax = axes[i]
        names = ["Greedy-Repair", "GA", "GA+Greedy", "Optimal-MILP"]
        if column == "gap_pct_mean":
            names = names[:3]
        for name in names:
            d = s[(s["algorithm"] == name) & (s["successes"] >= MIN_SUCCESSES)]
            line(ax, d["label"], d[column], name)
        ticks = sorted(s["label"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Runtime (ms)")
    axes[1].set_ylabel("Optimality gap (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(pad=0.3)
    save(fig, "scalability")


def fig_constraints(s: pd.DataFrame) -> None:
    modes = ["latency only", "cost only", "reliability only", "all three"]
    names = ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy"]
    levels = (0.65, 0.8)
    x = np.arange(len(modes))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.8), sharey=True)
    for ax, level in zip(axes, levels):
        d = s[s["level"] == level]
        milp = d[d["algorithm"] == "Optimal-MILP"].set_index("label").reindex(modes)
        share = milp["solvable"].fillna(0).to_numpy() / milp["instances"].to_numpy()
        ax.bar(x, share, width=0.92, color=FEASIBLE_BAR, zorder=0, label="Solvable (MILP)")
        for k, name in enumerate(names):
            vals = d[d["algorithm"] == name].set_index("label")["success_rate"].reindex(modes)
            ax.bar(x + (k - 1.5) * width, vals.fillna(0).to_numpy(), width=width - 0.03,
                   color=STYLE[name]["color"], label=name, zorder=2)
        for xi, frac in zip(x, share):
            if frac == 0:
                ax.text(xi, 0.04, "none\nsolvable", ha="center", va="bottom", fontsize=5.5,
                        color=INK_MUTED)
        ax.set_xticks(x, ["Lat.", "Cost", "Rel.", "All 3"])
        ax.set_title(rf"$\tau$ = {level}", loc="left")
    axes[0].set_ylabel("Success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(pad=0.3)
    save(fig, "single_vs_multi")


def fig_convergence() -> None:
    z = np.load(RESULTS / "raw" / "E6_convergence.npz")
    opt = z["opt_utility"]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.75))
    for name in ("GA", "GA+Greedy"):
        norm = np.nan_to_num(z[f"{name}_best_utility"] / opt[:, None], nan=0.0)
        generations = np.arange(norm.shape[1])
        line(axes[0], generations, norm.mean(0), name, markevery=50)
        line(axes[1], generations, z[f"{name}_feasible_ratio"].mean(0), name, markevery=50)
    axes[0].set_ylim(0.8, 1.01)
    axes[0].set_ylabel(r"Best $U/U^*$ (0 until feasible)")
    axes[1].set_ylabel("Feasible share of population")
    axes[0].set_title("(a) Best solution", loc="left")
    axes[1].set_title("(b) Population", loc="left")
    for ax in axes:
        ax.set_xlabel("Generation")
    axes[0].legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, "convergence")


def table_success(s: pd.DataFrame) -> None:
    piv = s.pivot(index="label", columns="algorithm", values="success_rate")[MAIN]
    rows = [" & ".join([f"{rho:.2f}"] + [f"{v:.3f}" for v in row]) + r" \\"
            for rho, row in zip(piv.index, piv.to_numpy())]
    header = r"$\rho$ & Greedy-Local & Greedy-Repair & GA & GA+Greedy & Random \\"
    write_table("success_rho", "cccccc", header, rows)


def table_gap(s: pd.DataFrame) -> None:
    names = ["Greedy-Repair", "GA", "GA+Greedy", "Random"]
    rows = []
    for rho in sorted(s["label"].unique()):
        cells = [f"{rho:.2f}"]
        for name in names:
            r = s[(s["label"] == rho) & (s["algorithm"] == name)].iloc[0]
            if r["successes"] < MIN_SUCCESSES:
                cells.append("--")
            else:
                cells.append(f"{r['gap_pct_mean']:.3f} ({100 * r['optimal_hits']:.0f})")
        rows.append(" & ".join(cells) + r" \\")
    header = r"$\rho$ & Greedy-Repair & GA & GA+Greedy & Random \\"
    write_table("gap_rho", "ccccc", header, rows)


def table_runtime(s: pd.DataFrame) -> None:
    names = MAIN + ["Optimal-MILP"]
    rows = []
    for name in names:
        d = s[s["algorithm"] == name]["runtime_ms_mean"]
        rows.append(f"{name} & {d.min():.2f}--{d.max():.2f} \\\\")
    write_table("runtime", "lc", r"Composer & Runtime (ms) \\", rows)


def table_scale(s_n: pd.DataFrame, s_m: pd.DataFrame) -> None:
    names = ["Greedy-Repair", "GA", "GA+Greedy"]
    rows = []
    for symbol, s in (("N", s_n), ("M", s_m)):
        for size in sorted(s["label"].unique()):
            d = s[s["label"] == size].set_index("algorithm")
            cells = [f"${symbol}={int(size)}$"]
            cells += [f"{d.loc[n, 'gap_pct_mean']:.3f}" for n in names]
            cells += [f"{d.loc[n, 'runtime_ms_mean']:.1f}" for n in names + ["Optimal-MILP"]]
            rows.append(" & ".join(cells) + r" \\")
    header = (r"Size & \multicolumn{3}{c}{Gap (\%)} & \multicolumn{4}{c}{Runtime (ms)} \\"
              "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}"
              "\n" r" & GR & GA & G+G & GR & GA & G+G & MILP \\")
    write_table("scale", "lccccccc", header, rows)


def table_handling(s: pd.DataFrame) -> None:
    names = ["GA (Deb)", "GA (penalty λ=1)", "GA (penalty λ=10)", "GA (penalty λ=100)", "GA+Greedy"]
    rows = []
    for rho in sorted(s["label"].unique()):
        cells = [f"{rho:.2f}"]
        for name in names:
            r = s[(s["label"] == rho) & (s["algorithm"] == name)].iloc[0]
            cells.append(f"{r['success_rate']:.2f} / {r['gap_pct_mean']:.2f}")
        rows.append(" & ".join(cells) + r" \\")
    header = r"$\rho$ & Deb & $\lambda=1$ & $\lambda=10$ & $\lambda=100$ & GA+Greedy \\"
    write_table("handling", "cccccc", header, rows)


def write_table(name: str, columns: str, header: str, rows: list[str]) -> None:
    body = "\n".join([rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header, r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}"])
    (TABLES / f"{name}.tex").write_text(body + "\n", encoding="utf-8")
    print(f"  {name}.tex")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = RESULTS / "tables"
    relative = pd.read_csv(tables / "E1_relative_summary.csv")
    fig_absolute()
    fig_success(relative)
    fig_gap(relative)
    fig_scalability(pd.read_csv(tables / "E2_tasks_summary.csv"),
                    pd.read_csv(tables / "E3_candidates_summary.csv"))
    fig_constraints(pd.read_csv(tables / "E4_attribute_summary.csv"))
    fig_convergence()
    table_success(relative)
    table_gap(relative)
    table_runtime(relative)
    table_scale(pd.read_csv(tables / "E2_tasks_summary.csv"),
                pd.read_csv(tables / "E3_candidates_summary.csv"))
    table_handling(pd.read_csv(tables / "E5_handling_summary.csv"))


if __name__ == "__main__":
    main()
