"""Build the column-width figures and LaTeX tables used by report/report.tex.

Reads results/qws and results/wsdream, so run make_figures.py for both first.

Usage: python experiments/make_report_figures.py
Output: report/figures/*.pdf, report/tables/*.tex
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import FEASIBLE_BAR, INK_MUTED, STYLE, line

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
DATASETS = {"qws": "QWS 2.0", "wsdream": "WS-DREAM"}

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
CORE = ["Greedy-Repair", "GA", "GA+Greedy"]
RHO = r"Relative tightness $\rho = \tau / \tau^*$"
MODES = ["latency only", "cost only", "reliability only", "latency + cost",
         "latency + reliability", "cost + reliability", "all three"]
MODE_TICKS = ["L", "C", "R", "L+C", "L+R", "C+R", "L+C+R"]


def table(dataset: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / dataset / "tables" / f"{name}_summary.csv")


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {name}.pdf")


def fig_success() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9), sharey=True)
    for ax, (key, label) in zip(axes, DATASETS.items()):
        s = table(key, "E1_relative")
        for name in MAIN:
            d = s[s["algorithm"] == name]
            line(ax, d["label"], d["success_rate"], name)
        ax.set_title(f"({'ab'[list(DATASETS).index(key)]}) {label}", loc="left")
        ax.set_xlabel(RHO)
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].legend(loc="lower left")
    fig.tight_layout(pad=0.3)
    save(fig, "success_rho")


def fig_constraints() -> None:
    """Composer success at the all-three limit, for every constraint combination."""
    x = np.arange(len(MODES))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.75), sharey=True)
    for ax, (key, label) in zip(axes, DATASETS.items()):
        d = table(key, "E4_attribute")
        d = d[d["level"] == 1.0].set_index(["algorithm", "label"])
        for k, name in enumerate(CORE):
            vals = d.loc[name]["success_rate"].reindex(MODES).to_numpy()
            ax.bar(x + (k - 1) * width, vals, width=width - 0.03, color=STYLE[name]["color"],
                   label=name)
        ax.set_xticks(x, MODE_TICKS)
        ax.set_title(f"({'ab'[list(DATASETS).index(key)]}) {label}", loc="left")
        ax.set_xlabel("Constraints tightened to the all-three limit")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(pad=0.3)
    save(fig, "constraint_combinations")


def fig_runtime() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.7), sharey=True)
    for ax, (key, label) in zip(axes, DATASETS.items()):
        s = table(key, "E2_tasks")
        for name in CORE + ["Optimal-MILP"]:
            d = s[s["algorithm"] == name]
            line(ax, d["label"], d["runtime_ms_mean"], name)
        ticks = sorted(s["label"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_title(f"({'ab'[list(DATASETS).index(key)]}) {label}", loc="left")
        ax.set_xlabel("Tasks $N$ ($M=20$)")
    axes[0].set_ylabel("Runtime (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(pad=0.3)
    save(fig, "runtime_tasks")


def fig_convergence() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.6), sharey=True)
    for ax, (key, label) in zip(axes, DATASETS.items()):
        z = np.load(RESULTS / key / "raw" / "E6_convergence.npz")
        for name in ("GA", "GA+Greedy"):
            norm = np.nan_to_num(z[f"{name}_best_utility"] / z["opt_utility"][:, None], nan=0.0)
            line(ax, np.arange(norm.shape[1]), norm.mean(0), name, markevery=50)
        ax.set_title(f"({'ab'[list(DATASETS).index(key)]}) {label}", loc="left")
        ax.set_xlabel("Generation")
        ax.set_ylim(0.8, 1.01)
    axes[0].set_ylabel(r"Best $U/U^*$ (0 until feasible)")
    axes[1].legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, "convergence")


def write_table(name: str, columns: str, header: str, rows: list[str]) -> None:
    body = "\n".join([rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header, r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}"])
    (TABLES / f"{name}.tex").write_text(body + "\n", encoding="utf-8")
    print(f"  {name}.tex")


def table_success() -> None:
    """Success rate vs rho for both datasets side by side."""
    tables = {k: table(k, "E1_relative").pivot(index="label", columns="algorithm",
                                                 values="success_rate") for k in DATASETS}
    order = ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy", "Random"]
    rows = []
    for rho in tables["qws"].index:
        cells = [f"{rho:.2f}"]
        for k in DATASETS:
            cells += [f"{tables[k].loc[rho, n]:.2f}" for n in order]
        rows.append(" & ".join(cells) + r" \\")
    header = (r" & \multicolumn{5}{c}{QWS 2.0} & \multicolumn{5}{c}{WS-DREAM} \\"
              "\n" r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}"
              "\n" r"$\rho$ & GL & GR & GA & G+G & Rnd & GL & GR & GA & G+G & Rnd \\")
    write_table("success_rho", "c" * 11, header, rows)


def table_gap_runtime() -> None:
    """Mean gap at the hardest solvable levels and runtime range, per dataset."""
    rows = []
    for k, label in DATASETS.items():
        s = table(k, "E1_relative").set_index(["algorithm", "label"])
        for name in ["Greedy-Local", "Greedy-Repair", "GA", "GA+Greedy", "Random", "Optimal-MILP"]:
            gaps = []
            for rho in (0.90, 0.95, 0.98):
                r = s.loc[(name, rho)]
                gaps.append("--" if r["successes"] < 5 else f"{r['gap_pct_mean']:.3f}")
            rt = s.loc[name]["runtime_ms_mean"]
            rows.append(f"{label if name == 'Greedy-Local' else ''} & {name} & "
                        + " & ".join(gaps) + f" & {rt.min():.1f}--{rt.max():.1f} \\\\")
        if k == "qws":
            rows.append(r"\midrule")
    header = r"Dataset & Composer & $\rho=0.90$ & $0.95$ & $0.98$ & Runtime (ms) \\"
    write_table("gap_runtime", "llcccc", header, rows)


def table_combinations() -> None:
    """Solvable instances (of 30) at 1.2 x the all-three limit."""
    rows = []
    for k, label in DATASETS.items():
        d = table(k, "E4_attribute")
        d = d[(d["level"] == 1.2) & (d["algorithm"] == "Optimal-MILP")].set_index("label")
        solv = d["solvable"].reindex(MODES).fillna(0).astype(int)
        rows.append(f"{label} & " + " & ".join(str(v) for v in solv) + r" \\")
    header = "Dataset & " + " & ".join(MODE_TICKS) + r" \\"
    write_table("combinations", "l" + "c" * len(MODES), header, rows)


def table_scale() -> None:
    rows = []
    for k, label in DATASETS.items():
        for sym, name in (("N", "E2_tasks"), ("M", "E3_candidates")):
            s = table(k, name)
            for size in sorted(s["label"].unique()):
                d = s[s["label"] == size].set_index("algorithm")
                cells = [label if (sym, size) == ("N", 5) else "", f"${sym}={int(size)}$"]
                cells += [f"{d.loc[n, 'success_rate']:.2f}" for n in ["Greedy-Local", "Random"]]
                cells += [f"{d.loc[n, 'gap_pct_mean']:.3f}" for n in CORE]
                cells += [f"{d.loc[n, 'runtime_ms_mean']:.1f}" for n in ["Greedy-Repair", "GA", "Optimal-MILP"]]
                rows.append(" & ".join(cells) + r" \\")
        if k == "qws":
            rows.append(r"\midrule")
    header = (r" & & \multicolumn{2}{c}{Success} & \multicolumn{3}{c}{Gap (\%)} & "
              r"\multicolumn{3}{c}{Runtime (ms)} \\"
              "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}"
              "\n" r"Data & Size & GL & Rnd & GR & GA & G+G & GR & GA & MILP \\")
    write_table("scale", "llcccccccc", header, rows)


def table_handling() -> None:
    names = ["GA (Deb)", "GA (penalty λ=1)", "GA (penalty λ=10)", "GA (penalty λ=100)", "GA+Greedy"]
    rows = []
    for k, label in DATASETS.items():
        s = table(k, "E5_handling").set_index(["label", "algorithm"])
        cells = [label] + [f"{s.loc[(1.0, n), 'success_rate']:.2f}" for n in names]
        rows.append(" & ".join(cells) + r" \\")
    header = r"Dataset & Deb & $\lambda=1$ & $\lambda=10$ & $\lambda=100$ & GA+Greedy \\"
    write_table("handling", "lccccc", header, rows)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    fig_success()
    fig_constraints()
    fig_runtime()
    fig_convergence()
    table_success()
    table_gap_runtime()
    table_combinations()
    table_scale()
    table_handling()


if __name__ == "__main__":
    main()
