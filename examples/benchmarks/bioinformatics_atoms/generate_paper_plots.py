#!/usr/bin/env python3
"""Generate publication-quality figures for Sections 4.5 and 4.6."""

from __future__ import annotations

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Styling parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.handlelength": 3.5,
    "legend.handletextpad": 0.6,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
    "grid.alpha": 0.3,
})

BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "experiments_snapshot" / "csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL_NAMES = {
    "wrapper_rulegp": "RuleGP",
    "wrapper_rulensga2": "RuleNSGA2",
    "wrapper_ruleplcs": "RulePLCS",
}

# -------------------------------------------------------------
# Figure 2: Atom Construction (SNP Main & Gene-Environment, p <= 1000)
# -------------------------------------------------------------
def generate_figure_construction():
    csv_main = CSV_DIR / "snp_glm_main_construction_summary.csv"
    csv_env = CSV_DIR / "snp_glm_environment_construction_summary.csv"
    
    data_main = {}
    with open(csv_main, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["model"]
            arm = row["arm"]
            p = int(row["n_snps"])
            f1 = float(row["f1_macro_mean"])
            t = float(row["fit_seconds_mean"])
            data_main.setdefault(m, {}).setdefault(arm, {})[p] = (f1, t)

    data_env = {}
    with open(csv_env, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["model"]
            arm = row["arm"]
            p = int(row["n_snps"])
            f1 = float(row["f1_macro_mean"])
            t = float(row["fit_seconds_mean"])
            data_env.setdefault(m, {}).setdefault(arm, {})[p] = (f1, t)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    
    models = ["wrapper_rulegp", "wrapper_rulensga2", "wrapper_ruleplcs"]
    colors = {
        "wrapper_rulegp": "#1f77b4",
        "wrapper_rulensga2": "#d62728",
        "wrapper_ruleplcs": "#2ca02c",
    }
    
    # Panel (a): SNP Track - Fit Time
    ax = axes[0, 0]
    for m in models:
        pts_gen = sorted(data_main[m]["hybrid"].items())
        pts_typ = sorted(data_main[m]["genotype_aware"].items())
        p_vals = [p for p, _ in pts_gen]
        t_gen = [t for _, (_, t) in pts_gen]
        t_typ = [t for _, (_, t) in pts_typ]
        
        lbl = MODEL_NAMES[m]
        ax.plot(p_vals, t_gen, linestyle="--", dashes=(4, 3), marker="o", color=colors[m], label=f"{lbl} (Generic)")
        ax.plot(p_vals, t_typ, linestyle="-", marker="s", color=colors[m], label=f"{lbl} (Type-Aware)")
    ax.set_ylabel("Fit Time [s]")
    ax.set_title("(a) SNP Track: Fit Time ($n=2{,}000$)")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, fontsize=7.5, handlelength=3.5)

    # Panel (b): SNP Track - Macro-F1
    ax = axes[0, 1]
    for m in models:
        pts_gen = sorted(data_main[m]["hybrid"].items())
        pts_typ = sorted(data_main[m]["genotype_aware"].items())
        p_vals = [p for p, _ in pts_gen]
        f1_gen = [f1 for _, (f1, _) in pts_gen]
        f1_typ = [f1 for _, (f1, _) in pts_typ]
        
        lbl = MODEL_NAMES[m]
        ax.plot(p_vals, f1_gen, linestyle="--", dashes=(4, 3), marker="o", color=colors[m], label=f"{lbl} (Generic)")
        ax.plot(p_vals, f1_typ, linestyle="-", marker="s", color=colors[m], label=f"{lbl} (Type-Aware)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("(b) SNP Track: Macro-F1 ($n=2{,}000$)")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, fontsize=7.5, handlelength=3.5)

    # Panel (c): Gene-Environment Track - Fit Time
    ax = axes[1, 0]
    for m in models:
        pts_gen = sorted(data_env[m]["generic"].items())
        pts_typ = sorted(data_env[m]["typed"].items())
        p_vals = [p for p, _ in pts_gen]
        t_gen = [t for _, (_, t) in pts_gen]
        t_typ = [t for _, (_, t) in pts_typ]
        
        lbl = MODEL_NAMES[m]
        ax.plot(p_vals, t_gen, linestyle="--", dashes=(4, 3), marker="o", color=colors[m], label=f"{lbl} (Generic)")
        ax.plot(p_vals, t_typ, linestyle="-", marker="s", color=colors[m], label=f"{lbl} (Type-Aware)")
    ax.set_xlabel("Number of Candidate Features ($p$)")
    ax.set_ylabel("Fit Time [s]")
    ax.set_title("(c) Gene-Environment Track: Fit Time ($n=2{,}000$)")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, fontsize=7.5, handlelength=3.5)

    # Panel (d): Gene-Environment Track - Macro-F1
    ax = axes[1, 1]
    for m in models:
        pts_gen = sorted(data_env[m]["generic"].items())
        pts_typ = sorted(data_env[m]["typed"].items())
        p_vals = [p for p, _ in pts_gen]
        f1_gen = [f1 for _, (f1, _) in pts_gen]
        f1_typ = [f1 for _, (f1, _) in pts_typ]
        
        lbl = MODEL_NAMES[m]
        ax.plot(p_vals, f1_gen, linestyle="--", dashes=(4, 3), marker="o", color=colors[m], label=f"{lbl} (Generic)")
        ax.plot(p_vals, f1_typ, linestyle="-", marker="s", color=colors[m], label=f"{lbl} (Type-Aware)")
    ax.set_xlabel("Number of Candidate Features ($p$)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("(d) Gene-Environment Track: Macro-F1 ($n=2{,}000$)")
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, fontsize=7.5, handlelength=3.5)

    plt.tight_layout()
    out_path = FIG_DIR / "figure_construction_time_and_f1.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure 3: Extreme Scaling up to p = 100,000
# -------------------------------------------------------------
def generate_figure_extreme():
    csv_extreme = CSV_DIR / "snp_glm_extreme_fs_vs_typed_summary.csv"
    data = {}
    with open(csv_extreme, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["model"]
            arm = row["arm"]
            p = int(row["n_snps"])
            f1 = float(row["f1_macro_mean"])
            t_fit = float(row["fit_seconds_mean"])
            t_tot = float(row["total_fit_seconds_mean"])
            data.setdefault(m, {}).setdefault(arm, {})[p] = (f1, t_fit, t_tot)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    models = ["wrapper_rulegp", "wrapper_rulensga2", "wrapper_ruleplcs"]
    colors = {"wrapper_rulegp": "#1f77b4", "wrapper_rulensga2": "#d62728", "wrapper_ruleplcs": "#2ca02c"}

    # Panel (a): Downstream Fit Time vs p (log-x, linear-y)
    ax = axes[0]
    p_vals = [2000, 10000, 50000, 100000]
    for m in models:
        lbl = MODEL_NAMES[m]
        t_gen = [data[m]["fs_generic"][p][1] for p in p_vals]
        t_typ = [data[m]["fs_typed"][p][1] for p in p_vals]
        ax.plot(p_vals, t_gen, linestyle="--", dashes=(4, 3), marker="o", color=colors[m], label=f"{lbl} (FS + Generic)")
        ax.plot(p_vals, t_typ, linestyle="-", marker="s", color=colors[m], label=f"{lbl} (FS + Type-Aware)")

    ax.set_xscale("log")
    ax.set_xlabel("Total Feature Count $p$")
    ax.set_ylabel("Fit Time [s] (after FS)")
    ax.set_ylim(0.5, 10.0)
    ax.set_title("(a) Downstream Fit Time across Dimensions ($k=20$)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=8, handlelength=3.5)

    # Panel (b): Predictive Macro-F1 across dimensions
    ax = axes[1]
    for m in models:
        lbl = MODEL_NAMES[m]
        f1_gen = [data[m]["fs_generic"][p][0] for p in p_vals]
        f1_typ = [data[m]["fs_typed"][p][0] for p in p_vals]
        ax.plot(p_vals, f1_gen, linestyle="--", dashes=(4, 3), marker="o", color=colors[m], label=f"{lbl} (FS + Generic)")
        ax.plot(p_vals, f1_typ, linestyle="-", marker="s", color=colors[m], label=f"{lbl} (FS + Type-Aware)")

    ax.set_xscale("log")
    ax.set_xlabel("Total Feature Count $p$")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.40, 0.58)
    ax.set_title("(b) Predictive Stability across Dimensions")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8, handlelength=3.5)

    plt.tight_layout()
    out_path = FIG_DIR / "figure_extreme_scaling.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# -------------------------------------------------------------
# Figure (single column): RuleGP fit-time speedup from type-aware construction
# -------------------------------------------------------------
def generate_figure_rulegp_speedup():
    csv_main = CSV_DIR / "snp_glm_main_construction_summary.csv"
    fit = {}  # p -> {"hybrid": t, "genotype_aware": t}
    for row in csv.DictReader(open(csv_main, encoding="utf-8")):
        if row["model"] != "wrapper_rulegp":
            continue
        p = int(row["n_snps"])
        fit.setdefault(p, {})[row["arm"]] = float(row["fit_seconds_mean"])

    p_vals = sorted(fit)
    t_gen = [fit[p]["hybrid"] for p in p_vals]
    t_typ = [fit[p]["genotype_aware"] for p in p_vals]

    import numpy as np
    x = np.arange(len(p_vals))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    b1 = ax.bar(x - w / 2, t_gen, w, color="#9ecae1", edgecolor="#1f77b4",
                hatch="//", label="Generic")
    b2 = ax.bar(x + w / 2, t_typ, w, color="#1f77b4", edgecolor="#1f77b4",
                label="Type-Aware")

    for xi, tg, tt in zip(x, t_gen, t_typ):
        ax.annotate(f"${tg / tt:.1f}\\times$", xy=(xi, tg), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}" for p in p_vals])
    ax.set_xlabel("Candidate features $p$")
    ax.set_ylabel("Fit time [s]")
    ax.set_ylim(0, max(t_gen) * 1.18)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout(pad=0.4)
    out_path = FIG_DIR / "figure_rulegp_speedup.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_figure_construction()
    generate_figure_extreme()
    generate_figure_rulegp_speedup()
