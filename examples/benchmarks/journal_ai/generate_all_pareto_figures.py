"""Unified generation script for Figures 1, 2, and 3.

Generates consistent, publication-quality Pareto trade-off figures across:
- Figure 1: Classification (Macro-F1 vs. Atoms) -> figures/classification_f1_vs_complexity.pdf
- Figure 2: Regression (R2 vs. Atoms) -> figures/regression_r2_vs_complexity.pdf
- Figure 3: Cluster Explanation (Fidelity vs. Atoms) -> figures/cluster_fidelity_vs_complexity.pdf
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_all_figures():
    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    style_name = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style_name)
    
    # Common model styling metadata
    models_meta = {
        "ruleNSGA-II": {"name": "ruleNSGA-II", "color": "#7570b3", "marker": "^", "size": 220},
        "ruleGP": {"name": "ruleGP", "color": "#d95f02", "marker": "o", "size": 220},
        "rulePLCS": {"name": "rulePLCS", "color": "#1b9e77", "marker": "D", "size": 200},
        "CART (d=4)": {"name": "CART (depth 4)", "color": "#377eb8", "marker": "s", "size": 200},
        "CART (Tree SRS)": {"name": "CART (Tree SRS)", "color": "#377eb8", "marker": "s", "size": 200},
        "CART (Full)": {"name": "CART (Full)", "color": "#4a4a4a", "marker": "X", "size": 200},
        "RuleFit": {"name": "RuleFit", "color": "#e41a1c", "marker": "*", "size": 280},
    }
    
    # =========================================================================
    # Figure 1: Classification Pareto Plot
    # =========================================================================
    csv_cls = script_dir / "results_classification/classification_5fold_benchmark_results.csv"
    df_cls = pd.read_csv(csv_cls)
    agg_cls = df_cls.groupby("model").agg({
        "macro_f1": "mean",
        "mean_atoms": "mean",
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    ax.axvspan(1, 20, color="#e6f5d0", alpha=0.5, zorder=0, label=r"High Simulatability Region ($\leq 20$ atoms)")
    
    for _, row in agg_cls.iterrows():
        m = row["model"]
        meta = models_meta.get(m, {"name": m, "color": "black", "marker": "o", "size": 200})
        f1_val = row["macro_f1"]
        atoms_val = row["mean_atoms"]
        
        ax.scatter(
            atoms_val,
            f1_val,
            color=meta["color"],
            marker=meta["marker"],
            s=meta["size"],
            label=f"{meta['name']} (Macro-F1={f1_val:.3f}, Atoms={atoms_val:.1f})",
            zorder=5,
            edgecolor="black",
            linewidth=1.2
        )
    
    # Pareto frontier points for classification
    pareto_cls = [(5.4, 0.761), (6.0, 0.836), (42.1, 0.874), (115.6, 0.906)]
    px, py = zip(*pareto_cls)
    ax.plot(px, py, linestyle="--", color="#666666", alpha=0.8, linewidth=1.5, zorder=3, label="Empirical Pareto Frontier")
    
    ax.set_xscale("log")
    ax.set_xlim(2, 2000)
    ax.set_ylim(0.70, 0.95)
    ax.set_xlabel("Average Model Complexity / Total Condition Count (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Out-of-Sample Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Trade-off: Classification Predictive Performance vs. Model Complexity", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    
    fig.savefig(figures_dir / "classification_f1_vs_complexity.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "classification_f1_vs_complexity.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[SUCCESS] Saved Figure 1 -> classification_f1_vs_complexity.pdf")
    
    # =========================================================================
    # Figure 2: Regression Pareto Plot
    # =========================================================================
    csv_reg = script_dir / "results_regression/regression_benchmark_results.csv"
    df_reg = pd.read_csv(csv_reg)
    
    model_mapping_reg = {
        "cart_full": "CART (Full)",
        "cart_d4": "CART (d=4)",
        "rulegp": "ruleGP",
        "rulensga2": "ruleNSGA-II",
        "ruleplcs": "rulePLCS",
        "rulefit": "RuleFit",
    }
    
    df_reg["model_name_clean"] = df_reg["model_key"].map(model_mapping_reg)
    agg_reg = df_reg.groupby("model_name_clean").agg({
        "mean_r2": "mean",
        "mean_atoms": "mean",
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    ax.axvspan(1, 20, color="#e6f5d0", alpha=0.5, zorder=0, label=r"High Simulatability Region ($\leq 20$ atoms)")
    
    for _, row in agg_reg.iterrows():
        m = row["model_name_clean"]
        meta = models_meta.get(m, {"name": m, "color": "black", "marker": "o", "size": 200})
        r2_val = row["mean_r2"]
        atoms_val = row["mean_atoms"]
        
        ax.scatter(
            atoms_val,
            r2_val,
            color=meta["color"],
            marker=meta["marker"],
            s=meta["size"],
            label=f"{meta['name']} ($R^2$={r2_val:.3f}, Atoms={atoms_val:.1f})",
            zorder=5,
            edgecolor="black",
            linewidth=1.2
        )
    
    # Pareto frontier points for regression: ruleNSGA-II (6.0, 0.522) -> ruleGP (11.8, 0.536) -> RuleFit (45.9, 0.749)
    pareto_reg = [(6.0, 0.522), (11.8, 0.536), (45.9, 0.749)]
    px, py = zip(*pareto_reg)
    ax.plot(px, py, linestyle="--", color="#666666", alpha=0.8, linewidth=1.5, zorder=3, label="Empirical Pareto Frontier")
    
    ax.set_xscale("log")
    ax.set_xlim(2, 8000)
    ax.set_ylim(0.35, 0.90)
    ax.set_xlabel("Average Model Complexity / Total Condition Count (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Out-of-Sample Predictive $R^2$", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Trade-off: Continuous Regression Accuracy vs. Model Complexity", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right", framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    
    fig.savefig(figures_dir / "regression_r2_vs_complexity.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "regression_r2_vs_complexity.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[SUCCESS] Saved Figure 2 -> regression_r2_vs_complexity.pdf")
    
    # =========================================================================
    # Figure 3: Cluster Explanation Pareto Plot
    # =========================================================================
    csv_cl = script_dir / "results_cluster/cluster_explanation_benchmark_results.csv"
    df_cl = pd.read_csv(csv_cl)
    
    # Aggregate over kmeans runs (matching Table 5)
    df_cl_kmeans = df_cl[df_cl["cluster_algorithm"] == "kmeans"]
    model_mapping_cl = {
        "cart": "CART (Tree SRS)",
        "rulegp": "ruleGP",
        "rulensga2": "ruleNSGA-II",
        "ruleplcs": "rulePLCS",
    }
    df_cl_kmeans["model_name_clean"] = df_cl_kmeans["backend"].map(model_mapping_cl)
    agg_cl = df_cl_kmeans.groupby("model_name_clean").agg({
        "test_fidelity": "mean",
        "mean_atoms": "mean",
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    ax.axvspan(1, 20, color="#e6f5d0", alpha=0.5, zorder=0, label=r"High Simulatability Region ($\leq 20$ atoms)")
    
    for _, row in agg_cl.iterrows():
        m = row["model_name_clean"]
        meta = models_meta.get(m, {"name": m, "color": "black", "marker": "o", "size": 200})
        fid_val = row["test_fidelity"] * 100.0
        atoms_val = row["mean_atoms"]
        
        ax.scatter(
            atoms_val,
            fid_val,
            color=meta["color"],
            marker=meta["marker"],
            s=meta["size"],
            label=f"{meta['name']} (Fidelity={fid_val:.1f}%, Atoms={atoms_val:.1f})",
            zorder=5,
            edgecolor="black",
            linewidth=1.2
        )
    
    # Pareto frontier points for clustering: ruleNSGA-II (3.2, 90.5) -> ruleGP (4.5, 94.8) -> CART (31.1, 95.6)
    pareto_cl = [(3.2, 90.5), (4.5, 94.8), (31.1, 95.6)]
    px, py = zip(*pareto_cl)
    ax.plot(px, py, linestyle="--", color="#666666", alpha=0.8, linewidth=1.5, zorder=3, label="Empirical Pareto Frontier")
    
    ax.set_xscale("log")
    ax.set_xlim(1.5, 100)
    ax.set_ylim(85, 100)
    ax.set_xlabel("Average Model Complexity / Total Condition Count (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Generalization Fidelity (%)", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Trade-off: Cluster Explanation Fidelity vs. Model Complexity", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    
    fig.savefig(figures_dir / "cluster_fidelity_vs_complexity.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "cluster_fidelity_vs_complexity.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[SUCCESS] Saved Figure 3 -> cluster_fidelity_vs_complexity.pdf")

if __name__ == "__main__":
    generate_all_figures()
