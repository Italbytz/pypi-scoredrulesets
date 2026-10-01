"""Generate Classification Pareto Plot: Macro-F1 vs. Model Complexity (Atom Count)."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_classification_pareto_plot():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "results_classification/classification_5fold_benchmark_results.csv"
    output_dir = base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Calculate overall averages per model across all 9 datasets
    agg_df = df.groupby("model").agg({
        "macro_f1": "mean",
        "macro_f1_std": "mean",
        "mean_atoms": "mean",
        "mean_time": "mean",
    }).reset_index()
    
    # Preferred order & metadata
    models_meta = {
        "ruleNSGA-II": {"name": "ruleNSGA-II", "color": "#7570b3", "marker": "^", "size": 220},
        "ruleGP": {"name": "ruleGP", "color": "#d95f02", "marker": "o", "size": 220},
        "rulePLCS": {"name": "rulePLCS", "color": "#1b9e77", "marker": "D", "size": 200},
        "CART (d=4)": {"name": "CART (depth 4)", "color": "#377eb8", "marker": "s", "size": 200},
        "CART (Full)": {"name": "CART (Full)", "color": "#4a4a4a", "marker": "X", "size": 200},
        "RuleFit": {"name": "RuleFit", "color": "#e41a1c", "marker": "*", "size": 280},
    }
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    # Shaded region for high simulatability
    ax.axvspan(1, 20, color="#e6f5d0", alpha=0.5, zorder=0, label=r"High Simulatability Region ($\leq 20$ atoms)")
    
    for _, row in agg_df.iterrows():
        m = row["model"]
        meta = models_meta.get(m, {"name": m, "color": "black", "marker": "o", "size": 180})
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
    
    # Connect empirical Pareto frontier: ruleNSGA-II -> ruleGP -> rulePLCS -> RuleFit
    # (Note: ruleNSGA-II: 5.4 atoms, 0.761 F1; ruleGP: 6.0 atoms, 0.836 F1; rulePLCS: 42.1 atoms, 0.874 F1; RuleFit: 115.6 atoms, 0.906 F1)
    pareto_points = [
        (5.4, 0.761),
        (6.0, 0.836),
        (42.1, 0.874),
        (115.6, 0.906),
    ]
    px, py = zip(*pareto_points)
    ax.plot(px, py, linestyle="--", color="#666666", alpha=0.75, linewidth=1.5, zorder=3, label="Empirical Pareto Frontier")
    
    ax.set_xscale("log")
    ax.set_xlim(2, 2000)
    ax.set_ylim(0.70, 0.95)
    ax.set_xlabel("Average Model Complexity / Total Condition Count (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Out-of-Sample Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Trade-off: Classification Predictive Performance vs. Model Complexity", fontsize=12, fontweight="bold", pad=12)
    
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    
    pdf_out = output_dir / "classification_f1_vs_complexity.pdf"
    png_out = output_dir / "classification_f1_vs_complexity.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[SUCCESS] Classification Pareto plot saved to {pdf_out}")

if __name__ == "__main__":
    generate_classification_pareto_plot()
