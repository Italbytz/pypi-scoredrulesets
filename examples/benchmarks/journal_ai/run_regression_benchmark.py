"""Comprehensive Regression Benchmark for Scored Rule Sets.

Evaluates regression scored rule sets across multiple tabular benchmarks,
comparing Tree-based SRS, Evolutionary Projected SRS (RuleGP, RuleNSGA-II),
and linear/tree baselines.
"""

from __future__ import annotations

import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    make_friedman1,
    make_friedman2,
    make_friedman3,
)
from imodels import RuleFitRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetRegressor


def get_regression_datasets():
    datasets = []
    
    # 1. Diabetes (Real Biomedical)
    X_diab, y_diab = load_diabetes(return_X_y=True)
    datasets.append((X_diab, y_diab, "Diabetes (Real Biomedical)", [f"Feature {i+1}" for i in range(X_diab.shape[1])]))
    
    # 2. Friedman #1 (Synthetic: 5 informative + 5 noise features, nonlinear)
    X_f1, y_f1 = make_friedman1(n_samples=400, n_features=10, noise=1.0, random_state=42)
    datasets.append((X_f1, y_f1, "Friedman #1 (Nonlinear Interaction)", [f"x{i+1}" for i in range(10)]))
    
    # 3. Friedman #2 (Synthetic: 4 features, severe nonlinearity)
    X_f2, y_f2 = make_friedman2(n_samples=400, noise=1.0, random_state=42)
    datasets.append((X_f2, y_f2, "Friedman #2 (Severe Nonlinearity)", [f"x{i+1}" for i in range(4)]))
    
    # 4. Friedman #3 (Synthetic: 4 features, impedance model)
    X_f3, y_f3 = make_friedman3(n_samples=400, noise=0.1, random_state=42)
    datasets.append((X_f3, y_f3, "Friedman #3 (Impedance)", [f"x{i+1}" for i in range(4)]))
    
    # 5. California Housing (Subsampled N=800 for cross-validation speed)
    cal = fetch_california_housing()
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(cal.data.shape[0], size=800, replace=False)
    X_cal, y_cal = cal.data[sub_idx], cal.target[sub_idx]
    datasets.append((X_cal, y_cal, "California Housing (Spatial/Macro)", list(cal.feature_names)))
    
    return datasets


def count_rf_atoms(estimator) -> int:
    """Extract literal condition count across active rules in RuleFit."""
    if hasattr(estimator, "_get_rules"):
        df = estimator._get_rules()
        active = df[(df.coef != 0) & (df.type == 'rule')]
        return sum(r.count('&') + 1 for r in active['rule'])
    return 0


def run_regression_benchmark(n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    datasets = get_regression_datasets()
    
    models = [
        ("cart_full", "CART (Full Tree SRS)", lambda: ScoredRuleSetRegressor(backend="cart", random_state=random_state)),
        ("cart_d4", "CART (Depth-4 SRS)", lambda: ScoredRuleSetRegressor(backend="cart", backend_params={"max_depth": 4}, random_state=random_state)),
        ("rulegp", "ruleGP (SRS Regressor)", lambda: ScoredRuleSetRegressor(backend="rulegp", random_state=random_state)),
        ("rulensga2", "ruleNSGA-II (SRS Regressor)", lambda: ScoredRuleSetRegressor(backend="rulensga2", random_state=random_state)),
        ("ruleplcs", "rulePLCS (SRS Regressor)", lambda: ScoredRuleSetRegressor(backend="ruleplcs", random_state=random_state)),
        ("rulefit", "RuleFit (Rule Ensemble)", lambda: RuleFitRegressor(random_state=random_state, max_rules=50)),
    ]
    
    results: List[Dict[str, Any]] = []
    
    for X, y, ds_name, feat_names in datasets:
        print(f"\n==================================================================")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]})")
        print(f"==================================================================")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        for key, model_name, model_fn in models:
            r2_list = []
            rmse_list = []
            mae_list = []
            rules_list = []
            atoms_list = []
            times_list = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                scaler = StandardScaler().fit(X[train_idx])
                X_train = scaler.transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]
                
                start_time = time.time()
                est = model_fn()
                est.fit(X_train, y_train)
                elapsed = time.time() - start_time
                
                preds = est.predict(X_test)
                
                r2 = float(r2_score(y_test, preds))
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                mae = float(mean_absolute_error(y_test, preds))
                
                if hasattr(est, "to_ruleset"):
                    ruleset = est.to_ruleset()
                    n_rules = len(ruleset.rules)
                    n_atoms = sum(len(r.atoms) for r in ruleset.rules)
                elif key == "rulefit":
                    n_atoms = count_rf_atoms(est)
                    n_rules = n_atoms
                else:
                    n_rules = 0
                    n_atoms = 0
                
                r2_list.append(r2)
                rmse_list.append(rmse)
                mae_list.append(mae)
                rules_list.append(n_rules)
                atoms_list.append(n_atoms)
                times_list.append(elapsed)
            
            mean_r2 = float(np.mean(r2_list))
            std_r2 = float(np.std(r2_list))
            mean_rmse = float(np.mean(rmse_list))
            mean_mae = float(np.mean(mae_list))
            mean_rules = float(np.mean(rules_list))
            mean_atoms = float(np.mean(atoms_list))
            mean_time = float(np.mean(times_list))
            
            print(f"[{model_name:>26}]: R2 = {mean_r2:6.3f} +/- {std_r2:5.3f} | RMSE = {mean_rmse:7.3f} | Rules = {mean_rules:5.1f} | Atoms = {mean_atoms:6.1f} | Time = {mean_time:5.3f}s")
            
            results.append({
                "dataset": ds_name,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "model_key": key,
                "model_name": model_name,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "mean_rmse": mean_rmse,
                "mean_mae": mean_mae,
                "mean_rules": mean_rules,
                "mean_atoms": mean_atoms,
                "mean_time": mean_time,
            })

    df = pd.DataFrame(results)
    return df


def generate_regression_plots_and_summary(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw CSV
    df.to_csv(output_dir / "regression_benchmark_results.csv", index=False)
    
    # Create Pareto Plot: R2 vs Atom Count
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", alpha=0.5, color="#d0d0d0", zorder=1)
    
    colors = {
        "cart_full": "#4daf4a",
        "cart_d4": "#377eb8",
        "rulegp": "#ff7f00",
        "rulensga2": "#984ea3",
        "ruleplcs": "#a65628",
        "rulefit": "#e41a1c",
    }
    markers = {
        "cart_full": "X",
        "cart_d4": "s",
        "rulegp": "o",
        "rulensga2": "^",
        "ruleplcs": "D",
        "rulefit": "*",
    }
    
    agg_df = df.groupby(["model_key", "model_name"]).agg({
        "mean_r2": "mean",
        "mean_atoms": "mean",
        "mean_rules": "mean",
        "mean_time": "mean"
    }).reset_index()
    
    for _, row in agg_df.iterrows():
        k = row["model_key"]
        r2_val = row["mean_r2"]
        r2_plot = max(r2_val, -0.15)
        ax.scatter(
            row["mean_atoms"],
            r2_plot,
            color=colors.get(k, "black"),
            marker=markers.get(k, "o"),
            s=220,
            label=f"{row['model_name']} ($R^2$={r2_val:.3f}, Atoms={row['mean_atoms']:.1f})",
            zorder=5,
            edgecolor="black",
            linewidth=1.2
        )
    
    ax.set_xscale("log")
    ax.set_xlim(3, 10000)
    ax.set_ylim(-0.25, 0.70)
    ax.set_xlabel("Average Model Complexity / Total Condition Count (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Out-of-Sample Predictive $R^2$", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Frontier: Interpretable Rule-Based Regression Accuracy vs. Model Complexity", fontsize=12, fontweight="bold", pad=12)
    ax.axvspan(3, 20, color="#e6f5d0", alpha=0.5, zorder=0, label=r"High Simulatability Region ($\leq 20$ atoms)")
    ax.legend(frameon=True, fontsize=9.5, loc="lower left", framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    fig.savefig(output_dir / "regression_r2_vs_complexity.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "regression_r2_vs_complexity.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    # Generate 1D / 2D Regression Function Approximation Visualization
    generate_regression_surface_plot(output_dir)
    
    # Generate LaTeX Summary Table
    latex_table = generate_regression_latex_table(df)
    with open(output_dir / "regression_benchmark_table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\n[INFO] Regression benchmark artifacts saved to {output_dir}")


def generate_regression_surface_plot(output_dir: Path):
    """Plot 1D nonlinear continuous target approximation."""
    rng = np.random.RandomState(42)
    X = np.sort(rng.uniform(-3, 3, size=(120, 1)), axis=0)
    # Ground truth: smooth nonlinear curve
    y = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 0]) + rng.normal(0, 0.15, size=X.shape[0])
    
    X_grid = np.linspace(-3.2, 3.2, 300).reshape(-1, 1)
    y_true = np.sin(X_grid[:, 0]) + 0.5 * np.cos(2 * X_grid[:, 0])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    models = [
        ("CART (Depth-3 SRS)", ScoredRuleSetRegressor(backend="cart", backend_params={"max_depth": 3}, random_state=42)),
        ("ruleGP (SRS Regressor)", ScoredRuleSetRegressor(backend="rulegp", random_state=42)),
        ("ruleNSGA-II (SRS Regressor)", ScoredRuleSetRegressor(backend="rulensga2", random_state=42)),
    ]
    
    for ax, (title, model) in zip(axes, models):
        model.fit(X, y)
        preds = model.predict(X_grid)
        ruleset = model.to_ruleset()
        n_rules = len(ruleset.rules)
        n_atoms = sum(len(r.atoms) for r in ruleset.rules)
        r2 = r2_score(y, model.predict(X))
        
        ax.scatter(X, y, color="#2b5c8f", alpha=0.6, s=30, label="Data Samples")
        ax.plot(X_grid, y_true, "k--", alpha=0.5, linewidth=1.5, label="True Function")
        ax.step(X_grid, preds, color="#d95f02", linewidth=2.2, label="SRS Prediction")
        
        ax.set_title(f"{title}\n$R^2$={r2:.2f} | Rules={n_rules} | Atoms={n_atoms}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Input Feature $x$", fontsize=10)
        ax.set_ylabel("Continuous Target $y$", fontsize=10)
        ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    
    plt.tight_layout()
    fig.savefig(output_dir / "regression_1d_step_approximation.pdf")
    fig.savefig(output_dir / "regression_1d_step_approximation.png")
    plt.close()


def generate_regression_latex_table(df: pd.DataFrame) -> str:
    piv_r2 = df.pivot(index="dataset", columns="model_key", values="mean_r2")
    piv_std = df.pivot(index="dataset", columns="model_key", values="std_r2")
    piv_atoms = df.pivot(index="dataset", columns="model_key", values="mean_atoms")
    
    order = ["cart_full", "cart_d4", "rulegp", "rulensga2", "ruleplcs", "rulefit"]
    col_names = {
        "cart_full": "CART (Full)",
        "cart_d4": "CART (d=4)",
        "rulegp": "ruleGP (SRS)",
        "rulensga2": "ruleNSGA-II (SRS)",
        "ruleplcs": "rulePLCS (SRS)",
        "rulefit": "RuleFit (Ensemble)",
    }
    
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Regression performance across continuous tabular benchmark datasets (5-fold cross-validation). Values report Out-of-Sample $R^2$ $\pm$ standard deviation and total model atom count in parentheses.}",
        r"\label{tab:regression_benchmark}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Dataset} & " + " & ".join([f"\\textbf{{{col_names[k]}}}" for k in order]) + r" \\",
        r"\midrule"
    ]
    
    for ds in piv_r2.index:
        row_str = f"{ds}"
        for k in order:
            r2 = piv_r2.loc[ds, k]
            std = piv_std.loc[ds, k]
            atoms = piv_atoms.loc[ds, k]
            row_str += f" & {r2:.3f} $\\pm$ {std:.3f} ({atoms:.1f})"
        row_str += r" \\"
        lines.append(row_str)
        
    lines.extend([
        r"\midrule",
        r"\textbf{Average} & " + " & ".join([
            f"{df[df['model_key']==k]['mean_r2'].mean():.3f} ({df[df['model_key']==k]['mean_atoms'].mean():.1f})"
            for k in order
        ]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "results_regression"
    df_results = run_regression_benchmark(n_splits=5, random_state=42)
    generate_regression_plots_and_summary(df_results, out_dir)
