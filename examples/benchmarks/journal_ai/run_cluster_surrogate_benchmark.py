"""Comprehensive Cluster Explanation Benchmark for Scored Rule Sets.

Evaluates surrogate rule sets across multiple clustering algorithms,
tabular datasets, and rule learning backends.
"""

from __future__ import annotations

import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_blobs,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


def load_seeds_dataset():
    """Load Seeds dataset from OpenML or synthetic fallback."""
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml(name="seeds", version=1, as_frame=False, parser="auto")
        X = data.data
        return X, 3, "Seeds", [f"Feature {i+1}" for i in range(X.shape[1])]
    except Exception:
        # High quality synthetic seeds-like dataset (3 clusters, 7 features)
        X, _ = make_blobs(n_samples=210, n_features=7, centers=3, cluster_std=1.2, random_state=42)
        return X, 3, "Seeds (Synthetic)", [f"Feature {i+1}" for i in range(X.shape[1])]


def get_all_datasets():
    datasets = []
    
    # 1. Iris
    iris = load_iris()
    datasets.append((iris.data, 3, "Iris", list(iris.feature_names)))
    
    # 2. Wine
    wine = load_wine()
    datasets.append((wine.data, 3, "Wine", list(wine.feature_names)))
    
    # 3. Seeds
    X_seeds, k_seeds, name_seeds, feats_seeds = load_seeds_dataset()
    datasets.append((X_seeds, k_seeds, name_seeds, feats_seeds))
    
    # 4. Breast Cancer
    cancer = load_breast_cancer()
    datasets.append((cancer.data, 2, "Breast Cancer", list(cancer.feature_names)))
    
    # 5. Synthetic 2D Blobs (for interpretable geometric analysis)
    X_blobs, _ = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.85, random_state=42)
    datasets.append((X_blobs, 4, "Synthetic 2D (4 Clusters)", ["Feature 1", "Feature 2"]))
    
    # 6. Synthetic Anisotropic Blobs (elongated clusters)
    X_aniso, _ = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.7, random_state=100)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    datasets.append((X_aniso, 3, "Anisotropic 2D (3 Clusters)", ["Feature 1", "Feature 2"]))
    
    return datasets


def run_benchmark(n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    datasets = get_all_datasets()
    backends = ["cart", "rulegp", "rulensga2", "ruleplcs"]
    cluster_algos = ["kmeans", "gmm"]
    
    results: List[Dict[str, Any]] = []
    
    for X, k, ds_name, feat_names in datasets:
        print(f"\n==================================================================")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]}, K={k})")
        print(f"==================================================================")
        
        # Standardize for clustering stability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for algo_name in cluster_algos:
            if algo_name == "kmeans":
                base_clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            elif algo_name == "gmm":
                base_clusterer = GaussianMixture(n_components=k, random_state=random_state)
            else:
                continue
            
            # Step 1: Generate ground truth cluster partition
            base_clusterer.fit(X_scaled)
            if hasattr(base_clusterer, "predict"):
                cluster_labels = base_clusterer.predict(X_scaled)
            else:
                cluster_labels = base_clusterer.labels_
            
            # Step 2: Cross-validation on surrogate fidelity
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            for backend in backends:
                train_fidelities = []
                test_fidelities = []
                rule_counts = []
                atom_counts = []
                fit_times = []
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, cluster_labels)):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = cluster_labels[train_idx], cluster_labels[test_idx]
                    
                    classifier = ScoredRuleSetClassifier(
                        backend=backend,
                        random_state=random_state + fold,
                    )
                    
                    start_time = time.time()
                    classifier.fit(X_train, y_train)
                    ruleset = classifier.to_ruleset()
                    elapsed = time.time() - start_time
                    
                    # Predictions
                    pred_train = classifier.predict(X_train)
                    pred_test = classifier.predict(X_test)
                    
                    train_fid = float(np.mean(pred_train == y_train))
                    test_fid = float(np.mean(pred_test == y_test))
                    
                    n_rules = len(ruleset.rules)
                    n_atoms = sum(len(r.atoms) for r in ruleset.rules)
                    
                    train_fidelities.append(train_fid)
                    test_fidelities.append(test_fid)
                    rule_counts.append(n_rules)
                    atom_counts.append(n_atoms)
                    fit_times.append(elapsed)
                
                mean_train_fid = float(np.mean(train_fidelities))
                mean_test_fid = float(np.mean(test_fidelities))
                std_test_fid = float(np.std(test_fidelities))
                mean_rules = float(np.mean(rule_counts))
                mean_atoms = float(np.mean(atom_counts))
                mean_time = float(np.mean(fit_times))
                
                print(f"[{algo_name.upper():>6}] [{backend:>10}]: Test Fid = {mean_test_fid*100:6.2f}% +/- {std_test_fid*100:4.2f}% | Rules = {mean_rules:4.1f} | Atoms = {mean_atoms:4.1f} | Time = {mean_time:5.3f}s")
                
                results.append({
                    "dataset": ds_name,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "n_clusters": k,
                    "cluster_algorithm": algo_name,
                    "backend": backend,
                    "train_fidelity": mean_train_fid,
                    "test_fidelity": mean_test_fid,
                    "test_fidelity_std": std_test_fid,
                    "mean_rules": mean_rules,
                    "mean_atoms": mean_atoms,
                    "mean_time": mean_time,
                })

    df = pd.DataFrame(results)
    return df


def generate_plots_and_summary(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw CSV
    df.to_csv(output_dir / "cluster_explanation_benchmark_results.csv", index=False)
    
    # Create Pareto Plot: Test Fidelity vs Atom Count
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    colors = {"cart": "#2b5c8f", "rulegp": "#d95f02", "rulensga2": "#7570b3", "ruleplcs": "#1b9e77"}
    markers = {"cart": "s", "rulegp": "o", "rulensga2": "^", "ruleplcs": "D"}
    names = {"cart": "CART (Tree SRS)", "rulegp": "ruleGP", "rulensga2": "ruleNSGA-II", "ruleplcs": "rulePLCS"}
    
    # Aggregate over all datasets and cluster algorithms
    agg_df = df.groupby("backend").agg({
        "test_fidelity": "mean",
        "mean_atoms": "mean",
        "mean_rules": "mean",
        "mean_time": "mean"
    }).reset_index()
    
    for _, row in agg_df.iterrows():
        b = row["backend"]
        ax.scatter(
            row["mean_atoms"],
            row["test_fidelity"] * 100,
            color=colors.get(b, "black"),
            marker=markers.get(b, "o"),
            s=180,
            label=f"{names.get(b, b)} (Fid={row['test_fidelity']*100:.1f}%, Atoms={row['mean_atoms']:.1f})",
            zorder=4,
            edgecolor="black",
            linewidth=1.2
        )
    
    ax.set_xlabel("Average Model Complexity (Total Atom Count)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Generalization Fidelity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Pareto Trade-off: Cluster Explanation Fidelity vs. Complexity", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    ax.set_ylim(70, 102)
    plt.tight_layout()
    fig.savefig(output_dir / "cluster_fidelity_vs_complexity.pdf")
    fig.savefig(output_dir / "cluster_fidelity_vs_complexity.png")
    plt.close()
    
    # Generate 2D Cluster Explanation Decision Boundaries Plot
    generate_2d_explanation_plot(output_dir)
    
    # Generate LaTeX Summary Table
    latex_table = generate_latex_table(df)
    with open(output_dir / "cluster_benchmark_table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\n[INFO] Benchmark artifacts saved to {output_dir}")


def generate_2d_explanation_plot(output_dir: Path):
    """Plot 2D clusters with overlaid decision boundaries from ScoredRuleSet explainers."""
    X, y = make_blobs(n_samples=250, n_features=2, centers=3, cluster_std=0.8, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    backends = [("cart", "CART (Tree SRS)"), ("rulegp", "ruleGP (Scored Rules)"), ("rulensga2", "ruleNSGA-II (Compact Rules)")]
    
    x_min, x_max = X_scaled[:, 0].min() - 0.8, X_scaled[:, 0].max() + 0.8
    y_min, y_max = X_scaled[:, 1].min() - 0.8, X_scaled[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    cluster_point_colors = ["#2b5c8f", "#d95f02", "#1b9e77"]
    cluster_bg_colors = ["#deebf7", "#fee6ce", "#e5f5e0"]
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(cluster_bg_colors)
    
    for idx, (ax, (b, title)) in enumerate(zip(axes, backends)):
        clf = ScoredRuleSetClassifier(backend=b, random_state=42)
        clf.fit(X_scaled, cluster_labels)
        preds = clf.predict(grid_points).reshape(xx.shape)
        
        ax.contourf(xx, yy, preds, alpha=0.55, cmap=cmap_bg, zorder=1)
        ax.contour(xx, yy, preds, colors=["#404040"], linewidths=1.3, linestyles="-", alpha=0.85, zorder=2)
        for c_idx in range(3):
            mask = (cluster_labels == c_idx)
            ax.scatter(
                X_scaled[mask, 0], X_scaled[mask, 1],
                color=cluster_point_colors[c_idx],
                edgecolor="white",
                linewidth=0.8,
                s=48,
                alpha=0.95,
                label=f"Cluster {c_idx+1}",
                zorder=3
            )
        
        ruleset = clf.to_ruleset()
        n_rules = len(ruleset.rules)
        n_atoms = sum(len(r.atoms) for r in ruleset.rules)
        fid = np.mean(clf.predict(X_scaled) == cluster_labels) * 100
        
        ax.set_title(f"{title}\nFidelity = {fid:.1f}% | Rules = {n_rules} | Atoms = {n_atoms}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Latent Feature 1", fontsize=10.5, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Latent Feature 2", fontsize=10.5, fontweight="bold")
            ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.92, facecolor="white", edgecolor="#cccccc")
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", alpha=0.4, color="#d0d0d0", zorder=0)
    
    plt.tight_layout()
    fig.savefig(output_dir / "cluster_2d_explanation_boundaries.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "cluster_2d_explanation_boundaries.png", bbox_inches="tight", dpi=300)
    plt.close()


def generate_latex_table(df: pd.DataFrame) -> str:
    kmeans_df = df[df["cluster_algorithm"] == "kmeans"]
    piv_fid = kmeans_df.pivot(index="dataset", columns="backend", values="test_fidelity") * 100
    piv_std = kmeans_df.pivot(index="dataset", columns="backend", values="test_fidelity_std") * 100
    piv_atoms = kmeans_df.pivot(index="dataset", columns="backend", values="mean_atoms")
    
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cluster explanation performance across benchmark datasets under $k$-means clustering (5-fold cross-validation). Values report Generalization Fidelity (\%) $\pm$ standard deviation and total atom count in parentheses.}",
        r"\label{tab:cluster_explanation_benchmark}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{CART (Tree SRS)} & \textbf{ruleGP} & \textbf{ruleNSGA-II} & \textbf{rulePLCS} \\",
        r"\midrule"
    ]
    
    for ds in piv_fid.index:
        row_str = f"{ds}"
        for b in ["cart", "rulegp", "rulensga2", "ruleplcs"]:
            fid = piv_fid.loc[ds, b]
            std = piv_std.loc[ds, b]
            atoms = piv_atoms.loc[ds, b]
            row_str += f" & {fid:.1f} $\\pm$ {std:.1f} ({atoms:.1f})"
        row_str += r" \\"
        lines.append(row_str)
        
    lines.extend([
        r"\midrule",
        r"\textbf{Average} & " + " & ".join([
            f"{kmeans_df[kmeans_df['backend']==b]['test_fidelity'].mean()*100:.1f}\\% ({kmeans_df[kmeans_df['backend']==b]['mean_atoms'].mean():.1f})"
            for b in ["cart", "rulegp", "rulensga2", "ruleplcs"]
        ]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "results_cluster"
    df_results = run_benchmark(n_splits=5, random_state=42)
    generate_plots_and_summary(df_results, out_dir)
