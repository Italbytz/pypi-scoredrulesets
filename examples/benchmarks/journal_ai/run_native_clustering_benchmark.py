"""Native Interpretable Clustering Benchmark.

Evaluates RuleEvoCluster against KMeans and GMM on clustering datasets
using Silhouette Score and Model Complexity (Atom count).
"""

import sys
import unittest.mock
from pathlib import Path

# Mock catgen so scoredrulesets.benchmarking doesn't fail
sys.modules['catgen'] = unittest.mock.MagicMock()
sys.modules['catgen.datasets'] = unittest.mock.MagicMock()

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scoredrulesets.estimators.evocluster import RuleEvoCluster
from run_cluster_surrogate_benchmark import get_all_datasets

def run_native_benchmark():
    datasets = get_all_datasets()
    results = []

    for X, k, ds_name, feat_names in datasets:
        print(f"\n==================================================================")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]}, K={k})")
        print(f"==================================================================")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        t0 = time.time()
        preds_km = km.fit_predict(X_scaled)
        t_km = time.time() - t0
        sil_km = silhouette_score(X_scaled, preds_km)
        
        # 2. GMM
        gmm = GaussianMixture(n_components=k, random_state=42)
        t0 = time.time()
        preds_gmm = gmm.fit_predict(X_scaled)
        t_gmm = time.time() - t0
        sil_gmm = silhouette_score(X_scaled, preds_gmm)
        
        # 3. RuleEvoCluster
        evo = RuleEvoCluster(n_clusters=k, random_state=42, max_generations=50, population_size=50)
        t0 = time.time()
        evo.fit(X_scaled)
        t_evo = time.time() - t0
        
        preds_evo = evo.labels_
        try:
            sil_evo = silhouette_score(X_scaled, preds_evo)
        except ValueError:
            sil_evo = -1.0
            
        atoms_evo = evo.ruleset_.metadata.get("rules_count", 0) # Just to be safe, but actually we want sum of atoms
        atoms_evo = sum(len(r.atoms) for r in evo.ruleset_.rules)
        
        print(f"KMeans : Sil = {sil_km:.3f} | Time = {t_km:.3f}s")
        print(f"GMM    : Sil = {sil_gmm:.3f} | Time = {t_gmm:.3f}s")
        print(f"EvoClus: Sil = {sil_evo:.3f} | Atoms = {atoms_evo} | Time = {t_evo:.3f}s")
        
        results.append({
            "Dataset": ds_name,
            "KMeans Silhouette": sil_km,
            "GMM Silhouette": sil_gmm,
            "RuleEvoCluster Silhouette": sil_evo,
            "RuleEvoCluster Atoms": atoms_evo,
        })
        
    df = pd.DataFrame(results)
    
    # Generate CSV instead of LaTeX to avoid jinja2 dependency
    out_path = _SCRIPT_DIR / "results_cluster" / "native_clustering_benchmark.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
        
    print(f"Saved CSV to {out_path}")

if __name__ == "__main__":
    run_native_benchmark()
