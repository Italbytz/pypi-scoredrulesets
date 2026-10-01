"""RuleFit Regression Benchmark.

Evaluates RuleFit (via imodels) on tabular regression datasets
for comparison with RuleGP and other SRS implementations.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

# Mock catgen if not installed in current environment
sys.modules['catgen'] = unittest.mock.MagicMock()
sys.modules['catgen.datasets'] = unittest.mock.MagicMock()

import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing, make_friedman1, make_friedman2, make_friedman3
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from imodels import RuleFitRegressor

def get_rulefit_datasets():
    datasets = []
    
    # 1. Friedman #1
    X_f1, y_f1 = make_friedman1(n_samples=400, n_features=10, noise=1.0, random_state=42)
    datasets.append((X_f1, y_f1, "Friedman #1"))

    # 2. Friedman #2
    X_f2, y_f2 = make_friedman2(n_samples=400, noise=1.0, random_state=42)
    datasets.append((X_f2, y_f2, "Friedman #2"))

    # 3. Friedman #3
    X_f3, y_f3 = make_friedman3(n_samples=400, noise=1.0, random_state=42)
    datasets.append((X_f3, y_f3, "Friedman #3"))

    # 4. Diabetes
    X_diab, y_diab = load_diabetes(return_X_y=True)
    datasets.append((X_diab, y_diab, "Diabetes"))

    # 5. California Housing (Subsampled)
    cal = fetch_california_housing()
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(cal.data.shape[0], size=800, replace=False)
    datasets.append((cal.data[sub_idx], cal.target[sub_idx], "California Housing (N=800)"))
    
    return datasets

def run_rulefit_benchmark():
    datasets = get_rulefit_datasets()
    records = []

    out_dir = Path(__file__).resolve().parent / "results_regression"
    out_dir.mkdir(exist_ok=True)

    for X, y, ds_name in datasets:
        print(f"\n==================================================================")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]})")
        print(f"==================================================================")
            
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = []
        rmse_scores = []
        atom_counts = []
        
        t0 = time.time()
        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # RuleFit using imodels
            model = RuleFitRegressor(random_state=42)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = root_mean_squared_error(y_test, preds)
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            
            # Estimate complexity (atom count)
            n_atoms = 0
            if hasattr(model, "rules_"):
                for r in model.rules_:
                    rule_str = str(r)
                    if " <= " in rule_str or " > " in rule_str:
                        n_atoms += len(rule_str.split(' and '))
            atom_counts.append(n_atoms)

        elapsed = time.time() - t0
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_rmse = np.mean(rmse_scores)
        mean_atoms = np.mean(atom_counts)
        
        print(f"RuleFit: R2 = {mean_r2:6.3f} +/- {std_r2:5.3f} | RMSE = {mean_rmse:7.3f} | Atoms = {mean_atoms:6.1f} | Time = {elapsed:.1f}s")
        
        records.append({
            "Dataset": ds_name,
            "Model": "RuleFit",
            "R2_Mean": mean_r2,
            "R2_Std": std_r2,
            "RMSE": mean_rmse,
            "Complexity (Atoms)": mean_atoms,
            "Runtime_s": elapsed,
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "rulefit_benchmark_results.csv", index=False)
    print(f"\nRuleFit benchmark results saved to {out_dir / 'rulefit_benchmark_results.csv'}")

if __name__ == '__main__':
    run_rulefit_benchmark()
