"""Takagi-Sugeno (Piecewise Linear) Regression Benchmark.

Evaluates RuleGP and RuleNSGA-II with Takagi-Sugeno linear consequences
against constant-weight rule sets and standard baselines on tabular benchmarks.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

# Mock catgen if not installed in current environment
sys.modules['catgen'] = unittest.mock.MagicMock()
sys.modules['catgen.datasets'] = unittest.mock.MagicMock()

# Ensure local scoredrulesets package is available
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_friedman1, make_friedman2, make_friedman3
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from scoredrulesets.estimators.rulegp_regressor import RuleGPRegressor
from scoredrulesets.estimators.rulensga2_regressor import RuleNSGA2Regressor


def get_bench_datasets():
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

    return datasets


def evaluate_model_cv(model_fn, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_list = []
    rmse_list = []
    atom_list = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2_list.append(r2_score(y_test, preds))
        rmse_list.append(root_mean_squared_error(y_test, preds))

        if hasattr(model, "to_ruleset"):
            rs = model.to_ruleset()
            atoms = sum(len(r.atoms) for r in rs.rules)
            atom_list.append(atoms)
        elif hasattr(model, "tree_"):
            atom_list.append(model.tree_.node_count)
        else:
            atom_list.append(X.shape[1])

    return np.mean(r2_list), np.std(r2_list), np.mean(rmse_list), np.mean(atom_list)


def run_benchmark():
    datasets = get_bench_datasets()
    records = []

    models = {
        "Ridge": lambda: Ridge(),
        "CART (d=4)": lambda: DecisionTreeRegressor(max_depth=4, random_state=42),
        "RuleGP (Constant)": lambda: RuleGPRegressor(prediction_type="constant", max_generations=40, population_size=60, random_state=42),
        "RuleGP (Takagi-Sugeno)": lambda: RuleGPRegressor(prediction_type="linear", max_generations=40, population_size=60, random_state=42),
        "RuleNSGA-II (Constant)": lambda: RuleNSGA2Regressor(prediction_type="constant", max_generations=40, population_size=60, random_state=42),
        "RuleNSGA-II (Takagi-Sugeno)": lambda: RuleNSGA2Regressor(prediction_type="linear", max_generations=40, population_size=60, random_state=42),
    }

    for X, y, ds_name in datasets:
        print(f"\n========================================================")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]})")
        print(f"========================================================")

        for m_name, m_fn in models.items():
            t0 = time.time()
            r2_m, r2_std, rmse_m, atoms_m = evaluate_model_cv(m_fn, X, y)
            elapsed = time.time() - t0
            print(f"{m_name:<28}: R2 = {r2_m:6.3f} +/- {r2_std:5.3f} | RMSE = {rmse_m:7.3f} | Atoms = {atoms_m:5.1f} ({elapsed:.1f}s)")
            records.append({
                "Dataset": ds_name,
                "Model": m_name,
                "R2_Mean": r2_m,
                "R2_Std": r2_std,
                "RMSE": rmse_m,
                "Complexity (Atoms)": atoms_m,
                "Runtime_s": elapsed,
            })

    df = pd.DataFrame(records)
    out_dir = Path(__file__).resolve().parent / "results_regression"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "takagi_sugeno_benchmark_results.csv", index=False)
    print(f"\nSaved results to {out_dir / 'takagi_sugeno_benchmark_results.csv'}")


if __name__ == "__main__":
    run_benchmark()
