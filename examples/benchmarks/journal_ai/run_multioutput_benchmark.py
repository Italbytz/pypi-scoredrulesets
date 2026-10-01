"""Multi-Target / Multi-Output Regression Benchmark.

Evaluates RuleGPRegressor with multivariate vector scores against Ridge
and DecisionTree on multi-output regression benchmarks.
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
from sklearn.datasets import make_regression, load_linnerud
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from scoredrulesets.estimators.rulegp_regressor import RuleGPRegressor


def get_multioutput_datasets():
    datasets = []

    # 1. Synthetic Multi-Output (3 Correlated Targets)
    X_syn, y_syn = make_regression(n_samples=300, n_features=10, n_informative=5, n_targets=3, noise=1.0, random_state=42)
    datasets.append((X_syn, y_syn, "Synthetic Multi-Output (D=10, Targets=3)"))

    # 2. Linnerud Dataset (Physical Exercise & Physiological, 3 Targets)
    lin = load_linnerud()
    datasets.append((lin.data, lin.target, "Linnerud (Physical Exercise, Targets=3)"))

    return datasets


def evaluate_multioutput_cv(model_fn, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_raw_list = []
    r2_mean_list = []
    rmse_list = []
    atom_list = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2_raw = r2_score(y_test, preds, multioutput="raw_values")
        r2_raw_list.append(r2_raw)
        r2_mean_list.append(np.mean(r2_raw))
        rmse_list.append(root_mean_squared_error(y_test, preds))

        if hasattr(model, "to_ruleset"):
            rs = model.to_ruleset()
            atoms = sum(len(r.atoms) for r in rs.rules)
            atom_list.append(atoms)
        elif hasattr(model, "tree_"):
            atom_list.append(model.tree_.node_count)
        else:
            atom_list.append(X.shape[1] * y.shape[1])

    mean_r2 = float(np.mean(r2_mean_list))
    std_r2 = float(np.std(r2_mean_list))
    per_target_r2 = np.mean(np.array(r2_raw_list), axis=0)
    mean_rmse = float(np.mean(rmse_list))
    mean_atoms = float(np.mean(atom_list))

    return mean_r2, std_r2, per_target_r2, mean_rmse, mean_atoms


def run_benchmark():
    datasets = get_multioutput_datasets()
    records = []

    models = {
        "Ridge": lambda: Ridge(),
        "CART (d=4)": lambda: DecisionTreeRegressor(max_depth=4, random_state=42),
        "RuleGP (Multi-Target)": lambda: RuleGPRegressor(population_size=60, max_generations=50, random_state=42),
    }

    out_dir = Path(__file__).resolve().parent / "results_regression"
    out_dir.mkdir(exist_ok=True)

    for X, y, ds_name in datasets:
        print(f"\n==================================================================")
        print(f"Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]}, Targets={y.shape[1]})")
        print(f"==================================================================")

        for m_name, m_fn in models.items():
            t0 = time.time()
            mean_r2, std_r2, per_target_r2, rmse, atoms = evaluate_multioutput_cv(m_fn, X, y)
            elapsed = time.time() - t0
            targets_str = ", ".join([f"T{i+1}={v:.3f}" for i, v in enumerate(per_target_r2)])
            print(f"{m_name:<24}: Mean R2 = {mean_r2:6.3f} +/- {std_r2:5.3f} | [{targets_str}] | RMSE = {rmse:7.3f} | Atoms = {atoms:5.1f} ({elapsed:.1f}s)")
            
            records.append({
                "Dataset": ds_name,
                "Model": m_name,
                "Mean_R2": mean_r2,
                "Std_R2": std_r2,
                "Per_Target_R2": str(np.round(per_target_r2, 3).tolist()),
                "RMSE": rmse,
                "Complexity (Atoms)": atoms,
                "Runtime_s": elapsed,
            })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "multioutput_benchmark_results.csv", index=False)
    print(f"\nMulti-Target benchmark results saved to {out_dir / 'multioutput_benchmark_results.csv'}")


if __name__ == "__main__":
    run_benchmark()
