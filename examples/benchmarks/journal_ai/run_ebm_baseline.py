"""Evaluate Explainable Boosting Machine (EBM) baseline across classification and regression benchmarks.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_friedman1,
    make_friedman2,
    make_friedman3,
)
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def run_ebm_regression():
    cal = fetch_california_housing()
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(cal.data.shape[0], size=800, replace=False)
    X_cal, y_cal = cal.data[sub_idx], cal.target[sub_idx]
    
    X_diab, y_diab = load_diabetes(return_X_y=True)
    X_f1, y_f1 = make_friedman1(n_samples=400, n_features=10, noise=1.0, random_state=42)
    X_f2, y_f2 = make_friedman2(n_samples=400, noise=1.0, random_state=42)
    X_f3, y_f3 = make_friedman3(n_samples=400, noise=0.1, random_state=42)
    
    datasets = [
        ("California Housing", X_cal, y_cal),
        ("Diabetes", X_diab, y_diab),
        ("Friedman #1", X_f1, y_f1),
        ("Friedman #2", X_f2, y_f2),
        ("Friedman #3", X_f3, y_f3),
    ]
    
    results = []
    for name, X, y in datasets:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_list = []
        for train_idx, test_idx in kf.split(X, y):
            scaler = StandardScaler().fit(X[train_idx])
            X_tr, X_te = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
            y_tr, y_te = y[train_idx], y[test_idx]
            
            ebm = ExplainableBoostingRegressor(random_state=42)
            ebm.fit(X_tr, y_tr)
            preds = ebm.predict(X_te)
            r2_list.append(r2_score(y_te, preds))
        
        results.append({
            "dataset": name,
            "ebm_r2_mean": float(np.mean(r2_list)),
            "ebm_r2_std": float(np.std(r2_list)),
        })
        print(f"EBM Regression [{name}]: R2 = {np.mean(r2_list):.3f} +/- {np.std(r2_list):.3f}")
    
    return pd.DataFrame(results)


def run_ebm_classification():
    from scoredrulesets.benchmarking.datasets import load_dataset_registry
    
    datasets_keys = [
        ("uci_breast_cancer_wisconsin_diagnostic", "Breast Cancer"),
        ("uci_wine", "Wine"),
        ("uci_car_evaluation", "Car Evaluation"),
        ("uci_heart_disease", "Heart Disease"),
        ("synth_dnf_3x2", "DNF (3x2)"),
        ("synth_xor_3bit", "XOR (3-bit)"),
        ("synth_checkerboard_4x4", "Checkerboard"),
        ("synth_monk3", "Monk-3"),
        ("synth_imbalanced_10pct", "Imbalanced"),
    ]
    registry = load_dataset_registry()
    results = []
    
    for key, name in datasets_keys:
        ds = registry.get(key)
        if ds is None:
            continue
        X, y = ds.X, ds.y
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y = np.asarray(y).ravel()
        if key == "uci_heart_disease":
            y = (y > 0).astype(int)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_list = []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler().fit(X[train_idx])
            X_tr, X_te = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
            y_tr, y_te = y[train_idx], y[test_idx]
            
            ebm = ExplainableBoostingClassifier(random_state=42)
            ebm.fit(X_tr, y_tr)
            preds = ebm.predict(X_te)
            f1_list.append(f1_score(y_te, preds, average="macro"))
            
        results.append({
            "dataset": name,
            "ebm_f1_mean": float(np.mean(f1_list)),
            "ebm_f1_std": float(np.std(f1_list)),
        })
        print(f"EBM Classification [{name}]: Macro-F1 = {np.mean(f1_list):.3f} +/- {np.std(f1_list):.3f}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    df_reg = run_ebm_regression()
    script_dir = Path(__file__).resolve().parent
    (script_dir / "results_regression").mkdir(parents=True, exist_ok=True)
    df_reg.to_csv(script_dir / "results_regression/ebm_regression_results.csv", index=False)
    df_cls = run_ebm_classification()
    (script_dir / "results_classification").mkdir(parents=True, exist_ok=True)
    df_cls.to_csv(script_dir / "results_classification/ebm_classification_results.csv", index=False)
