"""Multi-seed robustness evaluation for evolutionary scored rule sets.

Evaluates ruleGP and ruleNSGA-II across multiple random seeds (42, 100, 200, 300, 400)
with 5-fold cross-validation to assess variance attributable to optimization randomness.
"""

from __future__ import annotations

import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scoredrulesets.benchmarking.datasets import load_dataset_registry
from scoredrulesets.estimators.sklearn_wrapper import (
    ScoredRuleSetClassifier,
    ScoredRuleSetRegressor,
)

SEEDS = [42, 100, 200, 300, 400]


def evaluate_classification_seeds():
    registry = load_dataset_registry(include_online_uci=True, include_synthetic=True)
    datasets = []
    for key, name in [
        ("uci_breast_cancer_wisconsin_diagnostic", "Breast Cancer"),
        ("uci_wine", "Wine"),
    ]:
        bundle = registry[key]
        datasets.append((name, (np.asarray(bundle.X, dtype=float), np.asarray(bundle.y))))
    
    records = []
    for ds_name, (X, y) in datasets:
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for model_name, backend in [("ruleGP", "rulegp"), ("ruleNSGA-II", "rulensga2")]:
                f1_list, atom_list = [], []
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    # Mirror run_classification_benchmark.py protocol exactly:
                    # mean-imputation and per-fold seed derivation (base seed + fold).
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy="mean").fit(X[train_idx])
                    X_tr_i, X_te_i = imputer.transform(X[train_idx]), imputer.transform(X[test_idx])
                    scaler = StandardScaler().fit(X_tr_i)
                    X_tr = scaler.transform(X_tr_i)
                    X_te = scaler.transform(X_te_i)
                    y_tr, y_te = y[train_idx], y[test_idx]
                    fold_seed = seed + fold
                    
                    clf = ScoredRuleSetClassifier(backend=backend, backend_params={"random_state": fold_seed})
                    clf.fit(X_tr, y_tr)
                    preds = clf.predict(X_te)
                    f1 = f1_score(y_te, preds, average="macro")
                    atoms = sum(len(r.atoms) for r in clf.to_ruleset().rules)
                    f1_list.append(f1)
                    atom_list.append(atoms)
                
                records.append({
                    "task": "Classification",
                    "dataset": ds_name,
                    "model": model_name,
                    "seed": seed,
                    "macro_f1": np.mean(f1_list),
                    "atoms": np.mean(atom_list),
                })
    return pd.DataFrame(records)


def evaluate_regression_seeds():
    X_diab, y_diab = load_diabetes(return_X_y=True)
    X_f1, y_f1 = make_friedman1(n_samples=400, n_features=10, noise=1.0, random_state=42)
    
    datasets = [
        ("Diabetes", (X_diab, y_diab)),
        ("Friedman #1", (X_f1, y_f1)),
    ]
    
    records = []
    for ds_name, (X, y) in datasets:
        for seed in SEEDS:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for model_name, backend in [("ruleGP", "rulegp"), ("ruleNSGA-II", "rulensga2")]:
                r2_list, atom_list = [], []
                for train_idx, test_idx in kf.split(X, y):
                    scaler = StandardScaler().fit(X[train_idx])
                    X_tr = scaler.transform(X[train_idx])
                    X_te = scaler.transform(X[test_idx])
                    y_tr, y_te = y[train_idx], y[test_idx]
                    
                    reg = ScoredRuleSetRegressor(backend=backend, random_state=seed)
                    reg.fit(X_tr, y_tr)
                    preds = reg.predict(X_te)
                    r2 = r2_score(y_te, preds)
                    ruleset = reg.to_ruleset()
                    atoms = sum(len(r.atoms) for r in ruleset.rules)
                    r2_list.append(r2)
                    atom_list.append(atoms)
                
                records.append({
                    "task": "Regression",
                    "dataset": ds_name,
                    "model": model_name,
                    "seed": seed,
                    "r2": np.mean(r2_list),
                    "atoms": np.mean(atom_list),
                })
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Running classification multi-seed evaluation...")
    df_cls = evaluate_classification_seeds()
    print("\nClassification Seed Summary:")
    cls_summary = df_cls.groupby(["dataset", "model"]).agg(
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
        atoms_mean=("atoms", "mean"),
        atoms_std=("atoms", "std"),
    )
    print(cls_summary)
    
    print("\nRunning regression multi-seed evaluation...")
    df_reg = evaluate_regression_seeds()
    print("\nRegression Seed Summary:")
    reg_summary = df_reg.groupby(["dataset", "model"]).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        atoms_mean=("atoms", "mean"),
        atoms_std=("atoms", "std"),
    )
    print(reg_summary)
    
    _SCRIPT_DIR = Path(__file__).resolve().parent
    (_SCRIPT_DIR / "results_classification").mkdir(parents=True, exist_ok=True)
    (_SCRIPT_DIR / "results_regression").mkdir(parents=True, exist_ok=True)
    df_cls.to_csv(_SCRIPT_DIR / "results_classification/multiseed_robustness_cls.csv", index=False)
    df_reg.to_csv(_SCRIPT_DIR / "results_regression/multiseed_robustness_reg.csv", index=False)
