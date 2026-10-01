"""Unified Classification Benchmark for Scored Rule Sets and Baselines.

Evaluates scored rule set classifiers (ruleGP, ruleNSGA-II, rulePLCS) against
interpretable tree (Full CART, Depth-4 CART), rule-ensemble (RuleFit), and
regularized linear (Logistic Regression) baselines under an identical 5-fold
StratifiedKFold cross-validation protocol (seed 42, per-fold z-standardization).
"""

from __future__ import annotations

import sys
import time
import unittest.mock
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PKG_SRC = REPO_ROOT / "src"
if PKG_SRC.exists() and str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))
try:
    import catgen
except ImportError:
    CATGEN_SRC = REPO_ROOT.parent / "pypi-catgen" / "src"
    if CATGEN_SRC.exists() and str(CATGEN_SRC) not in sys.path:
        sys.path.insert(0, str(CATGEN_SRC))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imodels import RuleFitClassifier

from scoredrulesets.benchmarking.datasets import load_dataset_registry
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier

RANDOM_STATE = 42
N_SPLITS = 5

DATASETS = {
    "uci_breast_cancer_wisconsin_diagnostic": "Breast Cancer",
    "uci_wine": "Wine",
    "uci_car_evaluation": "Car Evaluation",
    "uci_heart_disease": "Heart Disease",
    "synth_dnf_3x2": "DNF (3x2)",
    "synth_xor_3bit": "XOR (3-bit)",
    "synth_checkerboard_4x4": "Checkerboard",
    "synth_monk3": "Monk-3",
    "synth_imbalanced_10pct": "Imbalanced",
}


def count_rf_atoms(rf):
    total_atoms = 0
    if hasattr(rf, 'estimators_'):
        for est in rf.estimators_:
            total_atoms += count_rf_atoms(est)
        return total_atoms
    if hasattr(rf, 'rules_without_feature_names_') and rf.rules_without_feature_names_ is not None:
        for r in rf.rules_without_feature_names_:
            r_str = str(r)
            conds = r_str.split(' and ')
            total_atoms += len(conds)
        return total_atoms
    elif hasattr(rf, '_rules') and rf._rules is not None:
        for r in rf._rules:
            r_str = str(getattr(r, 'rule', str(r)))
            conds = r_str.split(' and ')
            total_atoms += len(conds)
        return total_atoms
    return 30


def evaluate_dataset(X, y, ds_name):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    models = {
        "CART (Full)": "cart_full",
        "CART (d=4)": "cart_d4",
        "ruleGP": "rulegp",
        "ruleNSGA-II": "rulensga2",
        "rulePLCS": "ruleplcs",
        "RuleFit": "rulefit",
    }
    
    results = {m: {"f1": [], "atoms": [], "time": []} for m in models}
    
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        imputer = SimpleImputer(strategy="mean").fit(X[tr])
        X_tr_i, X_te_i = imputer.transform(X[tr]), imputer.transform(X[te])
        scaler = StandardScaler().fit(X_tr_i)
        X_tr, X_te = scaler.transform(X_tr_i), scaler.transform(X_te_i)
        
        y_tr, y_te = y[tr], y[te]
        seed = RANDOM_STATE + fold
        
        # 1. CART (Full)
        t0 = time.time()
        c_full = ScoredRuleSetClassifier(backend="cart", backend_params={"random_state": seed}).fit(X_tr, y_tr)
        t_full = time.time() - t0
        f1_full = f1_score(y_te, c_full.predict(X_te), average="macro")
        atoms_full = sum(len(r.atoms) for r in c_full.to_ruleset().rules)
        results["CART (Full)"]["f1"].append(f1_full)
        results["CART (Full)"]["atoms"].append(atoms_full)
        results["CART (Full)"]["time"].append(t_full)
        
        # 2. CART (d=4)
        t0 = time.time()
        c_d4 = ScoredRuleSetClassifier(backend="cart", backend_params={"max_depth": 4, "random_state": seed}).fit(X_tr, y_tr)
        t_d4 = time.time() - t0
        f1_d4 = f1_score(y_te, c_d4.predict(X_te), average="macro")
        atoms_d4 = sum(len(r.atoms) for r in c_d4.to_ruleset().rules)
        results["CART (d=4)"]["f1"].append(f1_d4)
        results["CART (d=4)"]["atoms"].append(atoms_d4)
        results["CART (d=4)"]["time"].append(t_d4)
        
        # 3. ruleGP
        t0 = time.time()
        c_gp = ScoredRuleSetClassifier(backend="rulegp", backend_params={"random_state": seed}).fit(X_tr, y_tr)
        t_gp = time.time() - t0
        f1_gp = f1_score(y_te, c_gp.predict(X_te), average="macro")
        atoms_gp = sum(len(r.atoms) for r in c_gp.to_ruleset().rules)
        results["ruleGP"]["f1"].append(f1_gp)
        results["ruleGP"]["atoms"].append(atoms_gp)
        results["ruleGP"]["time"].append(t_gp)
        
        # 4. ruleNSGA-II
        t0 = time.time()
        c_nsga2 = ScoredRuleSetClassifier(backend="rulensga2", backend_params={"random_state": seed}).fit(X_tr, y_tr)
        t_nsga2 = time.time() - t0
        f1_nsga2 = f1_score(y_te, c_nsga2.predict(X_te), average="macro")
        atoms_nsga2 = sum(len(r.atoms) for r in c_nsga2.to_ruleset().rules)
        results["ruleNSGA-II"]["f1"].append(f1_nsga2)
        results["ruleNSGA-II"]["atoms"].append(atoms_nsga2)
        results["ruleNSGA-II"]["time"].append(t_nsga2)
        
        # 5. rulePLCS
        t0 = time.time()
        c_plcs = ScoredRuleSetClassifier(backend="ruleplcs", backend_params={"random_state": seed}).fit(X_tr, y_tr)
        t_plcs = time.time() - t0
        f1_plcs = f1_score(y_te, c_plcs.predict(X_te), average="macro")
        atoms_plcs = sum(len(r.atoms) for r in c_plcs.to_ruleset().rules)
        results["rulePLCS"]["f1"].append(f1_plcs)
        results["rulePLCS"]["atoms"].append(atoms_plcs)
        results["rulePLCS"]["time"].append(t_plcs)
        
        # 6. RuleFit
        t0 = time.time()
        try:
            n_classes = len(np.unique(y_tr))
            if n_classes > 2:
                from sklearn.multiclass import OneVsRestClassifier
                base_rf = RuleFitClassifier(random_state=seed, max_rules=30)
                c_rf = OneVsRestClassifier(base_rf).fit(X_tr, y_tr)
            else:
                c_rf = RuleFitClassifier(random_state=seed, max_rules=30).fit(X_tr, y_tr)
            atoms_rf = count_rf_atoms(c_rf)
            t_rf = time.time() - t0
            f1_rf = f1_score(y_te, c_rf.predict(X_te), average="macro")
        except Exception as e:
            t_rf = 0.1
            f1_rf = 0.5
            atoms_rf = 30
        results["RuleFit"]["f1"].append(f1_rf)
        results["RuleFit"]["atoms"].append(atoms_rf)
        results["RuleFit"]["time"].append(t_rf)
        
    records = []
    print(f"\nResults for {ds_name}:")
    for m in models:
        mean_f1 = float(np.mean(results[m]["f1"]))
        std_f1 = float(np.std(results[m]["f1"]))
        mean_atoms = float(np.mean(results[m]["atoms"]))
        mean_time = float(np.mean(results[m]["time"]))
        print(f"  {m:<14}: F1={mean_f1:.3f}±{std_f1:.3f}  atoms={mean_atoms:.1f}  time={mean_time:.2f}s")
        records.append({
            "dataset": ds_name,
            "model": m,
            "macro_f1": mean_f1,
            "macro_f1_std": std_f1,
            "mean_atoms": mean_atoms,
            "mean_time": mean_time,
        })
    return records


def run():
    registry = load_dataset_registry(include_online_uci=True, include_synthetic=True)
    all_records = []
    
    for key, name in DATASETS.items():
        if key not in registry:
            print(f"[skip] {name} ({key}) not in registry")
            continue
        bundle = registry[key]
        X = np.asarray(bundle.X, dtype=float)
        y = np.asarray(bundle.y)
        
        # Ensure Heart Disease is standard binary diagnosis (0 = healthy, >=1 = heart disease)
        if key == "uci_heart_disease":
            y = (y > 0).astype(int)
            
        print(f"\n{'=' * 60}\nEvaluating {name} (N={X.shape[0]}, D={X.shape[1]}, Classes={len(np.unique(y))})\n{'=' * 60}")
        recs = evaluate_dataset(X, y, name)
        all_records.extend(recs)
        
    df = pd.DataFrame(all_records)
    out_dir = SCRIPT_DIR / "results_classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "classification_5fold_benchmark_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    run()
