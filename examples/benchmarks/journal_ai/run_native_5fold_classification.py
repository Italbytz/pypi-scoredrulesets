"""Protocol-consistency check: native backends under a unified 5-fold harness.

The classification results in Table~\\ref{tab:classification_benchmark} were
imported from prior work that used 10-fold cross-validation, whereas every other
experiment in this paper uses 5-fold CV. To quantify whether this protocol
difference materially affects our own contributions, we re-run the three native
scored-rule-set backends (ruleGP, ruleNSGA-II, rulePLCS) on the *same* datasets
under the identical 5-fold StratifiedKFold protocol (seed 42, per-fold z-score
standardization) used elsewhere.

Datasets are loaded from the package's benchmark registry so that the exact same
instances as the original study are used.
"""

import sys
import unittest.mock
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scoredrulesets.benchmarking.datasets import load_dataset_registry
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier

RANDOM_STATE = 42
N_SPLITS = 5

# Registry keys used by the original paper benchmark, with display names.
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

BACKENDS = ["rulegp", "rulensga2", "ruleplcs"]


def evaluate(X, y, backend):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    f1s, atoms = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        imputer = SimpleImputer(strategy="mean").fit(X[tr])
        X_tr_i, X_te_i = imputer.transform(X[tr]), imputer.transform(X[te])
        scaler = StandardScaler().fit(X_tr_i)
        X_tr, X_te = scaler.transform(X_tr_i), scaler.transform(X_te_i)
        clf = ScoredRuleSetClassifier(
            backend=backend,
            backend_params={"random_state": RANDOM_STATE + fold},
        )
        clf.fit(X_tr, y[tr])
        rs = clf.to_ruleset()
        atoms.append(sum(len(r.atoms) for r in rs.rules))
        f1s.append(f1_score(y[te], clf.predict(X_te), average="macro"))
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(atoms))


def run():
    registry = load_dataset_registry(include_online_uci=True, include_synthetic=True)
    records = []
    for key, name in DATASETS.items():
        if key not in registry:
            print(f"[skip] {name} ({key}) not available in registry")
            continue
        bundle = registry[key]
        X = np.asarray(bundle.X, dtype=float)
        y = np.asarray(bundle.y)
        print(f"\n{'=' * 60}\n{name}  (N={X.shape[0]}, D={X.shape[1]})\n{'=' * 60}")
        for backend in BACKENDS:
            f1, f1s, atom = evaluate(X, y, backend)
            print(f"  {backend:<10}: F1={f1:.3f}±{f1s:.3f}  atoms={atom:.1f}")
            records.append({"dataset": name, "backend": backend,
                            "macro_f1": f1, "macro_f1_std": f1s, "mean_atoms": atom})

    df = pd.DataFrame(records)
    out = _SCRIPT_DIR / "results_classification" / "native_5fold_classification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved CSV to {out}")


if __name__ == "__main__":
    run()
