"""Sensitivity / Ablation Study for the Evolutionary Search Budget.

Quantifies how the two central evolutionary hyperparameters of ``ruleGP`` ---
the number of generations and the population size --- affect the
predictive--complexity trade-off. All runs use the same 5-fold stratified
cross-validation protocol (seed 42, z-score standardization) as the main
classification benchmark.

Two one-factor-at-a-time sweeps are reported:
  (A) generations in {50, 150, 300, 500} at fixed population size 120
  (B) population size in {60, 120, 180} at fixed 300 generations
"""

import sys
import unittest.mock
from pathlib import Path

sys.modules['catgen'] = unittest.mock.MagicMock()
sys.modules['catgen.datasets'] = unittest.mock.MagicMock()

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR.parents[2] / "src"
if _PKG_SRC.exists() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier

RANDOM_STATE = 42
N_SPLITS = 5


def get_datasets():
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    return [
        (iris.data, iris.target, "Iris"),
        (wine.data, wine.target, "Wine"),
        (cancer.data, cancer.target, "Breast Cancer"),
    ]


def evaluate(X, y, backend_params):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    f1s, atoms, times = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr = StandardScaler().fit_transform(X[tr])
        X_te = StandardScaler().fit(X[tr]).transform(X[te])
        clf = ScoredRuleSetClassifier(
            backend="rulegp",
            backend_params={**backend_params, "random_state": RANDOM_STATE + fold},
        )
        t0 = time.time()
        clf.fit(X_tr, y[tr])
        times.append(time.time() - t0)
        rs = clf.to_ruleset()
        atoms.append(sum(len(r.atoms) for r in rs.rules))
        f1s.append(f1_score(y[te], clf.predict(X_te), average="macro"))
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(atoms)), float(np.mean(times))


def run():
    records = []
    gen_grid = [50, 150, 300, 500]
    pop_grid = [60, 120, 180]

    for X, y, name in get_datasets():
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

        # Sweep A: generations at fixed population 120
        for g in gen_grid:
            f1, f1s, atom, t = evaluate(
                X, y, {"max_generations": g, "population_size": 120})
            print(f"[A] gen={g:>3} pop=120 : F1={f1:.3f}±{f1s:.3f} atoms={atom:.1f} ({t:.1f}s)")
            records.append({"sweep": "generations", "dataset": name,
                            "max_generations": g, "population_size": 120,
                            "macro_f1": f1, "macro_f1_std": f1s,
                            "mean_atoms": atom, "mean_time_s": t})

        # Sweep B: population at fixed generations 300
        for p in pop_grid:
            if p == 120:
                continue  # already covered by sweep A (gen=300, pop=120)
            f1, f1s, atom, t = evaluate(
                X, y, {"max_generations": 300, "population_size": p})
            print(f"[B] gen=300 pop={p:>3} : F1={f1:.3f}±{f1s:.3f} atoms={atom:.1f} ({t:.1f}s)")
            records.append({"sweep": "population", "dataset": name,
                            "max_generations": 300, "population_size": p,
                            "macro_f1": f1, "macro_f1_std": f1s,
                            "mean_atoms": atom, "mean_time_s": t})

    df = pd.DataFrame(records)
    out = _SCRIPT_DIR / "results_classification" / "ablation_search_budget.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved CSV to {out}")


if __name__ == "__main__":
    run()
