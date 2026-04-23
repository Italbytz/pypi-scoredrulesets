#!/usr/bin/env python3
"""Quick LogicGP run on scrime simulated SNP data.

Demonstrates multiclass classification with LogicGPClassifier on the
scrime simulation dataset (50 SNPs, 3 phenotype classes).

Equivalent to quick_scrime_run.R in the paper-code companion repo.

Usage
-----
Run from the pypi-scoredrulesets repository root::

    python examples/snp/quick_scrime_run.py

The script uses a fixed random seed so results are reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from scoredrulesets import LogicGPClassifier

DATA_PATH = Path(__file__).parent / "data" / "scrime.csv"


def load_scrime(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load scrime simulated SNP data.

    The CSV uses commas as separators.  Columns named ``x.SNP1`` …
    ``x.SNP50`` are genotype values already coded 0–2.  Column ``y`` is
    the 3-class response (0, 1, 2).
    """
    raw = pd.read_csv(path)
    snp_cols = [c for c in raw.columns if c.startswith("x.SNP")]
    X = raw[snp_cols].to_numpy(dtype=object)
    y = raw["y"].to_numpy(dtype=int)
    return X, y


def main() -> None:
    X, y = load_scrime(DATA_PATH)
    n_samples, n_snps = X.shape
    print(f"Loaded scrime data: {n_samples} samples × {n_snps} SNPs")
    for cls in sorted(np.unique(y)):
        print(f"  class {cls}: {(y == cls).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    clf = LogicGPClassifier(
        max_generations=2_000,
        stagnation_generations=200,
        n_bins=3,
        random_state=42,
    )
    print("\nFitting LogicGPClassifier (max_generations=2000) …")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Test F1 (macro): {f1:.4f}")

    rs = clf.to_ruleset()
    print(f"\nRuleset ({len(rs.rules)} rule(s)):")
    for rule in rs.rules:
        print(f"  {rule}")


if __name__ == "__main__":
    main()
