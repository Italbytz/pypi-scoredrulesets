#!/usr/bin/env python3
"""Quick GPAS run on HapMap CHB vs JPT SNP data.

Demonstrates binary classification with GPASClassifier on real
HapMap data (chromosome 20 SNPs, CHB = Han Chinese in Beijing vs
JPT = Japanese in Tokyo).

Equivalent to quick_hapmap_run.R in the paper-code companion repo.

Usage
-----
Run from the pypi-scoredrulesets repository root::

    python examples/snp/quick_hapmap_run.py

The script uses a fixed random seed so results are reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from scoredrulesets import GPASClassifier

DATA_PATH = Path(__file__).parent / "data" / "hapmap157.csv"


def load_hapmap(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and recode HapMap data.

    The CSV uses semicolons as separators.  Column 0 is the population
    label (0 = CHB, 1 = JPT).  Columns 1–157 are SNP genotypes coded
    1–3; we subtract 1 to obtain the 0-based encoding expected by
    GPASClassifier.
    """
    raw = pd.read_csv(path, sep=";")
    y = raw.iloc[:, 0].to_numpy(dtype=int)
    X = (raw.iloc[:, 1:].to_numpy(dtype=int) - 1).astype(object)
    return X, y


def main() -> None:
    X, y = load_hapmap(DATA_PATH)
    n_samples, n_snps = X.shape
    print(f"Loaded HapMap data: {n_samples} samples × {n_snps} SNPs")
    print(f"  CHB (0): {(y == 0).sum()}, JPT (1): {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    clf = GPASClassifier(
        max_generations=15_000,
        stagnation_generations=1_000,
        n_bins=3,
        random_state=42,
    )
    print("\nFitting GPASClassifier (max_generations=15000) …")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="micro")
    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Test F1 (micro): {f1:.4f}")

    rs = clf.to_ruleset()
    print(f"\nRuleset ({len(rs.rules)} rule(s)):")
    for rule in rs.rules:
        print(f"  {rule}")


if __name__ == "__main__":
    main()
