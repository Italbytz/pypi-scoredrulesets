from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    source: str


def load_sklearn_datasets() -> dict[str, DatasetBundle]:
    datasets: dict[str, DatasetBundle] = {}

    iris_X, iris_y = load_iris(return_X_y=True)
    datasets["sklearn_iris"] = DatasetBundle(
        name="sklearn_iris",
        X=np.asarray(iris_X),
        y=np.asarray(iris_y),
        source="sklearn",
    )

    wine_X, wine_y = load_wine(return_X_y=True)
    datasets["sklearn_wine"] = DatasetBundle(
        name="sklearn_wine",
        X=np.asarray(wine_X),
        y=np.asarray(wine_y),
        source="sklearn",
    )

    breast_X, breast_y = load_breast_cancer(return_X_y=True)
    datasets["sklearn_breast_cancer"] = DatasetBundle(
        name="sklearn_breast_cancer",
        X=np.asarray(breast_X),
        y=np.asarray(breast_y),
        source="sklearn",
    )
    return datasets


def load_local_uci_datasets() -> dict[str, DatasetBundle]:
    """Laedt optionale UCI-Datensaetze aus SCORERULESETS_UCI_DIR als CSV.

    Erwartetes Format je Datei: letzte Spalte ist Zielvariable.
    Dateiname wird zu Dataset-Namen `uci_<stem>`.
    """
    root = os.environ.get("SCORERULESETS_UCI_DIR")
    if not root:
        return {}

    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return {}

    bundles: dict[str, DatasetBundle] = {}
    for csv_file in sorted(root_path.glob("*.csv")):
        raw = np.genfromtxt(csv_file, delimiter=",", dtype=object)
        if raw.ndim != 2 or raw.shape[1] < 2:
            continue

        # Optionaler Header: wenn letzte Spalte nicht in Label kodierbar, skippt load.
        try:
            X = raw[:, :-1]
            y = raw[:, -1]
            if X.shape[0] <= 2:
                continue
        except Exception:
            continue

        name = f"uci_{csv_file.stem}"
        bundles[name] = DatasetBundle(name=name, X=X, y=y, source="uci_local")
    return bundles


def load_dataset_registry() -> dict[str, DatasetBundle]:
    registry = load_sklearn_datasets()
    registry.update(load_local_uci_datasets())
    return registry

