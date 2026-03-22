from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    source: str


@dataclass(frozen=True)
class PaperUCIDatasetSpec:
    key: str
    canonical_name: str
    uci_id: int
    openml_data_ids: tuple[int, ...]
    openml_names: tuple[str, ...]
    aliases: tuple[str, ...]


# Datensatzkatalog aus dem Paper (UCI IDs: 12, 267, 17, 19, 45, 53, 78, 109).
PAPER_UCI_DATASET_SPECS: tuple[PaperUCIDatasetSpec, ...] = (
    PaperUCIDatasetSpec(
        key="balance_scale",
        canonical_name="uci_balance_scale",
        uci_id=12,
        openml_data_ids=(11,),
        openml_names=("balance-scale", "balance_scale"),
        aliases=("uci_balance-scale", "uci_balance"),
    ),
    PaperUCIDatasetSpec(
        key="banknote_authentication",
        canonical_name="uci_banknote_authentication",
        uci_id=267,
        openml_data_ids=(),
        openml_names=("banknote-authentication", "banknote_authentication", "banknote"),
        aliases=("uci_banknote", "uci_data_banknote_authentication"),
    ),
    PaperUCIDatasetSpec(
        key="breast_cancer_wisconsin_diagnostic",
        canonical_name="uci_breast_cancer_wisconsin_diagnostic",
        uci_id=17,
        openml_data_ids=(),
        openml_names=("wdbc", "breast-w", "breast_cancer_wisconsin_diagnostic"),
        aliases=("uci_wdbc", "sklearn_breast_cancer"),
    ),
    PaperUCIDatasetSpec(
        key="car_evaluation",
        canonical_name="uci_car_evaluation",
        uci_id=19,
        # data_id=991 ist eine *binarisierte* Version (P/N).
        # data_id=40975 liefert die originalen 4 Klassen (unacc, acc, good, vgood).
        openml_data_ids=(40975,),
        openml_names=("car", "car_evaluation"),
        aliases=("uci_car", "uci_car_evaluation_database"),
    ),
    PaperUCIDatasetSpec(
        key="heart_disease",
        canonical_name="uci_heart_disease",
        uci_id=45,
        # data_id=49 = heart-c (Cleveland, 303 Instanzen, binaer: <50 vs >50_1).
        # Binaer ist die konventionelle Aufgabe fuer diesen Datensatz.
        openml_data_ids=(49,),
        openml_names=("heart-c", "heart-disease", "heart_disease", "heart"),
        aliases=("uci_heart-disease", "uci_heart"),
    ),
    PaperUCIDatasetSpec(
        key="iris",
        canonical_name="uci_iris",
        uci_id=53,
        openml_data_ids=(61,),
        openml_names=("iris",),
        aliases=("sklearn_iris",),
    ),
    PaperUCIDatasetSpec(
        key="page_blocks_classification",
        canonical_name="uci_page_blocks_classification",
        uci_id=78,
        openml_data_ids=(30,),
        openml_names=("page-blocks", "page_blocks", "page_blocks_classification"),
        aliases=("uci_page_blocks", "uci_page-blocks"),
    ),
    PaperUCIDatasetSpec(
        key="wine",
        canonical_name="uci_wine",
        uci_id=109,
        openml_data_ids=(187,),
        openml_names=("wine",),
        aliases=("sklearn_wine",),
    ),
)

PAPER_UCI_DATASET_CANDIDATES: dict[str, tuple[str, ...]] = {
    spec.key: (spec.canonical_name, *spec.aliases) for spec in PAPER_UCI_DATASET_SPECS
}


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
    """Load optional UCI datasets from `SCORERULESETS_UCI_DIR` as CSV files.

    Expected format per file: the last column is the target variable.
    The filename is converted to the dataset name `uci_<stem>`.
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

        # Optional header: if the last column cannot be label-encoded, skip loading.
        try:
            # Remove an optional header row if present.
            if _looks_like_header(raw[0]):
                raw = raw[1:]
            X_raw = raw[:, :-1]
            y_raw = raw[:, -1]
            X, y = _coerce_supervised_arrays(X_raw, y_raw)
            if X.shape[0] <= 2:
                continue
        except Exception:
            continue

        name = f"uci_{csv_file.stem}"
        bundles[name] = DatasetBundle(name=name, X=X, y=y, source="uci_local")
    return bundles


def load_online_paper_uci_datasets() -> dict[str, DatasetBundle]:
    bundles: dict[str, DatasetBundle] = {}
    for spec in PAPER_UCI_DATASET_SPECS:
        loaded = _load_single_paper_dataset_online(spec)
        if loaded is not None:
            bundles[spec.canonical_name] = loaded
    return bundles


def _load_single_paper_dataset_online(spec: PaperUCIDatasetSpec) -> DatasetBundle | None:
    bundle = _load_from_ucimlrepo(spec)
    if bundle is not None:
        return bundle
    return _load_from_openml(spec)


def _load_from_ucimlrepo(spec: PaperUCIDatasetSpec) -> DatasetBundle | None:
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception:
        return None

    try:
        ds = fetch_ucirepo(id=spec.uci_id)
        features = getattr(ds.data, "features", None)
        targets = getattr(ds.data, "targets", None)
        if features is None or targets is None:
            return None
        X, y = _coerce_supervised_arrays(features, targets)
        return DatasetBundle(name=spec.canonical_name, X=X, y=y, source=f"ucimlrepo:{spec.uci_id}")
    except Exception:
        return None


def _load_from_openml(spec: PaperUCIDatasetSpec) -> DatasetBundle | None:
    for data_id in spec.openml_data_ids:
        try:
            ds = fetch_openml(data_id=int(data_id), as_frame=True)
            X, y = _coerce_supervised_arrays(ds.data, ds.target)
            return DatasetBundle(name=spec.canonical_name, X=X, y=y, source=f"openml_id:{data_id}")
        except Exception:
            continue

    for openml_name in spec.openml_names:
        try:
            ds = fetch_openml(name=openml_name, version="active", as_frame=True)
            X, y = _coerce_supervised_arrays(ds.data, ds.target)
            return DatasetBundle(name=spec.canonical_name, X=X, y=y, source=f"openml:{openml_name}")
        except Exception:
            continue
    return None


def _coerce_supervised_arrays(X_raw, y_raw) -> tuple[np.ndarray, np.ndarray]:
    X = _to_numeric_2d(X_raw)
    y = _to_label_encoded_1d(y_raw)

    if X.shape[0] != y.shape[0]:
        raw = X_raw.to_numpy(dtype=object) if hasattr(X_raw, "to_numpy") else np.asarray(X_raw, dtype=object)
        if raw.ndim == 2 and raw.shape[1] >= 2:
            fallback_y = _to_label_encoded_1d(raw[:, -1])
            if fallback_y.shape[0] == raw.shape[0]:
                X = _to_numeric_2d(raw[:, :-1])
                y = fallback_y

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Inconsistent shapes after coercion: X={X.shape}, y={y.shape}")

    return X, y


def _to_numeric_2d(X_raw) -> np.ndarray:
    if hasattr(X_raw, "to_numpy"):
        arr = X_raw.to_numpy(dtype=object)
    else:
        arr = np.asarray(X_raw, dtype=object)

    if arr.ndim != 2:
        arr = np.asarray(arr).reshape(len(arr), -1)

    n_rows, n_cols = arr.shape
    out = np.zeros((n_rows, n_cols), dtype=float)
    for col_idx in range(n_cols):
        col = arr[:, col_idx]
        try:
            out[:, col_idx] = np.asarray(col, dtype=float)
        except Exception:
            labels, encoded = np.unique(np.asarray(col, dtype=object), return_inverse=True)
            out[:, col_idx] = encoded.astype(float)
    return out


def _to_label_encoded_1d(y_raw) -> np.ndarray:
    if hasattr(y_raw, "to_numpy"):
        y_arr = y_raw.to_numpy(dtype=object)
    else:
        y_arr = np.asarray(y_raw, dtype=object)

    y_arr = np.asarray(y_arr, dtype=object)
    if y_arr.ndim == 2:
        if y_arr.shape[1] == 1:
            y_arr = y_arr[:, 0]
        elif y_arr.shape[0] == 1:
            y_arr = y_arr[0, :]
        else:
            y_arr = y_arr[:, 0]
    y_arr = np.asarray(y_arr, dtype=object).reshape(-1)

    # Manche Quellen liefern ein einzelnes Objekt, das den ganzen Zielvektor enthaelt.
    if y_arr.size == 1 and isinstance(y_arr[0], (list, tuple, np.ndarray)):
        y_arr = np.asarray(y_arr[0], dtype=object).reshape(-1)

    _, encoded = np.unique(y_arr, return_inverse=True)
    return encoded.astype(int)


def load_dataset_registry(*, include_online_uci: bool = True) -> dict[str, DatasetBundle]:
    registry = load_sklearn_datasets()
    registry.update(load_local_uci_datasets())
    if include_online_uci:
        uci_bundles = load_online_paper_uci_datasets()
        registry.update(uci_bundles)

        # Deduplizierung: Entferne sklearn-Eintraege, die nur Aliase eines
        # geladenen UCI-Datensatzes sind (z.B. sklearn_iris == uci_iris).
        for spec in PAPER_UCI_DATASET_SPECS:
            if spec.canonical_name in registry:
                for alias in spec.aliases:
                    if alias in registry and alias != spec.canonical_name:
                        del registry[alias]
    return registry


def resolve_dataset_names(
    requested_names: Iterable[str] | None,
    registry: dict[str, DatasetBundle],
    *,
    paper_uci_strict: bool = False,
) -> list[str]:
    """Loest besondere Dataset-Aliasse auf (z.B. ``paper_uci``, ``sklearn_iris`` → ``uci_iris``)."""
    if requested_names is None:
        return list(registry.keys())

    # Alias → kanonischer Name (z.B. sklearn_iris → uci_iris)
    alias_map: dict[str, str] = {}
    for spec in PAPER_UCI_DATASET_SPECS:
        for alias in spec.aliases:
            alias_map[alias] = spec.canonical_name

    resolved: list[str] = []
    for name in requested_names:
        normalized = str(name).strip()
        if not normalized:
            continue
        if normalized == "paper_uci":
            resolved.extend(resolve_paper_uci_dataset_names(registry, strict=paper_uci_strict))
        else:
            # Veraltete Aliase automatisch auf kanonischen Namen umschreiben.
            canonical = alias_map.get(normalized, normalized)
            if canonical in registry:
                resolved.append(canonical)
            elif normalized in registry:
                resolved.append(normalized)
            else:
                resolved.append(normalized)  # _validate_names gibt spaeter den Fehler

    # Reihenfolge behalten, Duplikate entfernen.
    unique: list[str] = []
    seen: set[str] = set()
    for name in resolved:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def resolve_paper_uci_dataset_names(registry: dict[str, DatasetBundle], *, strict: bool = False) -> list[str]:
    """Select available datasets from the UCI catalog used in the paper."""
    selected: list[str] = []
    for _, candidates in PAPER_UCI_DATASET_CANDIDATES.items():
        chosen = next((cand for cand in candidates if cand in registry), None)
        if chosen is not None:
            selected.append(chosen)
    if strict:
        missing = missing_paper_uci_dataset_keys(registry)
        if missing:
            raise ValueError(
                "paper_uci strict mode: missing datasets in registry: " + ", ".join(sorted(missing))
            )
    return selected


def missing_paper_uci_dataset_keys(registry: dict[str, DatasetBundle]) -> list[str]:
    missing: list[str] = []
    for key, candidates in PAPER_UCI_DATASET_CANDIDATES.items():
        if next((cand for cand in candidates if cand in registry), None) is None:
            missing.append(key)
    return missing


def _looks_like_header(row: np.ndarray) -> bool:
    # Heuristik: Eine Headerzeile enthaelt typischerweise mindestens ein nicht-numerisches Feld.
    non_numeric = 0
    for value in row:
        try:
            float(value)
        except Exception:
            non_numeric += 1
    return non_numeric > 0


