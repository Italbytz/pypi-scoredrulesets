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
    no_split: bool = False


@dataclass(frozen=True)
class PaperUCIDatasetSpec:
    key: str
    canonical_name: str
    uci_id: int
    openml_data_ids: tuple[int, ...]
    openml_names: tuple[str, ...]
    aliases: tuple[str, ...]


# Dataset catalog from the paper (UCI IDs: 12, 267, 17, 19, 45, 53, 78, 109).
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
        # data_id=991 is a *binarized* version (P/N).
        # data_id=40975 provides the original 4 classes (unacc, acc, good, vgood).
        openml_data_ids=(40975,),
        openml_names=("car", "car_evaluation"),
        aliases=("uci_car", "uci_car_evaluation_database"),
    ),
    PaperUCIDatasetSpec(
        key="heart_disease",
        canonical_name="uci_heart_disease",
        uci_id=45,
        # data_id=49 = heart-c (Cleveland, 303 instances, binary: <50 vs >50_1).
        # Binary is the conventional task for this dataset.
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

    # Some sources return a single object that contains the full target vector.
    if y_arr.size == 1 and isinstance(y_arr[0], (list, tuple, np.ndarray)):
        y_arr = np.asarray(y_arr[0], dtype=object).reshape(-1)

    _, encoded = np.unique(y_arr, return_inverse=True)
    return encoded.astype(int)


def load_dataset_registry(
    *,
    include_online_uci: bool = True,
    include_synthetic: bool = True,
    include_pmlb: bool = False,
) -> dict[str, DatasetBundle]:
    registry = load_sklearn_datasets()
    registry.update(load_local_uci_datasets())
    registry.update(load_multiplexer_datasets())
    if include_synthetic:
        registry.update(load_synthetic_datasets())
    if include_pmlb:
        registry.update(load_pmlb_datasets())
    if include_online_uci:
        uci_bundles = load_online_paper_uci_datasets()
        registry.update(uci_bundles)

        # Deduplication: remove sklearn entries that are only aliases of
        # already loaded UCI datasets (e.g. sklearn_iris == uci_iris).
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
    """Resolve special dataset aliases (e.g. ``paper_uci``, ``sklearn_iris`` -> ``uci_iris``)."""
    if requested_names is None:
        return list(registry.keys())

    # Alias -> canonical name (e.g. sklearn_iris -> uci_iris)
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
        elif normalized == "multiplexer":
            resolved.extend(
                name for name in registry if name.startswith("mux_")
            )
        elif normalized == "synthetic":
            resolved.extend(
                name for name in registry if name.startswith("synth_")
            )
        elif normalized == "epistasis":
            resolved.extend(
                name for name in registry
                if name.startswith("synth_epistasis") or "epistasis" in name.lower()
            )
        elif normalized == "cart_hard":
            # Datasets where CART is structurally disadvantaged:
            # DNF concepts, checkerboard, MONK, overlapping rules, modular sum.
            _cart_hard_prefixes = (
                "synth_dnf_", "synth_checkerboard_", "synth_monk",
                "synth_overlap_", "synth_modsum_",
            )
            resolved.extend(
                name for name in registry
                if any(name.startswith(p) for p in _cart_hard_prefixes)
            )
        elif normalized == "ruleset_hard":
            # Datasets where rule sets are structurally disadvantaged
            # (CART is better): deep trees, sequential splits,
            # hierarchical interactions.
            _ruleset_hard_prefixes = (
                "synth_deeptree_", "synth_seqthresh_", "synth_hierarch_",
            )
            resolved.extend(
                name for name in registry
                if any(name.startswith(p) for p in _ruleset_hard_prefixes)
            )
        elif normalized == "rule_hard":
            # Datasets that are hard for ALL rule-based estimators (CART +
            # rule sets): not axis-parallel separable.
            _rule_hard_prefixes = (
                "synth_circle", "synth_diagonal_", "synth_spiral",
                "synth_rings_",
            )
            resolved.extend(
                name for name in registry
                if any(name.startswith(p) for p in _rule_hard_prefixes)
            )
        elif normalized == "pmlb":
            resolved.extend(
                name for name in registry if name.startswith("pmlb_")
            )
        else:
            # Automatically rewrite deprecated aliases to canonical names.
            canonical = alias_map.get(normalized, normalized)
            if canonical in registry:
                resolved.append(canonical)
            elif normalized in registry:
                resolved.append(normalized)
            else:
                resolved.append(normalized)  # _validate_names will raise later if needed

    # Preserve order and remove duplicates.
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
    # Heuristic: a header row typically contains at least one non-numeric field.
    non_numeric = 0
    for value in row:
        try:
            float(value)
        except Exception:
            non_numeric += 1
    return non_numeric > 0


# ---------------------------------------------------------------------------
# Multiplexer datasets (boolean classification)
# ---------------------------------------------------------------------------

_MUX_CONFIGS: tuple[tuple[str, int], ...] = (
    ("mux_6", 2),     # 2 address bits + 4 data bits  = 6 features, 2^6 = 64 instances
    ("mux_11", 3),    # 3 address bits + 8 data bits  = 11 features, 2^11 = 2048 instances
    ("mux_20", 4),    # 4 address bits + 16 data bits = 20 features, 2^20 = 1_048_576 instances
)


def generate_multiplexer_dataset(
    n_address_bits: int,
    *,
    max_samples: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a multiplexer dataset.

    A k-multiplexer has ``k`` address bits and ``2**k`` data bits.
    The total number of features is ``k + 2**k``.
    The output value is the data bit at the position encoded by address bits.

    With full enumeration, the dataset contains ``2**(k + 2**k)`` rows.
    For large k, ``max_samples`` can cap row count via random sampling.

    Parameters
    ----------
    n_address_bits : int
        Number of address bits (1, 2, 3, ...).
    max_samples : int or None
        Maximum row count. ``None`` = full enumeration.
    random_state : int or None
        Seed for reproducible sampling (only relevant if ``max_samples`` is set).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype int
    y : np.ndarray, shape (n_samples,), dtype int (0 oder 1)
    """
    n_data_bits = 1 << n_address_bits  # 2**k
    n_features = n_address_bits + n_data_bits
    n_total = 1 << n_features  # 2**(k + 2**k)

    use_sampling = max_samples is not None and max_samples < n_total

    if use_sampling:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_total, size=max_samples, replace=False)
        indices.sort()
    else:
        indices = np.arange(n_total)

    n_samples = len(indices)
    X = np.zeros((n_samples, n_features), dtype=int)
    y = np.zeros(n_samples, dtype=int)

    for row, idx in enumerate(indices):
        # Binary representation of this instance.
        bits = [(idx >> b) & 1 for b in range(n_features)]
        X[row] = bits
        # Address bits -> position in data-bit region.
        address = sum(bits[a] << a for a in range(n_address_bits))
        # Data bit at that position.
        y[row] = bits[n_address_bits + address]

    return X, y


def load_multiplexer_datasets(
    *,
    max_samples_large: int = 10_000,
) -> dict[str, DatasetBundle]:
    """Generate standard multiplexer datasets (mux_6 to mux_37).

    For large multiplexers (>= 2^16 instances), sampling is used to keep
    runtime manageable.

    Parameters
    ----------
    max_samples_large : int
        Maximum row count for large multiplexers (default: 10_000).
    """
    bundles: dict[str, DatasetBundle] = {}
    for name, n_addr in _MUX_CONFIGS:
        n_features = n_addr + (1 << n_addr)
        n_total = 1 << n_features
        # Sample only for large datasets.
        if n_total > max_samples_large:
            X, y_arr = generate_multiplexer_dataset(
                n_addr, max_samples=max_samples_large, random_state=42
            )
        else:
            X, y_arr = generate_multiplexer_dataset(n_addr)
        bundles[name] = DatasetBundle(
            name=name,
            X=X,
            y=y_arr,
            source=f"multiplexer_{n_addr + (1 << n_addr)}",
            no_split=True,
        )
    return bundles


# ---------------------------------------------------------------------------
# Synthetic biomedical / SNP datasets
# ---------------------------------------------------------------------------


def generate_epistasis_dataset(
    n_samples: int = 1600,
    n_snps: int = 20,
    n_interacting: int = 2,
    heritability: float = 0.4,
    minor_allele_freq: float = 0.3,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic epistasis dataset (SNP-SNP interaction).

    The class is determined exclusively by an interaction among
    ``n_interacting`` SNPs; all other SNPs are noise features.
    The model emulates GAMETES behavior: causal SNPs jointly create an
    XOR-like pattern (no main effect, only interaction).

    Parameters
    ----------
    n_samples : int
        Number of instances.
    n_snps : int
        Total number of SNP features (including causal SNPs).
    n_interacting : int
        Number of interacting causal SNPs (2 or 3 recommended).
    heritability : float
        Strength of genetic signal (0 = no signal, 1 = perfectly separable).
        Controls the penetrance table.
    minor_allele_freq : float
        Minor allele frequency for Hardy-Weinberg equilibrium (0 < maf < 0.5).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_snps), dtype int  (Werte: 0, 1, 2)
    y : np.ndarray, shape (n_samples,), dtype int  (0 oder 1)
    """
    rng = np.random.default_rng(random_state)
    assert n_interacting <= n_snps, "n_interacting must not exceed n_snps"
    assert 0 < minor_allele_freq < 0.5, "minor_allele_freq must be in (0, 0.5)"

    # SNP genotypes under Hardy-Weinberg equilibrium
    p = minor_allele_freq
    hw_probs = [(1 - p) ** 2, 2 * p * (1 - p), p ** 2]  # P(0), P(1), P(2)
    X = rng.choice(3, size=(n_samples, n_snps), p=hw_probs)

    # Causal SNPs: the first n_interacting features
    causal = X[:, :n_interacting]

    # XOR-like interaction: class = 1 if the sum of causal alleles
    # is odd (pure epistasis, no main effect).
    interaction_signal = np.sum(causal, axis=1) % 2  # 0 oder 1

    # Penetrance by heritability: P(case | signal=1) = 0.5 + h/2,
    # P(case | signal=0) = 0.5 - h/2. At h=1 -> 1.0 / 0.0 (deterministic).
    pen_high = min(1.0, 0.5 + heritability / 2)
    pen_low = max(0.0, 0.5 - heritability / 2)
    probs = np.where(interaction_signal == 1, pen_high, pen_low)
    y = (rng.random(n_samples) < probs).astype(int)

    # Permute feature columns so causal SNPs are not always at the front
    # (otherwise the benchmark is too easy).
    perm = rng.permutation(n_snps)
    X = X[:, perm]

    return X, y


def generate_xor_parity_dataset(
    n_bits: int = 6,
    n_noise_features: int = 14,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an XOR/parity dataset with noise features.

    The class is the parity (XOR) of the first ``n_bits`` binary features.
    Additionally, ``n_noise_features`` random binary features are appended.
    This dataset is non-linearly separable and requires conjunctive rules.

    Parameters
    ----------
    n_bits : int
        Number of relevant parity bits.
    n_noise_features : int
        Number of irrelevant noise features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_bits + n_noise_features), dtype int
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    rng = np.random.default_rng(random_state)
    total_features = n_bits + n_noise_features
    X = rng.integers(0, 2, size=(n_samples, total_features))
    y = np.bitwise_xor.reduce(X[:, :n_bits], axis=1)

    # Permute columns
    perm = rng.permutation(total_features)
    X = X[:, perm]

    return X, y


def generate_highdim_lowsample_dataset(
    n_samples: int = 120,
    n_features: int = 500,
    n_informative: int = 5,
    n_classes: int = 2,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a high-dimensional dataset with few samples (p >> n).

    Typical for genomics scenarios (e.g. microarray, SNP panels): many features,
    few patients, and only a few informative features.

    Parameters
    ----------
    n_samples : int
        Number of instances (typically small, e.g. 80-200).
    n_features : int
        Total number of features (typically large, e.g. 200-2000).
    n_informative : int
        Number of truly informative features.
    n_classes : int
        Number of classes.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), dtype int
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        n_classes=n_classes,
        flip_y=0.03,
        class_sep=1.0,
        random_state=random_state,
    )
    return X, y


def generate_imbalanced_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 5,
    imbalance_ratio: float = 0.1,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with strong class imbalance.

    Simulates rare diseases or rare events.

    Parameters
    ----------
    n_samples : int
        Total number of instances.
    n_features : int
        Number of features.
    n_informative : int
        Number of informative features.
    imbalance_ratio : float
        Minority class ratio (e.g. 0.1 = 10%).
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 or 1)
    """
    from sklearn.datasets import make_classification

    n_minority = max(2, int(n_samples * imbalance_ratio))
    n_majority = n_samples - n_minority
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        flip_y=0.01,
        random_state=random_state,
    )
    return X, y


# ---------------------------------------------------------------------------
# Datasets that explicitly expose CART weaknesses (DNF, checkerboard, ...)
# ---------------------------------------------------------------------------


def generate_dnf_concept_dataset(
    n_disjuncts: int = 3,
    n_conjuncts: int = 2,
    n_noise_features: int = 10,
    n_samples: int = 2000,
    noise_rate: float = 0.0,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with a DNF decision rule.

    The true concept is a disjunction of ``n_disjuncts`` conjunctions,
    where each conjunction requires ``n_conjuncts`` different binary features.
    CART must duplicate subtrees for this; rule sets can represent each
    disjunct directly as a separate rule.

    Parameters
    ----------
    n_disjuncts : int
        Number of disjuncts (OR terms).
    n_conjuncts : int
        Number of conjuncts per disjunct (AND terms).
    n_noise_features : int
        Additional irrelevant binary features.
    n_samples : int
        Number of instances.
    noise_rate : float
        Fraction of randomly flipped labels (0 = no noise).
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_relevant = n_disjuncts * n_conjuncts
    n_features = n_relevant + n_noise_features
    X = rng.integers(0, 2, size=(n_samples, n_features))

    # DNF: y = OR_d( AND_c( X[:, d*n_conjuncts + c] == 1 ) )
    y = np.zeros(n_samples, dtype=int)
    for d in range(n_disjuncts):
        clause = np.ones(n_samples, dtype=bool)
        for c in range(n_conjuncts):
            clause &= X[:, d * n_conjuncts + c] == 1
        y |= clause.astype(int)

    # Label noise
    if noise_rate > 0:
        flip = rng.random(n_samples) < noise_rate
        y[flip] = 1 - y[flip]

    # Permute columns
    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_checkerboard_dataset(
    n_tiles: int = 4,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2D checkerboard dataset with noise features.

    The checkerboard pattern (XOR over quantized continuous features)
    requires O(n_tiles^2) leaves for CART, while rule sets can represent
    regions directly as rules.

    Parameters
    ----------
    n_tiles : int
        Number of tiles per axis (e.g. 4 -> 4x4 checkerboard).
    n_noise_features : int
        Additional uniform noise features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    X_rel = rng.uniform(0, 1, size=(n_samples, 2))
    tile_x = np.floor(X_rel[:, 0] * n_tiles).astype(int)
    tile_y = np.floor(X_rel[:, 1] * n_tiles).astype(int)
    y = ((tile_x + tile_y) % 2).astype(int)

    X_noise = rng.uniform(0, 1, size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])

    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, y


def generate_monk1_dataset(
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the MONK-1 dataset (disjunction: (a1==a2) OR (a5==1)).

    MONK-1 is a classic benchmark for rule learners. The decision rule is a
    disjunction that CART can represent only via subtree duplication.

    Attribute:
        a1 in {1,2,3}, a2 in {1,2,3}, a3 in {1,2},
        a4 in {1,2,3}, a5 in {1,2,3,4}, a6 in {1,2}

    Parameters
    ----------
    n_samples : int
        Number of instances (sampling with replacement if > 432).
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    domains = [3, 3, 2, 3, 4, 2]  # a1..a6
    n_total = 1
    for d in domains:
        n_total *= d  # 432

    # Full enumeration
    rows = np.zeros((n_total, 6), dtype=int)
    idx = 0
    for a1 in range(domains[0]):
        for a2 in range(domains[1]):
            for a3 in range(domains[2]):
                for a4 in range(domains[3]):
                    for a5 in range(domains[4]):
                        for a6 in range(domains[5]):
                            rows[idx] = [a1, a2, a3, a4, a5, a6]
                            idx += 1

    # MONK-1 rule: (a1 == a2) OR (a5 == 1)
    # (0-based indexing: a5 == 1 -> rows[:, 4] == 1)
    y_full = ((rows[:, 0] == rows[:, 1]) | (rows[:, 4] == 1)).astype(int)

    if n_samples >= n_total:
        X, y = rows, y_full
    else:
        chosen = rng.choice(n_total, size=n_samples, replace=True)
        X, y = rows[chosen], y_full[chosen]

    return X.astype(float), y


def generate_monk3_dataset(
    n_samples: int = 2000,
    noise_rate: float = 0.05,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the MONK-3 dataset (conjunction + exception + noise).

    MONK-3 rule: (a5 != 3 AND a4 != 1) OR (a5 == 3 AND a2 != 3),
    plus ``noise_rate`` zufaellig geflippter Labels.

    Parameters
    ----------
    n_samples : int
        Number of instances.
    noise_rate : float
        Fraction of randomly flipped labels (default: 5%).
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    domains = [3, 3, 2, 3, 4, 2]
    n_total = 1
    for d in domains:
        n_total *= d

    rows = np.zeros((n_total, 6), dtype=int)
    idx = 0
    for a1 in range(domains[0]):
        for a2 in range(domains[1]):
            for a3 in range(domains[2]):
                for a4 in range(domains[3]):
                    for a5 in range(domains[4]):
                        for a6 in range(domains[5]):
                            rows[idx] = [a1, a2, a3, a4, a5, a6]
                            idx += 1

    # MONK-3: (a5 != 3 AND a4 != 1) OR (a5 == 3 AND a2 != 3)
    # 0-based indexing: a5 != 3 -> rows[:, 4] != 3; a4 != 1 -> rows[:, 3] != 1; etc.
    y_full = (
        ((rows[:, 4] != 3) & (rows[:, 3] != 1))
        | ((rows[:, 4] == 3) & (rows[:, 1] != 2))
    ).astype(int)

    if n_samples >= n_total:
        X, y = rows, y_full
    else:
        chosen = rng.choice(n_total, size=n_samples, replace=True)
        X, y = rows[chosen], y_full[chosen]

    # Label noise
    if noise_rate > 0:
        flip = rng.random(len(y)) < noise_rate
        y[flip] = 1 - y[flip]

    return X.astype(float), y


def generate_overlapping_rules_dataset(
    n_rules: int = 4,
    n_features_per_rule: int = 2,
    n_noise_features: int = 10,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with multiple independent, overlapping rules.

    Each rule defines a positive region using a threshold over
    ``n_features_per_rule`` continuous features. The class is 1 if at least
    one rule fires. CART struggles because the rules are not hierarchical.

    Parameters
    ----------
    n_rules : int
        Number of independent rules.
    n_features_per_rule : int
        Number of features per rule.
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_relevant = n_rules * n_features_per_rule
    n_total_features = n_relevant + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_total_features))

    y = np.zeros(n_samples, dtype=int)
    for r in range(n_rules):
        start = r * n_features_per_rule
        end = start + n_features_per_rule
        # Rule: all features in [0.6, 1.0]
        fired = np.all(X[:, start:end] > 0.6, axis=1)
        y[fired] = 1

    # Permute columns
    perm = rng.permutation(n_total_features)
    X = X[:, perm]
    return X, y


def generate_modular_sum_dataset(
    n_relevant: int = 4,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    modulus: int = 3,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset where class depends on a modular sum.

    y = (sum(X[:, :n_relevant]) mod modulus == 0). This pattern creates
    non-axis-parallel decision boundaries that CART can only approximate
    inefficiently.

    Parameters
    ----------
    n_relevant : int
        Number of relevant features.
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    modulus : int
        Modulus for the sum function.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_total = n_relevant + n_noise_features
    X = rng.integers(0, modulus + 1, size=(n_samples, n_total))
    y = (np.sum(X[:, :n_relevant], axis=1) % modulus == 0).astype(int)

    perm = rng.permutation(n_total)
    X = X[:, perm]
    return X.astype(float), y


# ---------------------------------------------------------------------------
# Datasets where CART outperforms rule sets
# (deep hierarchies, sequential splits - natural tree structures)
# ---------------------------------------------------------------------------


def generate_deep_tree_dataset(
    depth: int = 5,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset whose true concept is a deep binary tree.

    At each internal node, a different feature is split on
    (feature i, threshold 0.5). The leaf determines class (0 or 1,
    alternating). CART can represent this tree directly; rule sets need
    2^(depth-1) rules because each path is a separate conjunction.

    Parameters
    ----------
    depth : int
        Depth of the true tree (requires at least ``depth`` features).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_relevant = depth
    n_features = n_relevant + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))

    # Tree traversal: split on feature t at depth t (threshold 0.5).
    # The leaf path encodes a binary number -> class = parity of path directions.
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        node = 0
        for t in range(depth):
            if X[i, t] > 0.5:
                node = 2 * node + 2
            else:
                node = 2 * node + 1
        y[i] = node % 2

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_sequential_threshold_dataset(
    n_bins: int = 5,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with sequential thresholds on one feature.

    Class depends on which of ``n_bins`` intervals a single continuous
    feature falls into: even bins -> class 0, odd bins -> class 1.
    CART needs only (n_bins - 1) splits; rule sets must create one rule
    with lower and upper bound for each interval.

    Parameters
    ----------
    n_bins : int
        Number of intervals (alternating class 0/1).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_features = 1 + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    # Relevant feature: column 0
    bin_idx = np.clip(np.floor(X[:, 0] * n_bins).astype(int), 0, n_bins - 1)
    y = (bin_idx % 2).astype(int)

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_hierarchical_interaction_dataset(
    n_context_features: int = 3,
    n_response_features: int = 3,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with hierarchical feature interaction.

    A context feature first determines which response feature is relevant.
    Class then depends on the value of the selected response feature.
    CART represents this naturally as a tree (split on context, then on
    response); rule sets cannot encode this conditional relevance compactly.

    Parameters
    ----------
    n_context_features : int
        Number of possible contexts (one feature, n values).
    n_response_features : int
        Number of response features (one relevant per context).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_used_responses = min(n_context_features, n_response_features)
    n_features = 1 + n_response_features + n_noise_features  # 1 context + responses + noise
    X = rng.uniform(0, 1, size=(n_samples, n_features))

    # Context feature (column 0) is quantized into n_context_features bins.
    context = np.clip(
        np.floor(X[:, 0] * n_context_features).astype(int), 0, n_context_features - 1
    )
    # Depending on context, a different response feature is relevant.
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        resp_idx = context[i] % n_used_responses
        y[i] = int(X[i, 1 + resp_idx] > 0.5)

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


# ---------------------------------------------------------------------------
# Datasets that are hard for ALL rule-based estimators
# (not axis-parallel separable -> SVM, kNN, etc. are better)
# ---------------------------------------------------------------------------


def generate_circle_boundary_dataset(
    n_noise_features: int = 8,
    n_samples: int = 2000,
    radius: float = 0.7,
    noise_std: float = 0.05,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with a circular decision boundary.

    Class 1 if the point lies inside a circle (Euclidean distance from
    center < radius). Axis-parallel rules (CART and rule sets) must
    approximate the circle with many rectangles. SVM with RBF kernel
    or kNN solves this easily.

    Parameters
    ----------
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    radius : float
        Radius of the circle boundary (features in [0, 1], center (0.5, 0.5)).
    noise_std : float
        Gaussian noise on the radius.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    X_rel = rng.uniform(0, 1, size=(n_samples, 2))
    dist = np.sqrt((X_rel[:, 0] - 0.5) ** 2 + (X_rel[:, 1] - 0.5) ** 2)
    effective_radius = radius / 2  # scaled to [0, 1] space
    y = (dist + rng.normal(0, noise_std, n_samples) < effective_radius).astype(int)

    X_noise = rng.uniform(0, 1, size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, y


def generate_diagonal_boundary_dataset(
    n_relevant: int = 4,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    noise_std: float = 0.1,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with a diagonal (45 degree) decision boundary.

    The true boundary is a hyperplane sum(X[:, :n_relevant]) > threshold.
    Axis-parallel methods need a staircase approximation; linear models,
    SVM, or kNN solve this directly.

    Parameters
    ----------
    n_relevant : int
        Number of relevant features (their sum determines class).
    n_noise_features : int
        Additional irrelevant features.
    n_samples : int
        Number of instances.
    noise_std : float
        Gaussian noise on the decision function.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_features = n_relevant + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    decision = np.sum(X[:, :n_relevant], axis=1) + rng.normal(0, noise_std, n_samples)
    threshold = n_relevant / 2.0  # center point
    y = (decision > threshold).astype(int)

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


def generate_spiral_dataset(
    n_samples: int = 2000,
    n_noise_features: int = 8,
    noise_std: float = 0.15,
    n_turns: float = 1.5,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-spirals dataset with noise features.

    Two interleaved spirals are a classic benchmark for nonlinear
    classifiers. Neither CART nor rule sets can approximate spiral
    structure efficiently, while kNN and neural networks handle it well.

    Parameters
    ----------
    n_samples : int
        Total number of instances (half per spiral).
    n_noise_features : int
        Additional noise features.
    noise_std : float
        Radial noise.
    n_turns : float
        Number of spiral turns.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2
    theta = np.sqrt(rng.uniform(0, 1, n_half)) * n_turns * 2 * np.pi

    # Spiral 1
    r1 = theta + rng.normal(0, noise_std, n_half)
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)

    # Spiral 2 (rotated by 180 degrees)
    r2 = theta + rng.normal(0, noise_std, n_half)
    x2 = -r2 * np.cos(theta)
    y2 = -r2 * np.sin(theta)

    X_rel = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2]),
    ])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_half)]).astype(int)

    X_noise = rng.uniform(
        X_rel.min(), X_rel.max(), size=(n_samples, n_noise_features)
    )
    X = np.hstack([X_rel, X_noise])

    # Shuffle and permute columns
    shuffle = rng.permutation(n_samples)
    X = X[shuffle]
    labels = labels[shuffle]
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, labels


def generate_concentric_rings_dataset(
    n_rings: int = 3,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    noise_std: float = 0.05,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with concentric rings (alternating classes).

    Ring i has class (i mod 2). This pattern requires circular
    decision boundaries that axis-parallel methods only approximate coarsely.

    Parameters
    ----------
    n_rings : int
        Number of rings.
    n_noise_features : int
        Additional noise features.
    n_samples : int
        Number of instances.
    noise_std : float
        Radial Gaussian noise.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    angle = rng.uniform(0, 2 * np.pi, n_samples)
    # Distribute uniformly across rings
    ring_idx = rng.integers(0, n_rings, n_samples)
    # Radius: ring i has radius i/n_rings (normalized)
    radius = (ring_idx + 0.5) / n_rings + rng.normal(0, noise_std, n_samples)
    X_rel = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = (ring_idx % 2).astype(int)

    X_noise = rng.uniform(-1, 1, size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    return X, y


# -- Standard configurations for synthetic datasets ------------------

_SYNTH_CONFIGS: dict[str, dict] = {
    # Epistasis: 2-way interaction, medium
    "synth_epistasis_2way_easy": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=1600, n_snps=20, n_interacting=2, heritability=0.6, random_state=42),
    ),
    "synth_epistasis_2way_hard": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=1600, n_snps=20, n_interacting=2, heritability=0.2, random_state=42),
    ),
    # Epistasis: 3-way interaction (harder)
    "synth_epistasis_3way": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=2000, n_snps=30, n_interacting=3, heritability=0.4, random_state=42),
    ),
    # Epistasis: high-dimensional (many noise SNPs)
    "synth_epistasis_highdim": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=1600, n_snps=100, n_interacting=2, heritability=0.4, random_state=42),
    ),
    # XOR / parity
    "synth_xor_3bit": dict(
        generator=generate_xor_parity_dataset,
        params=dict(n_bits=3, n_noise_features=7, n_samples=2000, random_state=42),
    ),
    "synth_xor_5bit": dict(
        generator=generate_xor_parity_dataset,
        params=dict(n_bits=5, n_noise_features=15, n_samples=2000, random_state=42),
    ),
    # p >> n (genomics scenario)
    "synth_highdim_p500_n120": dict(
        generator=generate_highdim_lowsample_dataset,
        params=dict(n_samples=120, n_features=500, n_informative=5, random_state=42),
    ),
    "synth_highdim_p1000_n80": dict(
        generator=generate_highdim_lowsample_dataset,
        params=dict(n_samples=80, n_features=1000, n_informative=4, random_state=42),
    ),
    # Imbalanced (rare disease)
    "synth_imbalanced_10pct": dict(
        generator=generate_imbalanced_dataset,
        params=dict(n_samples=1000, n_features=12, n_informative=5, imbalance_ratio=0.1, random_state=42),
    ),
    "synth_imbalanced_5pct": dict(
        generator=generate_imbalanced_dataset,
        params=dict(n_samples=2000, n_features=15, n_informative=6, imbalance_ratio=0.05, random_state=42),
    ),
    # --- CART-difficult datasets (rule sets should perform better) ---
    # DNF: easy - 3 disjuncts with 2 conjuncts each
    "synth_dnf_3x2": dict(
        generator=generate_dnf_concept_dataset,
        params=dict(n_disjuncts=3, n_conjuncts=2, n_noise_features=10, n_samples=2000, random_state=42),
    ),
    # DNF: complex - 5 disjuncts with 3 conjuncts each
    "synth_dnf_5x3": dict(
        generator=generate_dnf_concept_dataset,
        params=dict(n_disjuncts=5, n_conjuncts=3, n_noise_features=15, n_samples=3000, random_state=42),
    ),
    # DNF: noisy
    "synth_dnf_3x2_noisy": dict(
        generator=generate_dnf_concept_dataset,
        params=dict(n_disjuncts=3, n_conjuncts=2, n_noise_features=10, n_samples=2000, noise_rate=0.05, random_state=42),
    ),
    # Checkerboard: 4x4 grid
    "synth_checkerboard_4x4": dict(
        generator=generate_checkerboard_dataset,
        params=dict(n_tiles=4, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Checkerboard: 6x6 grid (harder)
    "synth_checkerboard_6x6": dict(
        generator=generate_checkerboard_dataset,
        params=dict(n_tiles=6, n_noise_features=8, n_samples=3000, random_state=42),
    ),
    # MONK-1: disjunction (a1==a2) OR (a5==1) - classic rule-learning benchmark
    "synth_monk1": dict(
        generator=generate_monk1_dataset,
        params=dict(n_samples=2000, random_state=42),
    ),
    # MONK-3: conjunction + exception + 5% noise
    "synth_monk3": dict(
        generator=generate_monk3_dataset,
        params=dict(n_samples=2000, noise_rate=0.05, random_state=42),
    ),
    # Overlapping subgroups: 4 independent rules
    "synth_overlap_4rules": dict(
        generator=generate_overlapping_rules_dataset,
        params=dict(n_rules=4, n_features_per_rule=2, n_noise_features=10, n_samples=2000, random_state=42),
    ),
    # Overlapping subgroups: 6 rules (harder)
    "synth_overlap_6rules": dict(
        generator=generate_overlapping_rules_dataset,
        params=dict(n_rules=6, n_features_per_rule=2, n_noise_features=12, n_samples=3000, random_state=42),
    ),
    # Modular sum: mod 3
    "synth_modsum_mod3": dict(
        generator=generate_modular_sum_dataset,
        params=dict(n_relevant=4, n_noise_features=8, n_samples=2000, modulus=3, random_state=42),
    ),
    # Modular sum: mod 4 (harder)
    "synth_modsum_mod4": dict(
        generator=generate_modular_sum_dataset,
        params=dict(n_relevant=5, n_noise_features=10, n_samples=2500, modulus=4, random_state=42),
    ),
    # --- Ruleset-difficult datasets (CART should perform better) ---
    # Deep tree: depth=5 -> 16 leaves, rule sets need 8 rules
    "synth_deeptree_d5": dict(
        generator=generate_deep_tree_dataset,
        params=dict(depth=5, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Deep tree: depth=7 -> 64 leaves, even harder for rule sets
    "synth_deeptree_d7": dict(
        generator=generate_deep_tree_dataset,
        params=dict(depth=7, n_noise_features=8, n_samples=3000, random_state=42),
    ),
    # Sequential thresholds: 5 bins on one feature
    "synth_seqthresh_5bin": dict(
        generator=generate_sequential_threshold_dataset,
        params=dict(n_bins=5, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Sequential thresholds: 8 bins (harder)
    "synth_seqthresh_8bin": dict(
        generator=generate_sequential_threshold_dataset,
        params=dict(n_bins=8, n_noise_features=10, n_samples=3000, random_state=42),
    ),
    # Hierarchical interaction: 3 contexts x 3 responses
    "synth_hierarch_3x3": dict(
        generator=generate_hierarchical_interaction_dataset,
        params=dict(n_context_features=3, n_response_features=3, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Hierarchical interaction: 5 contexts x 5 responses (harder)
    "synth_hierarch_5x5": dict(
        generator=generate_hierarchical_interaction_dataset,
        params=dict(n_context_features=5, n_response_features=5, n_noise_features=10, n_samples=3000, random_state=42),
    ),
    # --- Difficult for ALL rule-based estimators (SVM/kNN better) ---
    # Circular boundary
    "synth_circle": dict(
        generator=generate_circle_boundary_dataset,
        params=dict(n_noise_features=8, n_samples=2000, radius=0.7, noise_std=0.03, random_state=42),
    ),
    # Circular boundary: stronger noise
    "synth_circle_noisy": dict(
        generator=generate_circle_boundary_dataset,
        params=dict(n_noise_features=8, n_samples=2000, radius=0.7, noise_std=0.08, random_state=42),
    ),
    # Diagonal boundary: 4 relevant features
    "synth_diagonal_4d": dict(
        generator=generate_diagonal_boundary_dataset,
        params=dict(n_relevant=4, n_noise_features=8, n_samples=2000, noise_std=0.1, random_state=42),
    ),
    # Diagonal boundary: 8 relevant features (harder)
    "synth_diagonal_8d": dict(
        generator=generate_diagonal_boundary_dataset,
        params=dict(n_relevant=8, n_noise_features=8, n_samples=3000, noise_std=0.1, random_state=42),
    ),
    # Two Spirals
    "synth_spiral": dict(
        generator=generate_spiral_dataset,
        params=dict(n_samples=2000, n_noise_features=8, noise_std=0.15, n_turns=1.5, random_state=42),
    ),
    # Concentric rings: 3 rings
    "synth_rings_3": dict(
        generator=generate_concentric_rings_dataset,
        params=dict(n_rings=3, n_noise_features=8, n_samples=2000, noise_std=0.04, random_state=42),
    ),
    # Concentric rings: 5 rings (harder)
    "synth_rings_5": dict(
        generator=generate_concentric_rings_dataset,
        params=dict(n_rings=5, n_noise_features=8, n_samples=3000, noise_std=0.03, random_state=42),
    ),
}


def load_synthetic_datasets() -> dict[str, DatasetBundle]:
    """Generate all predefined synthetic benchmark datasets."""
    bundles: dict[str, DatasetBundle] = {}
    for name, cfg in _SYNTH_CONFIGS.items():
        X, y = cfg["generator"](**cfg["params"])
        bundles[name] = DatasetBundle(
            name=name,
            X=np.asarray(X, dtype=float),
            y=np.asarray(y, dtype=int),
            source="synthetic",
        )
    return bundles


# ---------------------------------------------------------------------------
# PMLB datasets (Penn Machine Learning Benchmarks)
# ---------------------------------------------------------------------------

# Curated selection of biomedically relevant PMLB datasets.
# Includes GAMETES epistasis datasets and additional interesting tasks.
_PMLB_DATASET_NAMES: tuple[str, ...] = (
    # GAMETES epistasis (2-way, different heritabilities / EDMs)
    "GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1",
    "GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1",
    "GAMETES_Epistasis_2-Way_1000atts_0.4H_EDM-1_EDM-1_1",
    # GAMETES epistasis (3-way)
    "GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1",
    # Heterogeneous (mix of main effect + interaction)
    "GAMETES_Heterogeneous_20atts_1600_Het_0.4_0.2_50_EDM-2_001",
    # Additional biomedically relevant datasets
    "saheart",       # South-African Heart Disease
    "pima",          # Pima-Indians Diabetes
    "analcatdata_aids",  # AIDS-Daten
)


def load_pmlb_datasets() -> dict[str, DatasetBundle]:
    """Load curated datasets from the Penn Machine Learning Benchmark Suite.

    Requires ``pip install pmlb``. Returns an empty dict if pmlb is
    not installed or if download fails.

    Returns
    -------
    dict[str, DatasetBundle]
        Mapping ``pmlb_<name>`` -> DatasetBundle.
    """
    try:
        import pmlb
    except ImportError:
        return {}

    bundles: dict[str, DatasetBundle] = {}
    for ds_name in _PMLB_DATASET_NAMES:
        try:
            df = pmlb.fetch_data(ds_name, local_cache_dir=None)
            if "target" not in df.columns:
                continue
            y_raw = df["target"].values
            X_raw = df.drop(columns=["target"]).values
            X = np.asarray(X_raw, dtype=float)
            _, y = np.unique(y_raw, return_inverse=True)
            y = y.astype(int)
            if X.shape[0] < 10 or X.shape[1] < 1:
                continue
            bundle_name = f"pmlb_{ds_name}"
            bundles[bundle_name] = DatasetBundle(
                name=bundle_name,
                X=X,
                y=y,
                source=f"pmlb:{ds_name}",
            )
        except Exception:
            continue
    return bundles


