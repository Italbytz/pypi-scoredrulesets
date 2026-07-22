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
# Multiplexer and synthetic datasets — delegated to catgen
# ---------------------------------------------------------------------------
from catgen.datasets import (  # noqa: E402
    generate_multiplexer_dataset,
    generate_xor_parity_dataset,
    generate_dnf_concept_dataset,
    generate_monk1_dataset,
    generate_monk3_dataset,
    generate_overlapping_rules_dataset,
    generate_modular_sum_dataset,
    generate_epistasis_dataset,
    generate_highdim_lowsample_dataset,
    generate_imbalanced_dataset,
    generate_checkerboard_dataset,
    generate_circle_boundary_dataset,
    generate_diagonal_boundary_dataset,
    generate_spiral_dataset,
    generate_concentric_rings_dataset,
    generate_deep_tree_dataset,
    generate_sequential_threshold_dataset,
    generate_hierarchical_interaction_dataset,
)


def load_multiplexer_datasets(
    *,
    max_samples_large: int = 10_000,
) -> dict[str, DatasetBundle]:
    """Generate standard multiplexer datasets (mux_6 to mux_20).

    Delegates generation to ``catgen`` and wraps results in
    :class:`DatasetBundle` objects for use within scoredrulesets.

    Parameters
    ----------
    max_samples_large : int
        Maximum row count for large multiplexers (default: 10_000).
    """
    bundles: dict[str, DatasetBundle] = {}
    
    # Generate multiplexer datasets from mux_6 to mux_20
    # Larger ones get capped at max_samples_large
    for k in range(6, 21):
        try:
            X, y = generate_multiplexer_dataset(k=k, random_state=42)
            
            # Cap large datasets
            if X.shape[0] > max_samples_large:
                indices = np.random.RandomState(42).choice(X.shape[0], max_samples_large, replace=False)
                X = X[indices]
                y = y[indices]
            
            bundles[f"mux_{k}"] = DatasetBundle(
                name=f"mux_{k}",
                X=X,
                y=y,
                source=f"multiplexer_{k}",
                no_split=True,
            )
        except Exception:
            continue
    
    return bundles




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


