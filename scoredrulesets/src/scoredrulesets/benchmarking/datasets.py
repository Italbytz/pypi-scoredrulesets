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
        elif normalized == "pmlb":
            resolved.extend(
                name for name in registry if name.startswith("pmlb_")
            )
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


# ---------------------------------------------------------------------------
# Multiplexer-Datensaetze (boolesche Klassifikation)
# ---------------------------------------------------------------------------

_MUX_CONFIGS: tuple[tuple[str, int], ...] = (
    ("mux_6", 2),     # 2 Adressbits + 4 Datenbits  = 6 Features, 2^6 = 64 Instanzen
    ("mux_11", 3),    # 3 Adressbits + 8 Datenbits  = 11 Features, 2^11 = 2048 Instanzen
    ("mux_20", 4),    # 4 Adressbits + 16 Datenbits = 20 Features, 2^20 = 1_048_576 Inst.
)


def generate_multiplexer_dataset(
    n_address_bits: int,
    *,
    max_samples: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt einen Multiplexer-Datensatz.

    Ein k-Multiplexer hat ``k`` Adressbits und ``2**k`` Datenbits.
    Die Gesamtzahl der Features ist ``k + 2**k``.
    Der Ausgabewert ist der Datenbit an der durch die Adressbits codierten Position.

    Bei voller Enumeration enthaelt der Datensatz ``2**(k + 2**k)`` Zeilen.
    Fuer grosse k kann ``max_samples`` die Zeilenanzahl durch zufaelliges Sampling
    begrenzen.

    Parameters
    ----------
    n_address_bits : int
        Anzahl der Adressbits (1, 2, 3, ...).
    max_samples : int or None
        Maximale Zeilenanzahl. ``None`` = volle Enumeration.
    random_state : int or None
        Seed fuer reproduzierbares Sampling (nur relevant wenn ``max_samples`` gesetzt).

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
        # Binaerdarstellung der Instanz
        bits = [(idx >> b) & 1 for b in range(n_features)]
        X[row] = bits
        # Adressbits → Position im Datenbereich
        address = sum(bits[a] << a for a in range(n_address_bits))
        # Datenbit an dieser Position
        y[row] = bits[n_address_bits + address]

    return X, y


def load_multiplexer_datasets(
    *,
    max_samples_large: int = 10_000,
) -> dict[str, DatasetBundle]:
    """Erzeugt die Standard-Multiplexer-Datensaetze (mux_6 bis mux_37).

    Fuer grosse Multiplexer (>= 2^16 Instanzen) wird Sampling verwendet,
    um die Laufzeit handhabbar zu halten.

    Parameters
    ----------
    max_samples_large : int
        Maximale Zeilenanzahl fuer grosse Multiplexer (default: 10_000).
    """
    bundles: dict[str, DatasetBundle] = {}
    for name, n_addr in _MUX_CONFIGS:
        n_features = n_addr + (1 << n_addr)
        n_total = 1 << n_features
        # Sampling nur fuer grosse Datensaetze
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
# Synthetische biomedizinische / SNP-Datensaetze
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
    """Erzeugt einen synthetischen Epistasie-Datensatz (SNP-SNP-Interaktion).

    Die Klasse wird ausschliesslich durch eine Interaktion zwischen
    ``n_interacting`` SNPs bestimmt; alle anderen SNPs sind Rausch-Features.
    Das Modell emuliert das Verhalten von GAMETES: Die kausalen SNPs erzeugen
    zusammen ein XOR-aehnliches Muster (kein Haupteffekt, nur Interaktion).

    Parameters
    ----------
    n_samples : int
        Anzahl Instanzen.
    n_snps : int
        Gesamtzahl der SNP-Features (inklusive kausaler SNPs).
    n_interacting : int
        Anzahl kausaler SNPs, die interagieren (2 oder 3 empfohlen).
    heritability : float
        Staerke des genetischen Signals (0 = kein Signal, 1 = perfekt trennbar).
        Steuert die Penetranz-Tabelle.
    minor_allele_freq : float
        Minor-Allel-Frequenz fuer das Hardy-Weinberg-Gleichgewicht (0 < maf < 0.5).
    random_state : int or None
        Seed fuer Reproduzierbarkeit.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_snps), dtype int  (Werte: 0, 1, 2)
    y : np.ndarray, shape (n_samples,), dtype int  (0 oder 1)
    """
    rng = np.random.default_rng(random_state)
    assert n_interacting <= n_snps, "n_interacting darf nicht groesser als n_snps sein"
    assert 0 < minor_allele_freq < 0.5, "minor_allele_freq muss in (0, 0.5) liegen"

    # SNP-Genotypen gemaess Hardy-Weinberg-Gleichgewicht
    p = minor_allele_freq
    hw_probs = [(1 - p) ** 2, 2 * p * (1 - p), p ** 2]  # P(0), P(1), P(2)
    X = rng.choice(3, size=(n_samples, n_snps), p=hw_probs)

    # Kausale SNPs: Die ersten n_interacting Features
    causal = X[:, :n_interacting]

    # XOR-aehnliche Interaktion: Klasse = 1, wenn die Summe der kausalen
    # Allele ungerade ist (reine Epistasie, kein Haupteffekt).
    interaction_signal = np.sum(causal, axis=1) % 2  # 0 oder 1

    # Penetranz gemaess Heritabilitaet: P(krank | Signal=1) = 0.5 + h/2,
    # P(krank | Signal=0) = 0.5 - h/2.  Bei h=1 → 1.0 / 0.0 (deterministisch).
    pen_high = min(1.0, 0.5 + heritability / 2)
    pen_low = max(0.0, 0.5 - heritability / 2)
    probs = np.where(interaction_signal == 1, pen_high, pen_low)
    y = (rng.random(n_samples) < probs).astype(int)

    # Permutation der Feature-Spalten, damit die kausalen SNPs nicht immer
    # vorne stehen (sonst ist der Benchmark zu einfach).
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
    """Erzeugt einen XOR-/Paritaets-Datensatz mit Rausch-Features.

    Die Klasse ist die Paritaet (XOR) der ersten ``n_bits`` binaeren Features.
    Zusaetzlich werden ``n_noise_features`` zufaellige binaere Features angehaengt.
    Dieser Datensatz ist nicht-linear trennbar und erfordert konjunktive Regeln.

    Parameters
    ----------
    n_bits : int
        Anzahl der relevanten Paritaets-Bits.
    n_noise_features : int
        Anzahl irrelevanter Rausch-Features.
    n_samples : int
        Anzahl Instanzen.
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_bits + n_noise_features), dtype int
    y : np.ndarray, shape (n_samples,), dtype int (0 oder 1)
    """
    rng = np.random.default_rng(random_state)
    total_features = n_bits + n_noise_features
    X = rng.integers(0, 2, size=(n_samples, total_features))
    y = np.bitwise_xor.reduce(X[:, :n_bits], axis=1)

    # Spalten permutieren
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
    """Erzeugt einen hochdimensionalen Datensatz mit wenigen Samples (p >> n).

    Typisch fuer Genomik-Szenarien (z.B. Microarray, SNP-Panel): Viele Features,
    wenige Patienten, nur wenige informative Features.

    Parameters
    ----------
    n_samples : int
        Anzahl Instanzen (typischerweise klein, z.B. 80–200).
    n_features : int
        Gesamtzahl Features (typischerweise gross, z.B. 200–2000).
    n_informative : int
        Anzahl tatsaechlich informativer Features.
    n_classes : int
        Anzahl Klassen.
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
    """Erzeugt einen Datensatz mit starkem Klassen-Ungleichgewicht.

    Simuliert seltene Krankheiten oder seltene Ereignisse.

    Parameters
    ----------
    n_samples : int
        Gesamtzahl Instanzen.
    n_features : int
        Anzahl Features.
    n_informative : int
        Anzahl informativer Features.
    imbalance_ratio : float
        Anteil der Minoritaetsklasse (z.B. 0.1 = 10 %).
    random_state : int or None
        Seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), dtype int (0 oder 1)
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


# -- Standardkonfigurationen fuer synthetische Datensaetze ------------------

_SYNTH_CONFIGS: dict[str, dict] = {
    # Epistasie: 2-Weg-Interaktion, mittel
    "synth_epistasis_2way_easy": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=1600, n_snps=20, n_interacting=2, heritability=0.6, random_state=42),
    ),
    "synth_epistasis_2way_hard": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=1600, n_snps=20, n_interacting=2, heritability=0.2, random_state=42),
    ),
    # Epistasie: 3-Weg-Interaktion (schwieriger)
    "synth_epistasis_3way": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=2000, n_snps=30, n_interacting=3, heritability=0.4, random_state=42),
    ),
    # Epistasie: hochdimensional (viele Rausch-SNPs)
    "synth_epistasis_highdim": dict(
        generator=generate_epistasis_dataset,
        params=dict(n_samples=1600, n_snps=100, n_interacting=2, heritability=0.4, random_state=42),
    ),
    # XOR / Paritaet
    "synth_xor_3bit": dict(
        generator=generate_xor_parity_dataset,
        params=dict(n_bits=3, n_noise_features=7, n_samples=2000, random_state=42),
    ),
    "synth_xor_5bit": dict(
        generator=generate_xor_parity_dataset,
        params=dict(n_bits=5, n_noise_features=15, n_samples=2000, random_state=42),
    ),
    # p >> n (Genomik-Szenario)
    "synth_highdim_p500_n120": dict(
        generator=generate_highdim_lowsample_dataset,
        params=dict(n_samples=120, n_features=500, n_informative=5, random_state=42),
    ),
    "synth_highdim_p1000_n80": dict(
        generator=generate_highdim_lowsample_dataset,
        params=dict(n_samples=80, n_features=1000, n_informative=4, random_state=42),
    ),
    # Imbalanced (seltene Krankheit)
    "synth_imbalanced_10pct": dict(
        generator=generate_imbalanced_dataset,
        params=dict(n_samples=1000, n_features=12, n_informative=5, imbalance_ratio=0.1, random_state=42),
    ),
    "synth_imbalanced_5pct": dict(
        generator=generate_imbalanced_dataset,
        params=dict(n_samples=2000, n_features=15, n_informative=6, imbalance_ratio=0.05, random_state=42),
    ),
}


def load_synthetic_datasets() -> dict[str, DatasetBundle]:
    """Erzeugt alle vordefinierten synthetischen Benchmark-Datensaetze."""
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
# pmlb-Datensaetze (Penn Machine Learning Benchmarks)
# ---------------------------------------------------------------------------

# Kuratierte Auswahl biomedizinisch relevanter pmlb-Datensaetze.
# Enthaelt GAMETES-Epistasie-Datensaetze und weitere interessante Probleme.
_PMLB_DATASET_NAMES: tuple[str, ...] = (
    # GAMETES Epistasie (2-Weg, verschiedene Heritabilitaeten / EDMs)
    "GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1",
    "GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1",
    "GAMETES_Epistasis_2-Way_1000atts_0.4H_EDM-1_EDM-1_1",
    # GAMETES Epistasie (3-Weg)
    "GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1",
    # Heterogeneous (Mischung aus Haupteffekt + Interaktion)
    "GAMETES_Heterogeneous_20atts_1600_Het_0.4_0.2_50_EDM-2_001",
    # Weitere biomedizinisch relevante Datensaetze
    "saheart",       # South-African Heart Disease
    "pima",          # Pima-Indians Diabetes
    "analcatdata_aids",  # AIDS-Daten
)


def load_pmlb_datasets() -> dict[str, DatasetBundle]:
    """Laedt kuratierte Datensaetze aus der Penn Machine Learning Benchmark Suite.

    Benoetigt ``pip install pmlb``.  Gibt ein leeres Dict zurueck, wenn pmlb
    nicht installiert ist oder der Download fehlschlaegt.

    Returns
    -------
    dict[str, DatasetBundle]
        Mapping ``pmlb_<name>`` → DatasetBundle.
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


