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
        elif normalized == "cart_hard":
            # Datensaetze, bei denen CART strukturell benachteiligt ist:
            # DNF-Konzepte, Checkerboard, MONK, ueberlappende Regeln, modulare Summe.
            _cart_hard_prefixes = (
                "synth_dnf_", "synth_checkerboard_", "synth_monk",
                "synth_overlap_", "synth_modsum_",
            )
            resolved.extend(
                name for name in registry
                if any(name.startswith(p) for p in _cart_hard_prefixes)
            )
        elif normalized == "ruleset_hard":
            # Datensaetze, bei denen Rule Sets strukturell benachteiligt
            # sind (CART ist besser): tiefe Baeume, sequentielle Splits,
            # hierarchische Interaktionen.
            _ruleset_hard_prefixes = (
                "synth_deeptree_", "synth_seqthresh_", "synth_hierarch_",
            )
            resolved.extend(
                name for name in registry
                if any(name.startswith(p) for p in _ruleset_hard_prefixes)
            )
        elif normalized == "rule_hard":
            # Datensaetze, die fuer ALLE regelbasierten Schaetzer (CART +
            # Rule Sets) schwierig sind – nicht achsenparallel trennbar.
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


# ---------------------------------------------------------------------------
# Datensaetze, die CART-Schwaechen gezielt exponieren (DNF, Checkerboard, …)
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
    """Erzeugt einen Datensatz mit einer DNF-Entscheidungsregel.

    Das wahre Konzept ist eine Disjunktion von ``n_disjuncts`` Konjunktionen,
    wobei jede Konjunktion ``n_conjuncts`` verschiedene binaere Features
    erfordert.  CART muss dafuer Teilbaeume duplizieren; Rule Sets bilden
    jedes Disjunkt direkt als separate Regel ab.

    Parameters
    ----------
    n_disjuncts : int
        Anzahl der Disjunkte (OR-Glieder).
    n_conjuncts : int
        Anzahl der Konjunkte pro Disjunkt (AND-Glieder).
    n_noise_features : int
        Zusaetzliche irrelevante binaere Features.
    n_samples : int
        Anzahl Instanzen.
    noise_rate : float
        Anteil zufaellig geflippter Labels (0 = kein Rauschen).
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

    # Label-Rauschen
    if noise_rate > 0:
        flip = rng.random(n_samples) < noise_rate
        y[flip] = 1 - y[flip]

    # Spalten permutieren
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
    """Erzeugt einen 2D-Schachbrett-Datensatz mit Rausch-Features.

    Das Schachbrett-Muster (XOR auf quantisierten kontinuierlichen Features)
    erfordert bei CART O(n_tiles^2) Blaetter, waehrend Rule Sets die
    Regionen direkt als Regeln formulieren koennen.

    Parameters
    ----------
    n_tiles : int
        Anzahl Kacheln pro Achse (z.B. 4 → 4×4-Schachbrett).
    n_noise_features : int
        Zusaetzliche Uniform-Rausch-Features.
    n_samples : int
        Anzahl Instanzen.
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
    """Erzeugt den MONK-1 Datensatz (Disjunktion: (a1==a2) OR (a5==1)).

    MONK-1 ist ein klassischer Benchmark fuer Regellerner.  Die Entscheidung
    ist eine Disjunktion, die CART nur durch Subtree-Duplikation abbilden kann.

    Attribute:
        a1 in {1,2,3}, a2 in {1,2,3}, a3 in {1,2},
        a4 in {1,2,3}, a5 in {1,2,3,4}, a6 in {1,2}

    Parameters
    ----------
    n_samples : int
        Anzahl Instanzen (Sampling mit Zuruecklegen falls > 432).
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    domains = [3, 3, 2, 3, 4, 2]  # a1..a6
    n_total = 1
    for d in domains:
        n_total *= d  # 432

    # Volle Enumeration
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

    # MONK-1 Regel: (a1 == a2) OR (a5 == 1)
    # (0-basiert: a5 == 1 → rows[:, 4] == 1)
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
    """Erzeugt den MONK-3 Datensatz (Konjunktion + Ausnahme + Rauschen).

    MONK-3 Regel: (a5 != 3 AND a4 != 1) OR (a5 == 3 AND a2 != 3),
    plus ``noise_rate`` zufaellig geflippter Labels.

    Parameters
    ----------
    n_samples : int
        Anzahl Instanzen.
    noise_rate : float
        Anteil zufaellig geflippter Labels (Standard: 5 %).
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
    # 0-basiert: a5 != 3 → rows[:, 4] != 3; a4 != 1 → rows[:, 3] != 1; usw.
    y_full = (
        ((rows[:, 4] != 3) & (rows[:, 3] != 1))
        | ((rows[:, 4] == 3) & (rows[:, 1] != 2))
    ).astype(int)

    if n_samples >= n_total:
        X, y = rows, y_full
    else:
        chosen = rng.choice(n_total, size=n_samples, replace=True)
        X, y = rows[chosen], y_full[chosen]

    # Label-Rauschen
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
    """Erzeugt einen Datensatz mit mehreren unabhaengigen, ueberlappenden Regeln.

    Jede Regel definiert eine positive Region durch einen Schwellenwert auf
    ``n_features_per_rule`` kontinuierlichen Features.  Die Klasse ist 1,
    wenn mindestens eine Regel feuert.  CART hat Schwierigkeiten, weil die
    Regeln nicht hierarchisch sind.

    Parameters
    ----------
    n_rules : int
        Anzahl unabhaengiger Regeln.
    n_features_per_rule : int
        Anzahl Features pro Regel.
    n_noise_features : int
        Zusaetzliche Rausch-Features.
    n_samples : int
        Anzahl Instanzen.
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
        # Regel: alle Features in [0.6, 1.0]
        fired = np.all(X[:, start:end] > 0.6, axis=1)
        y[fired] = 1

    # Spalten permutieren
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
    """Erzeugt einen Datensatz, bei dem die Klasse von einer modularen Summe abhaengt.

    y = (sum(X[:, :n_relevant]) mod modulus == 0).  Dieses Muster erzeugt
    nicht-achsenparallele Entscheidungsgrenzen, die CART nur ineffizient
    approximieren kann.

    Parameters
    ----------
    n_relevant : int
        Anzahl relevanter Features.
    n_noise_features : int
        Zusaetzliche Rausch-Features.
    n_samples : int
        Anzahl Instanzen.
    modulus : int
        Modulus fuer die Summenfunktion.
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
# Datensaetze, bei denen CART besser als Rule Sets abschneidet
# (tiefe Hierarchien, sequentielle Splits – natuerliche Baumstrukturen)
# ---------------------------------------------------------------------------


def generate_deep_tree_dataset(
    depth: int = 5,
    n_noise_features: int = 8,
    n_samples: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt einen Datensatz, dessen wahres Konzept ein tiefer Binaerbaum ist.

    An jedem inneren Knoten wird auf ein anderes Feature gesplittet
    (Feature i, Schwellenwert 0.5).  Das Blatt bestimmt die Klasse (0 oder 1,
    alternierend).  CART bildet diesen Baum direkt ab; Rule Sets brauchen
    2^(depth-1) Regeln, weil jeder Pfad eine separate Konjunktion ist.

    Parameters
    ----------
    depth : int
        Tiefe des wahren Baums (benoetigt mindestens ``depth`` Features).
    n_noise_features : int
        Zusaetzliche irrelevante Features.
    n_samples : int
        Anzahl Instanzen.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_relevant = depth
    n_features = n_relevant + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))

    # Baum-Traversierung: An Tiefe t wird auf Feature t gesplittet (Schwelle 0.5).
    # Der Blattpfad codiert eine Binärzahl → Klasse = Parität der Pfadrichtungen.
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
    """Erzeugt einen Datensatz mit sequentiellen Schwellenwerten auf einem Feature.

    Die Klasse haengt davon ab, in welches von ``n_bins`` Intervallen ein
    einziges kontinuierliches Feature faellt: gerade Bins → Klasse 0, ungerade
    → Klasse 1.  CART braucht nur (n_bins - 1) Splits; Rule Sets muessen
    fuer jedes Intervall eine eigene Regel mit Ober- und Untergrenze erzeugen.

    Parameters
    ----------
    n_bins : int
        Anzahl Intervalle (abwechselnd Klasse 0/1).
    n_noise_features : int
        Zusaetzliche irrelevante Features.
    n_samples : int
        Anzahl Instanzen.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_features = 1 + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    # Relevantes Feature: Spalte 0
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
    """Erzeugt einen Datensatz mit hierarchischer Feature-Interaktion.

    Erst bestimmt ein Kontext-Feature, welches Response-Feature relevant ist.
    Die Klasse haengt dann vom Wert des ausgewaehlten Response-Features ab.
    CART bildet dies natuerlich als Baum (erst split auf Kontext, dann auf
    Response); Rule Sets koennen die bedingte Relevanz nicht kompakt abbilden.

    Parameters
    ----------
    n_context_features : int
        Anzahl moeglicher Kontexte (ein Feature, n Werte).
    n_response_features : int
        Anzahl Response-Features (eins pro Kontext relevant).
    n_noise_features : int
        Zusaetzliche irrelevante Features.
    n_samples : int
        Anzahl Instanzen.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_used_responses = min(n_context_features, n_response_features)
    n_features = 1 + n_response_features + n_noise_features  # 1 Kontext + Responses + Noise
    X = rng.uniform(0, 1, size=(n_samples, n_features))

    # Kontext-Feature (Spalte 0) wird in n_context_features Bins quantisiert.
    context = np.clip(
        np.floor(X[:, 0] * n_context_features).astype(int), 0, n_context_features - 1
    )
    # Je nach Kontext ist ein anderes Response-Feature relevant.
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        resp_idx = context[i] % n_used_responses
        y[i] = int(X[i, 1 + resp_idx] > 0.5)

    perm = rng.permutation(n_features)
    X = X[:, perm]
    return X, y


# ---------------------------------------------------------------------------
# Datensaetze, die fuer ALLE regelbasierten Schaetzer schwierig sind
# (nicht achsenparallel trennbar → SVM, kNN, etc. sind besser)
# ---------------------------------------------------------------------------


def generate_circle_boundary_dataset(
    n_noise_features: int = 8,
    n_samples: int = 2000,
    radius: float = 0.7,
    noise_std: float = 0.05,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt einen Datensatz mit kreisfoermiger Entscheidungsgrenze.

    Klasse 1, wenn der Punkt innerhalb eines Kreises liegt (euklidischer
    Abstand vom Ursprung < radius).  Achsenparallele Regeln (CART und Rule
    Sets) muessen den Kreis durch viele Rechtecke approximieren.  SVM mit
    RBF-Kernel oder kNN loesen dies trivial.

    Parameters
    ----------
    n_noise_features : int
        Zusaetzliche Rausch-Features.
    n_samples : int
        Anzahl Instanzen.
    radius : float
        Radius der Kreisgrenze (Features in [0, 1], Mittelpunkt (0.5, 0.5)).
    noise_std : float
        Gauss-Rauschen auf den Radius.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    X_rel = rng.uniform(0, 1, size=(n_samples, 2))
    dist = np.sqrt((X_rel[:, 0] - 0.5) ** 2 + (X_rel[:, 1] - 0.5) ** 2)
    effective_radius = radius / 2  # skaliert auf [0, 1]-Raum
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
    """Erzeugt einen Datensatz mit diagonaler (45°) Entscheidungsgrenze.

    Die wahre Grenze ist eine Hyperebene sum(X[:, :n_relevant]) > Schwelle.
    Achsenparallele Methoden brauchen eine Treppenfunktion; lineare Modelle,
    SVM oder kNN loesen dies direkt.

    Parameters
    ----------
    n_relevant : int
        Anzahl relevanter Features (die Summe bestimmt die Klasse).
    n_noise_features : int
        Zusaetzliche irrelevante Features.
    n_samples : int
        Anzahl Instanzen.
    noise_std : float
        Gauss-Rauschen auf die Entscheidungsfunktion.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_features = n_relevant + n_noise_features
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    decision = np.sum(X[:, :n_relevant], axis=1) + rng.normal(0, noise_std, n_samples)
    threshold = n_relevant / 2.0  # Mittelpunkt
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
    """Erzeugt einen Two-Spirals-Datensatz mit Rausch-Features.

    Zwei ineinander verschlungene Spiralen sind ein klassisches Benchmark
    fuer nichtlineare Klassifizierer.  Weder CART noch Rule Sets koennen
    die Spiralform effizient approximieren, waehrend kNN und neuronale
    Netze dies gut schaffen.

    Parameters
    ----------
    n_samples : int
        Gesamtzahl Instanzen (je Haelfte pro Spirale).
    n_noise_features : int
        Zusaetzliche Rausch-Features.
    noise_std : float
        Radiales Rauschen.
    n_turns : float
        Anzahl Spiralwindungen.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2
    theta = np.sqrt(rng.uniform(0, 1, n_half)) * n_turns * 2 * np.pi

    # Spirale 1
    r1 = theta + rng.normal(0, noise_std, n_half)
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)

    # Spirale 2 (180° gedreht)
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

    # Shuffle und permutiere Spalten
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
    """Erzeugt einen Datensatz mit konzentrischen Ringen (alternierende Klassen).

    Ring i hat Klasse (i mod 2).  Das Muster erfordert kreisfoermige
    Entscheidungsgrenzen, die achsenparallele Methoden nur grob approximieren.

    Parameters
    ----------
    n_rings : int
        Anzahl Ringe.
    n_noise_features : int
        Zusaetzliche Rausch-Features.
    n_samples : int
        Anzahl Instanzen.
    noise_std : float
        Radiales Gauss-Rauschen.
    random_state : int or None
        Seed.
    """
    rng = np.random.default_rng(random_state)
    angle = rng.uniform(0, 2 * np.pi, n_samples)
    # Gleichmaessig auf Ringe verteilen
    ring_idx = rng.integers(0, n_rings, n_samples)
    # Radius: Ring i hat Radius i/n_rings (normiert)
    radius = (ring_idx + 0.5) / n_rings + rng.normal(0, noise_std, n_samples)
    X_rel = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = (ring_idx % 2).astype(int)

    X_noise = rng.uniform(-1, 1, size=(n_samples, n_noise_features))
    X = np.hstack([X_rel, X_noise])
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
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
    # --- CART-schwierige Datensaetze (Rule Sets sollten besser sein) ---
    # DNF: einfach – 3 Disjunkte mit je 2 Konjunkten
    "synth_dnf_3x2": dict(
        generator=generate_dnf_concept_dataset,
        params=dict(n_disjuncts=3, n_conjuncts=2, n_noise_features=10, n_samples=2000, random_state=42),
    ),
    # DNF: komplex – 5 Disjunkte mit je 3 Konjunkten
    "synth_dnf_5x3": dict(
        generator=generate_dnf_concept_dataset,
        params=dict(n_disjuncts=5, n_conjuncts=3, n_noise_features=15, n_samples=3000, random_state=42),
    ),
    # DNF: verrauscht
    "synth_dnf_3x2_noisy": dict(
        generator=generate_dnf_concept_dataset,
        params=dict(n_disjuncts=3, n_conjuncts=2, n_noise_features=10, n_samples=2000, noise_rate=0.05, random_state=42),
    ),
    # Checkerboard: 4×4-Schachbrett
    "synth_checkerboard_4x4": dict(
        generator=generate_checkerboard_dataset,
        params=dict(n_tiles=4, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Checkerboard: 6×6-Schachbrett (schwieriger)
    "synth_checkerboard_6x6": dict(
        generator=generate_checkerboard_dataset,
        params=dict(n_tiles=6, n_noise_features=8, n_samples=3000, random_state=42),
    ),
    # MONK-1: Disjunktion (a1==a2) OR (a5==1) – klassischer RL-Benchmark
    "synth_monk1": dict(
        generator=generate_monk1_dataset,
        params=dict(n_samples=2000, random_state=42),
    ),
    # MONK-3: Konjunktion + Ausnahme + 5 % Rauschen
    "synth_monk3": dict(
        generator=generate_monk3_dataset,
        params=dict(n_samples=2000, noise_rate=0.05, random_state=42),
    ),
    # Ueberlappende Subgruppen: 4 unabhaengige Regeln
    "synth_overlap_4rules": dict(
        generator=generate_overlapping_rules_dataset,
        params=dict(n_rules=4, n_features_per_rule=2, n_noise_features=10, n_samples=2000, random_state=42),
    ),
    # Ueberlappende Subgruppen: 6 Regeln (schwieriger)
    "synth_overlap_6rules": dict(
        generator=generate_overlapping_rules_dataset,
        params=dict(n_rules=6, n_features_per_rule=2, n_noise_features=12, n_samples=3000, random_state=42),
    ),
    # Modulare Summe: mod 3
    "synth_modsum_mod3": dict(
        generator=generate_modular_sum_dataset,
        params=dict(n_relevant=4, n_noise_features=8, n_samples=2000, modulus=3, random_state=42),
    ),
    # Modulare Summe: mod 4 (schwieriger)
    "synth_modsum_mod4": dict(
        generator=generate_modular_sum_dataset,
        params=dict(n_relevant=5, n_noise_features=10, n_samples=2500, modulus=4, random_state=42),
    ),
    # --- Ruleset-schwierige Datensaetze (CART sollte besser sein) ---
    # Tiefer Baum: depth=5 → 16 Blaetter, Rule Sets brauchen 8 Regeln
    "synth_deeptree_d5": dict(
        generator=generate_deep_tree_dataset,
        params=dict(depth=5, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Tiefer Baum: depth=7 → 64 Blaetter, noch schwieriger fuer Rule Sets
    "synth_deeptree_d7": dict(
        generator=generate_deep_tree_dataset,
        params=dict(depth=7, n_noise_features=8, n_samples=3000, random_state=42),
    ),
    # Sequentielle Schwellenwerte: 5 Bins auf einem Feature
    "synth_seqthresh_5bin": dict(
        generator=generate_sequential_threshold_dataset,
        params=dict(n_bins=5, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Sequentielle Schwellenwerte: 8 Bins (schwieriger)
    "synth_seqthresh_8bin": dict(
        generator=generate_sequential_threshold_dataset,
        params=dict(n_bins=8, n_noise_features=10, n_samples=3000, random_state=42),
    ),
    # Hierarchische Interaktion: 3 Kontexte × 3 Responses
    "synth_hierarch_3x3": dict(
        generator=generate_hierarchical_interaction_dataset,
        params=dict(n_context_features=3, n_response_features=3, n_noise_features=8, n_samples=2000, random_state=42),
    ),
    # Hierarchische Interaktion: 5 Kontexte × 5 Responses (schwieriger)
    "synth_hierarch_5x5": dict(
        generator=generate_hierarchical_interaction_dataset,
        params=dict(n_context_features=5, n_response_features=5, n_noise_features=10, n_samples=3000, random_state=42),
    ),
    # --- Schwierig fuer ALLE regelbasierten Schaetzer (SVM/kNN besser) ---
    # Kreisfoermige Grenze
    "synth_circle": dict(
        generator=generate_circle_boundary_dataset,
        params=dict(n_noise_features=8, n_samples=2000, radius=0.7, noise_std=0.03, random_state=42),
    ),
    # Kreisfoermige Grenze: starkeres Rauschen
    "synth_circle_noisy": dict(
        generator=generate_circle_boundary_dataset,
        params=dict(n_noise_features=8, n_samples=2000, radius=0.7, noise_std=0.08, random_state=42),
    ),
    # Diagonale Grenze: 4 relevante Features
    "synth_diagonal_4d": dict(
        generator=generate_diagonal_boundary_dataset,
        params=dict(n_relevant=4, n_noise_features=8, n_samples=2000, noise_std=0.1, random_state=42),
    ),
    # Diagonale Grenze: 8 relevante Features (schwieriger)
    "synth_diagonal_8d": dict(
        generator=generate_diagonal_boundary_dataset,
        params=dict(n_relevant=8, n_noise_features=8, n_samples=3000, noise_std=0.1, random_state=42),
    ),
    # Two Spirals
    "synth_spiral": dict(
        generator=generate_spiral_dataset,
        params=dict(n_samples=2000, n_noise_features=8, noise_std=0.15, n_turns=1.5, random_state=42),
    ),
    # Konzentrische Ringe: 3 Ringe
    "synth_rings_3": dict(
        generator=generate_concentric_rings_dataset,
        params=dict(n_rings=3, n_noise_features=8, n_samples=2000, noise_std=0.04, random_state=42),
    ),
    # Konzentrische Ringe: 5 Ringe (schwieriger)
    "synth_rings_5": dict(
        generator=generate_concentric_rings_dataset,
        params=dict(n_rings=5, n_noise_features=8, n_samples=3000, noise_std=0.03, random_state=42),
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


