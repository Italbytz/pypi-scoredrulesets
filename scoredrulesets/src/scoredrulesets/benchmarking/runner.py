from __future__ import annotations

import json as _json
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path
from time import perf_counter
from typing import Any
import signal
import warnings

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .datasets import load_dataset_registry, resolve_dataset_names
from .estimators import default_estimator_specs
from .metrics import model_size_metrics


class _RunTimeoutError(Exception):
    """Raised when a single benchmark run exceeds the configured timeout."""


@dataclass
class BenchmarkConfig:
    dataset_names: list[str] | None = None
    estimator_names: list[str] | None = None
    test_size: float = 0.3
    use_paper_split_policy: bool = False
    include_online_uci: bool = True
    include_synthetic: bool = True
    include_pmlb: bool = False
    paper_uci_strict: bool = False
    repeats: int = 1
    random_state: int = 0
    show_progress: bool = False
    timeout_seconds: float | None = 300.0  # max Sekunden pro Einzellauf (None = kein Limit)
    checkpoint_path: str | Path | None = None  # JSONL-Datei fuer Checkpoint/Resume (None = deaktiviert)


@dataclass
class BenchmarkResult:
    dataset: str
    estimator: str
    repeat: int
    status: str
    skip_reason: str | None
    error: str | None
    f1_macro: float | None
    fit_seconds: float | None
    predict_seconds: float | None
    n_rules: int | None
    n_atoms: int | None
    ruleset_json_bytes: int | None
    n_train: int | None
    n_test: int | None
    n_classes: int | None = None
    validation_action: str | None = None
    validation_message: str | None = None


@dataclass
class AggregatedBenchmarkResult:
    dataset: str
    estimator: str
    n_repeats: int
    status: str
    f1_macro_mean: float | None
    f1_macro_error: float | None
    fit_seconds_mean: float | None
    fit_seconds_error: float | None
    predict_seconds_mean: float | None
    predict_seconds_error: float | None
    n_rules_mean: float | None
    n_rules_error: float | None
    n_atoms_mean: float | None
    n_atoms_error: float | None
    ruleset_json_bytes_mean: float | None
    ruleset_json_bytes_error: float | None
    validation_warning_count: int = 0
    validation_warning_example: str | None = None


def run_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    dataset_registry = load_dataset_registry(
        include_online_uci=bool(config.include_online_uci),
        include_synthetic=bool(config.include_synthetic),
        include_pmlb=bool(config.include_pmlb),
    )
    estimator_registry = default_estimator_specs()

    dataset_names = resolve_dataset_names(
        config.dataset_names,
        dataset_registry,
        paper_uci_strict=bool(config.paper_uci_strict),
    )
    estimator_names = config.estimator_names or list(estimator_registry.keys())

    _validate_names("dataset", dataset_names, dataset_registry)
    _validate_names("estimator", estimator_names, estimator_registry)

    total_repeats = max(1, int(config.repeats))

    # -- Checkpoint laden (Resume) -------------------------------------------
    checkpoint_path: Path | None = None
    if config.checkpoint_path is not None:
        checkpoint_path = Path(config.checkpoint_path)

    existing_results: list[BenchmarkResult] = []
    done_keys: set[tuple[str, str, int]] = set()
    if checkpoint_path is not None:
        existing_results, done_keys = load_checkpoint_results(checkpoint_path)
        if done_keys:
            print(
                f"[checkpoint] {len(done_keys)} fertige Laeufe aus {checkpoint_path} geladen, "
                f"ueberspringe diese.",
                flush=True,
            )

    total_runs = len(dataset_names) * len(estimator_names) * total_repeats
    skipped_runs = 0
    overall_start = perf_counter()
    completed_runs = 0

    if config.show_progress:
        _print_progress_header(
            dataset_names=dataset_names,
            estimator_names=estimator_names,
            total_repeats=total_repeats,
            total_runs=total_runs,
        )

    new_results: list[BenchmarkResult] = []
    for dataset_name in dataset_names:
        bundle = dataset_registry[dataset_name]
        test_size = _resolve_test_size(bundle, config)
        is_no_split = getattr(bundle, "no_split", False)
        for repeat in range(total_repeats):
            split_seed = int(config.random_state + repeat)
            if is_no_split:
                X_train = X_test = bundle.X
                y_train = y_test = bundle.y
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    bundle.X,
                    bundle.y,
                    test_size=test_size,
                    random_state=split_seed,
                    stratify=bundle.y,
                )

            for estimator_name in estimator_names:
                run_key = (dataset_name, estimator_name, repeat)
                run_number = completed_runs + skipped_runs + 1

                # -- Resume: Lauf ueberspringen wenn schon im Checkpoint ----
                if run_key in done_keys:
                    skipped_runs += 1
                    if config.show_progress:
                        print(
                            f"[progress {run_number}/{total_runs}] SKIP (checkpoint) "
                            f"dataset={dataset_name} repeat={repeat + 1}/{total_repeats} "
                            f"estimator={estimator_name}",
                            flush=True,
                        )
                    continue

                spec = estimator_registry[estimator_name]
                run_started = perf_counter()
                n_classes = int(len(np.unique(bundle.y)))
                if config.show_progress:
                    _print_progress_start(
                        run_number=run_number,
                        total_runs=total_runs,
                        dataset_name=dataset_name,
                        repeat=repeat + 1,
                        total_repeats=total_repeats,
                        estimator_name=spec.name,
                        n_train=len(y_train),
                        n_test=len(y_test),
                        n_classes=n_classes,
                    )
                result = _run_single(
                    estimator_name=spec.name,
                    estimator_factory=spec.factory,
                    dataset_name=dataset_name,
                    repeat=repeat,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    timeout_seconds=config.timeout_seconds,
                    is_no_split=is_no_split,
                )
                new_results.append(result)
                completed_runs += 1

                # -- Checkpoint sofort schreiben -----------------------------
                if checkpoint_path is not None:
                    _append_checkpoint(checkpoint_path, result)

                if config.show_progress:
                    _print_progress_end(
                        result=result,
                        run_number=run_number,
                        total_runs=total_runs,
                        run_elapsed=perf_counter() - run_started,
                        total_elapsed=perf_counter() - overall_start,
                        completed_runs=completed_runs,
                    )

    if skipped_runs > 0:
        print(
            f"[checkpoint] {skipped_runs} Laeufe uebersprungen, "
            f"{completed_runs} neue Laeufe durchgefuehrt.",
            flush=True,
        )

    return existing_results + new_results


# ---------------------------------------------------------------------------
# Checkpoint: JSONL-basiertes Speichern und Laden
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION = 1


def load_checkpoint_results(
    path: str | Path,
) -> tuple[list[BenchmarkResult], set[tuple[str, str, int]]]:
    """Laedt Ergebnisse aus einer JSONL-Checkpoint-Datei.

    Returns
    -------
    results : list[BenchmarkResult]
        Alle geladenen Ergebnisse.
    done_keys : set[tuple[str, str, int]]
        Set der fertigen ``(dataset, estimator, repeat)``-Schluessel.
    """
    path = Path(path)
    results: list[BenchmarkResult] = []
    done_keys: set[tuple[str, str, int]] = set()
    if not path.exists():
        return results, done_keys

    valid_fields = {f.name for f in dataclass_fields(BenchmarkResult)}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = _json.loads(line)
                # Metadaten-Felder entfernen, die nicht zum Dataclass gehoeren
                d.pop("_checkpoint_version", None)
                d.pop("_checkpoint_ts", None)
                # Unbekannte Felder entfernen (Vorwaertskompatibilitaet)
                filtered = {k: v for k, v in d.items() if k in valid_fields}
                result = BenchmarkResult(**filtered)
                results.append(result)
                done_keys.add((result.dataset, result.estimator, result.repeat))
            except Exception as exc:
                warnings.warn(
                    f"[checkpoint] Zeile {line_no} in {path} uebersprungen "
                    f"(moeglicherweise korrupt): {exc}",
                    stacklevel=2,
                )
    return results, done_keys


def _append_checkpoint(path: Path, result: BenchmarkResult) -> None:
    """Haengt ein einzelnes BenchmarkResult als JSON-Zeile an die Checkpoint-Datei an."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(result)
    d["_checkpoint_version"] = _CHECKPOINT_VERSION
    d["_checkpoint_ts"] = perf_counter()  # monotoner Zeitstempel fuer Sortierung
    with path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(d, default=str) + "\n")
        f.flush()


def _resolve_test_size(bundle, config: BenchmarkConfig) -> float:
    if not config.use_paper_split_policy:
        return float(config.test_size)

    n_obs = int(bundle.X.shape[0])
    if n_obs < 500:
        return 0.30
    if n_obs < 5000:
        return 0.25
    return 0.20


def results_as_dicts(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def aggregate_benchmark_results(
    results: list[BenchmarkResult],
    error_bar: str = "std",
) -> list[AggregatedBenchmarkResult]:
    if error_bar not in {"std", "sem"}:
        raise ValueError("Invalid error_bar. Expected one of: 'std', 'sem'.")

    groups: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        if result.status != "ok":
            continue
        groups.setdefault((result.dataset, result.estimator), []).append(result)

    aggregated: list[AggregatedBenchmarkResult] = []
    for (dataset, estimator), group in sorted(groups.items()):
        aggregated.append(
            AggregatedBenchmarkResult(
                dataset=dataset,
                estimator=estimator,
                n_repeats=len(group),
                status="ok",
                f1_macro_mean=_metric_mean(group, "f1_macro"),
                f1_macro_error=_metric_error(group, "f1_macro", error_bar),
                fit_seconds_mean=_metric_mean(group, "fit_seconds"),
                fit_seconds_error=_metric_error(group, "fit_seconds", error_bar),
                predict_seconds_mean=_metric_mean(group, "predict_seconds"),
                predict_seconds_error=_metric_error(group, "predict_seconds", error_bar),
                n_rules_mean=_metric_mean(group, "n_rules"),
                n_rules_error=_metric_error(group, "n_rules", error_bar),
                n_atoms_mean=_metric_mean(group, "n_atoms"),
                n_atoms_error=_metric_error(group, "n_atoms", error_bar),
                ruleset_json_bytes_mean=_metric_mean(group, "ruleset_json_bytes"),
                ruleset_json_bytes_error=_metric_error(group, "ruleset_json_bytes", error_bar),
                validation_warning_count=sum(1 for item in group if item.validation_action == "warn"),
                validation_warning_example=next(
                    (item.validation_message for item in group if item.validation_action == "warn" and item.validation_message),
                    None,
                ),
            )
        )
    return aggregated


def aggregated_results_as_dicts(
    results: list[AggregatedBenchmarkResult],
) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def build_benchmark_leaderboard(
    results: list[AggregatedBenchmarkResult],
    primary_metric: str = "f1_macro_mean",
) -> list[AggregatedBenchmarkResult]:
    valid_metrics = {
        "f1_macro_mean",
        "fit_seconds_mean",
        "predict_seconds_mean",
        "n_rules_mean",
        "n_atoms_mean",
        "ruleset_json_bytes_mean",
    }
    if primary_metric not in valid_metrics:
        raise ValueError(
            f"Invalid primary_metric '{primary_metric}'. Expected one of {sorted(valid_metrics)}"
        )

    reverse = primary_metric == "f1_macro_mean"

    def _sort_key(result: AggregatedBenchmarkResult):
        primary_value = getattr(result, primary_metric)
        primary = float("-inf") if primary_value is None else float(primary_value)
        if not reverse:
            primary = -primary
        return (
            -primary,
            float(result.fit_seconds_mean or float("inf")),
            float(result.n_rules_mean or float("inf")),
            float(result.n_atoms_mean or float("inf")),
            result.dataset,
            result.estimator,
        )

    return sorted(results, key=_sort_key)


def compute_pareto_front(
    results: list[AggregatedBenchmarkResult],
    *,
    quality_attr: str = "f1_macro_mean",
    size_attr: str = "n_atoms_mean",
) -> list[AggregatedBenchmarkResult]:
    """Return the Pareto-optimal subset w.r.t. quality (higher=better) and size (lower=better).

    A result is Pareto-optimal if no other result has both strictly better quality
    and strictly smaller (or equal) size.
    """
    valid = [
        r for r in results
        if getattr(r, quality_attr) is not None and getattr(r, size_attr) is not None
    ]
    if not valid:
        return []

    # Sort by quality descending, size ascending as tiebreaker
    valid.sort(
        key=lambda r: (-float(getattr(r, quality_attr)), float(getattr(r, size_attr)))
    )

    pareto: list[AggregatedBenchmarkResult] = []
    min_size = float("inf")
    for r in valid:
        size = float(getattr(r, size_attr))
        if size < min_size:
            pareto.append(r)
            min_size = size

    return pareto


def build_pareto_per_dataset(
    results: list[AggregatedBenchmarkResult],
    *,
    quality_attr: str = "f1_macro_mean",
    size_attr: str = "n_atoms_mean",
) -> dict[str, list[AggregatedBenchmarkResult]]:
    """Compute the Pareto front per dataset.

    Returns a dict mapping dataset name to the list of Pareto-optimal results.
    """
    grouped: dict[str, list[AggregatedBenchmarkResult]] = {}
    for r in results:
        grouped.setdefault(r.dataset, []).append(r)

    return {
        dataset: compute_pareto_front(
            group, quality_attr=quality_attr, size_attr=size_attr
        )
        for dataset, group in grouped.items()
    }


# -- Backends, die validation_fraction als Constructor-Parameter akzeptieren --
_BACKENDS_WITH_VALIDATION_FRACTION = frozenset(
    {"rulelcs", "rulegp", "rulenln", "rulekit_native", "logicgp"}
)


def _disable_validation_fraction(estimator: object) -> None:
    """Set ``validation_fraction=0.0`` on estimators that support it.

    Called for *no_split* datasets (e.g. MUX) where ``X_train == X_test`` and
    holding out a validation set would unnecessarily reduce training data.

    Works for both :class:`ScoredRuleSetClassifier` (patches ``backend_params``)
    and direct backend estimators (patches the attribute).
    """
    # ScoredRuleSetClassifier: patch backend_params before fit() builds the
    # backend estimator.
    backend = getattr(estimator, "backend", None)
    if backend is not None and str(backend).lower() in _BACKENDS_WITH_VALIDATION_FRACTION:
        bp = dict(getattr(estimator, "backend_params", None) or {})
        bp["validation_fraction"] = 0.0
        estimator.backend_params = bp  # type: ignore[attr-defined]
        return

    # Direct backend estimator (e.g. RuleLCSClassifier used without
    # the wrapper): patch the attribute directly.
    if hasattr(estimator, "validation_fraction"):
        estimator.validation_fraction = 0.0  # type: ignore[attr-defined]


def _run_single(
    estimator_name: str,
    estimator_factory,
    dataset_name: str,
    repeat: int,
    X_train,
    X_test,
    y_train,
    y_test,
    timeout_seconds: float | None = None,
    is_no_split: bool = False,
) -> BenchmarkResult:
    """Fuehrt einen einzelnen Benchmark-Lauf aus.

    Faengt **alle** Fehler ab (Timeout, RuntimeError, ImportError, etc.) und
    gibt immer ein BenchmarkResult zurueck – der aeussere Loop crasht nie.
    """
    # -- Timeout-Setup (nur UNIX/macOS, Signal-basiert) -----------------------
    _alarm_was_set = False
    _old_handler = None

    def _timeout_handler(signum, frame):
        raise _RunTimeoutError(
            f"Timeout nach {timeout_seconds:.0f}s fuer '{estimator_name}' "
            f"auf '{dataset_name}'"
        )

    if timeout_seconds is not None and timeout_seconds > 0:
        try:
            _old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout_seconds))
            _alarm_was_set = True
        except (AttributeError, OSError):
            # Windows oder kein SIGALRM verfuegbar → kein Timeout
            pass

    try:
        return _run_single_inner(
            estimator_name=estimator_name,
            estimator_factory=estimator_factory,
            dataset_name=dataset_name,
            repeat=repeat,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            is_no_split=is_no_split,
        )
    except _RunTimeoutError as exc:
        print(f"[TIMEOUT] {exc}", flush=True)
        return BenchmarkResult(
            dataset=dataset_name,
            estimator=estimator_name,
            repeat=repeat,
            status="timeout",
            skip_reason=str(exc),
            error=str(exc),
            f1_macro=None,
            fit_seconds=None,
            predict_seconds=None,
            n_rules=None,
            n_atoms=None,
            ruleset_json_bytes=None,
            n_train=len(y_train),
            n_test=len(y_test),
        )
    except Exception as exc:
        # Generischer Fang: ImportError, RuntimeError, ValueError, ...
        # → Lauf als "error" markieren, aber weiter zum naechsten Lauf.
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {estimator_name} | {dataset_name} | {error_msg}", flush=True)
        return BenchmarkResult(
            dataset=dataset_name,
            estimator=estimator_name,
            repeat=repeat,
            status="error",
            skip_reason=None,
            error=error_msg,
            f1_macro=None,
            fit_seconds=None,
            predict_seconds=None,
            n_rules=None,
            n_atoms=None,
            ruleset_json_bytes=None,
            n_train=len(y_train),
            n_test=len(y_test),
        )
    finally:
        if _alarm_was_set:
            signal.alarm(0)  # Alarm aufheben
            if _old_handler is not None:
                signal.signal(signal.SIGALRM, _old_handler)


def _run_single_inner(
    estimator_name: str,
    estimator_factory,
    dataset_name: str,
    repeat: int,
    X_train,
    X_test,
    y_train,
    y_test,
    is_no_split: bool = False,
) -> BenchmarkResult:
    try:
        estimator = estimator_factory()
    except Exception as exc:
        raise RuntimeError(
            f"Schätzer '{estimator_name}' konnte nicht initialisiert werden: {exc}"
        ) from exc

    # -- no_split-Datensätze: validation_fraction auf 0.0 setzen, damit
    #    Estimatoren 100 % der Daten zum Training nutzen (X_train == X_test).
    if is_no_split:
        _disable_validation_fraction(estimator)

    try:
        fit_start = perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_start

        # Nach dem Training: F1-Score des nativen Modells (vor Transformation), falls möglich
        f1_native = None
        try:
            native_model = getattr(estimator, 'estimator_', None)
            if native_model is not None and hasattr(native_model, 'predict'):
                y_pred_native = native_model.predict(X_test)
                f1_native = _compute_macro_f1_robust(y_test, y_pred_native)
        except Exception:
            pass

        predict_start = perf_counter()
        y_pred = estimator.predict(X_test)
        predict_seconds = perf_counter() - predict_start

        f1_macro = _compute_macro_f1_robust(y_test, y_pred)
        ruleset = estimator.to_ruleset()
        n_rules, n_atoms, ruleset_json_bytes = model_size_metrics(ruleset)
        n_classes = int(len(np.unique(np.concatenate([y_train, y_test]))))

        # Kompakte Konsolenausgabe
        native_info = f" (nativ={f1_native:.4f})" if f1_native is not None else ""
        print(
            f"[BENCHMARK] {estimator_name} | {dataset_name} | "
            f"F1={f1_macro:.4f}{native_info} | "
            f"Klassen={n_classes} | Regeln={n_rules} | Atome={n_atoms} | "
            f"fit={fit_seconds:.2f}s",
            flush=True,
        )

        # F1-Validierung: staged policy (warn vs abort) fuer Transformationen
        is_lossy = getattr(estimator, "transformation_lossy_", False)
        validation_action, validation_message = _evaluate_transformation_gap(
            estimator_name=estimator_name,
            is_lossy=is_lossy,
            f1_native=f1_native,
            f1_transformed=f1_macro,
            dataset_name=dataset_name,
        )
        if validation_action == "warn" and validation_message:
            warnings.warn(validation_message, UserWarning)
            print(f"[WARNING] {validation_message}", flush=True)
        elif validation_action == "abort" and validation_message:
            raise RuntimeError(validation_message)

        return BenchmarkResult(
            dataset=dataset_name,
            estimator=estimator_name,
            repeat=repeat,
            status="ok",
            skip_reason=None,
            error=None,
            f1_macro=f1_macro,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            n_rules=n_rules,
            n_atoms=n_atoms,
            ruleset_json_bytes=ruleset_json_bytes,
            n_train=len(y_train),
            n_test=len(y_test),
            n_classes=n_classes,
            validation_action=validation_action,
            validation_message=validation_message,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Fehlende Abhängigkeit für '{estimator_name}': {exc}. "
            f"Tipp: pip install 'scoredrulesets[all]' installiert alle optionalen Backends."
        ) from exc
    except RuntimeError:
        raise  # F1-Validierung und Init-Fehler durchlassen
    except Exception as exc:
        raise RuntimeError(
            f"Fehler bei '{estimator_name}' auf '{dataset_name}': {exc}"
        ) from exc


def _compute_macro_f1_robust(y_true, y_pred) -> float:
    """Compute macro-F1 robustly across common label format mismatches.

    Some backends return numerically equivalent labels with different types
    (e.g. 1 vs 1.0). A plain string-cast would treat them as different
    classes and can produce artificial F1=0.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)

    # Fast path: raw labels already compatible
    try:
        return float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    except Exception:
        pass

    # Numeric-coercion path (handles int vs float label representations)
    try:
        y_true_num = y_true_arr.astype(float)
        y_pred_num = y_pred_arr.astype(float)
        return float(f1_score(y_true_num, y_pred_num, average="macro"))
    except Exception:
        pass

    # Final fallback: string normalization
    y_true_str = y_true_arr.astype(str)
    y_pred_str = y_pred_arr.astype(str)
    return float(f1_score(y_true_str, y_pred_str, average="macro"))


def _evaluate_transformation_gap(
    *,
    estimator_name: str,
    is_lossy: bool,
    f1_native: float | None,
    f1_transformed: float,
    dataset_name: str,
) -> tuple[str, str | None]:
    """Return (action, message) for transformation-gap validation.

    Actions:
      - "ok": no issue
      - "warn": print a strong warning but continue
      - "abort": raise RuntimeError in caller

    For most non-lossy transformations we keep the existing strict policy.
    For ExSTraCS shrinking variants, moderate gaps only warn because the
    shrinking step is allowed to simplify the ruleset slightly, but very large
    gaps still abort.
    """
    if is_lossy or f1_native is None or f1_native <= 0.0:
        return "ok", None

    f1_drop = float(f1_native - f1_transformed)
    if f1_drop <= 0.0:
        return "ok", None

    rel_drop = f1_drop / float(f1_native)
    base_message = (
        f"F1-Validierung für '{estimator_name}' auf '{dataset_name}': "
        f"F1 nativ={f1_native:.4f} → transformiert={f1_transformed:.4f} "
        f"(Verlust={f1_drop:.4f}, {rel_drop*100:.1f}%)."
    )

    # ExSTraCS shrinking/pruning: moderate degradation => warning, only large degradation => abort.
    if estimator_name.startswith("wrapper_exstracs_shrink") or estimator_name in (
        "wrapper_exstracs_pruned", "wrapper_exstracs_compact",
    ):
        warn_abs_drop = 0.10
        warn_rel_drop = 0.15
        abort_abs_drop = 0.25
        abort_rel_drop = 0.40
        if f1_drop > abort_abs_drop and rel_drop > abort_rel_drop:
            return "abort", base_message + " Die Shrinking-Transformation hat die Vorhersagequalität stark zerstört."
        if f1_drop > warn_abs_drop and rel_drop > warn_rel_drop:
            return "warn", base_message + " Deutliche Warnung: ExSTraCS-Shrinking weicht merklich vom nativen Modell ab."
        return "ok", None

    # Standard policy for non-lossy transformations.
    max_abs_drop = 0.10
    max_rel_drop = 0.20
    if f1_drop > max_abs_drop and rel_drop > max_rel_drop:
        return "abort", base_message + " Die Transformation hat die Vorhersagequalität zerstört."

    return "ok", None


def _validate_names(kind: str, names: list[str], registry: dict[str, Any]) -> None:
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(
            f"Unknown {kind} names: {unknown}. Available: {sorted(registry.keys())}"
        )


def _metric_mean(results: list[BenchmarkResult], attr: str) -> float | None:
    values = [getattr(result, attr) for result in results if getattr(result, attr) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _metric_error(results: list[BenchmarkResult], attr: str, error_bar: str) -> float | None:
    values = [float(getattr(result, attr)) for result in results if getattr(result, attr) is not None]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    std = float(np.std(values, ddof=1))
    if error_bar == "std":
        return std
    return float(std / np.sqrt(len(values)))


def _print_progress_header(
    dataset_names: list[str],
    estimator_names: list[str],
    total_repeats: int,
    total_runs: int,
) -> None:
    slow_estimators = [
        name
        for name in estimator_names
        if name.startswith("gp") or "logicgp" in name or "exstracs" in name
    ]
    print(
        "[progress] Benchmark started: "
        f"datasets={len(dataset_names)}, estimators={len(estimator_names)}, "
        f"repeats={total_repeats}, total_runs={total_runs}",
        flush=True,
    )
    if slow_estimators:
        print(
            "[progress] Potentially slow estimators detected: "
            + ", ".join(slow_estimators),
            flush=True,
        )


def _print_progress_start(
    run_number: int,
    total_runs: int,
    dataset_name: str,
    repeat: int,
    total_repeats: int,
    estimator_name: str,
    n_train: int,
    n_test: int,
    n_classes: int = 0,
) -> None:
    print(
        f"[progress {run_number}/{total_runs}] START "
        f"dataset={dataset_name} repeat={repeat}/{total_repeats} "
        f"estimator={estimator_name} train={n_train} test={n_test} classes={n_classes}",
        flush=True,
    )


def _print_progress_end(
    result: BenchmarkResult,
    run_number: int,
    total_runs: int,
    run_elapsed: float,
    total_elapsed: float,
    completed_runs: int,
) -> None:
    remaining_runs = max(0, total_runs - completed_runs)
    avg_run_seconds = total_elapsed / max(1, completed_runs)
    eta_seconds = remaining_runs * avg_run_seconds
    status = result.status.upper()

    parts = [
        f"[progress {run_number}/{total_runs}] DONE status={status}",
        f"run={_format_duration(run_elapsed)}",
        f"total={_format_duration(total_elapsed)}",
        f"eta={_format_duration(eta_seconds)}",
    ]
    if result.status == "ok":
        parts.extend(
            [
                f"f1={float(result.f1_macro):.4f}",
                f"fit={_format_duration(float(result.fit_seconds or 0.0))}",
                f"predict={_format_duration(float(result.predict_seconds or 0.0))}",
                f"rules={int(result.n_rules or 0)}",
                f"atoms={int(result.n_atoms or 0)}",
            ]
        )
    elif result.skip_reason:
        parts.append(f"reason={result.skip_reason}")
    if result.error:
        parts.append(f"error={_shorten_text(result.error, max_len=160)}")
    print(" | ".join(parts), flush=True)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m {sec:04.1f}s"


def _shorten_text(value: str, max_len: int = 120) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 1] + "…"

