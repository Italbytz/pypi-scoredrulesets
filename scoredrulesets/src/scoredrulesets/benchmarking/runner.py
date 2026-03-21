from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .datasets import load_dataset_registry
from .estimators import default_estimator_specs
from .metrics import model_size_metrics


@dataclass
class BenchmarkConfig:
    dataset_names: list[str] | None = None
    estimator_names: list[str] | None = None
    test_size: float = 0.3
    repeats: int = 1
    random_state: int = 0


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


def run_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    dataset_registry = load_dataset_registry()
    estimator_registry = default_estimator_specs()

    dataset_names = config.dataset_names or list(dataset_registry.keys())
    estimator_names = config.estimator_names or list(estimator_registry.keys())

    _validate_names("dataset", dataset_names, dataset_registry)
    _validate_names("estimator", estimator_names, estimator_registry)

    results: list[BenchmarkResult] = []
    for dataset_name in dataset_names:
        bundle = dataset_registry[dataset_name]
        for repeat in range(max(1, int(config.repeats))):
            split_seed = int(config.random_state + repeat)
            X_train, X_test, y_train, y_test = train_test_split(
                bundle.X,
                bundle.y,
                test_size=config.test_size,
                random_state=split_seed,
                stratify=bundle.y,
            )

            for estimator_name in estimator_names:
                spec = estimator_registry[estimator_name]
                result = _run_single(
                    estimator_name=spec.name,
                    estimator_factory=spec.factory,
                    dataset_name=dataset_name,
                    repeat=repeat,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
                results.append(result)
    return results


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


def _run_single(
    estimator_name: str,
    estimator_factory,
    dataset_name: str,
    repeat: int,
    X_train,
    X_test,
    y_train,
    y_test,
) -> BenchmarkResult:
    try:
        estimator = estimator_factory()
    except Exception as exc:
        return BenchmarkResult(
            dataset=dataset_name,
            estimator=estimator_name,
            repeat=repeat,
            status="skipped",
            skip_reason="init_failed",
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

    try:
        fit_start = perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_start

        predict_start = perf_counter()
        y_pred = estimator.predict(X_test)
        predict_seconds = perf_counter() - predict_start

        y_true_norm = np.asarray(y_test, dtype=object).astype(str)
        y_pred_norm = np.asarray(y_pred, dtype=object).astype(str)
        f1_macro = float(f1_score(y_true_norm, y_pred_norm, average="macro"))
        ruleset = estimator.to_ruleset()
        n_rules, n_atoms, ruleset_json_bytes = model_size_metrics(ruleset)

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
        )
    except ImportError as exc:
        return BenchmarkResult(
            dataset=dataset_name,
            estimator=estimator_name,
            repeat=repeat,
            status="skipped",
            skip_reason="missing_optional_dependency",
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
        return BenchmarkResult(
            dataset=dataset_name,
            estimator=estimator_name,
            repeat=repeat,
            status="error",
            skip_reason=None,
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


