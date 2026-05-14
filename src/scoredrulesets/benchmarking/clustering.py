from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

from ..estimators.cluster_wrapper import ScoredRuleSetClusterer


@dataclass
class ClusterApproxRunResult:
    seed: int
    status: str
    error: str | None
    fit_seconds: float | None
    train_fidelity: float | None
    test_fidelity: float | None
    test_ari: float | None
    n_rules: int | None
    n_atoms: int | None


@dataclass
class ClusterApproxSummary:
    n_runs: int
    train_fidelity_mean: float | None
    train_fidelity_std: float | None
    test_fidelity_mean: float | None
    test_fidelity_std: float | None
    test_ari_mean: float | None
    test_ari_std: float | None
    n_rules_mean: float | None
    n_rules_std: float | None
    n_atoms_mean: float | None
    n_atoms_std: float | None


def benchmark_cluster_approximation(
    X: np.ndarray,
    *,
    n_clusters: int,
    seeds: list[int] | None = None,
    test_size: float = 0.3,
    clusterer_factory: Callable[[int], Any] | None = None,
    approximation_backend: str = "cart",
    approximation_backend_params: dict[str, Any] | None = None,
    transform_params: dict[str, Any] | None = None,
) -> tuple[list[ClusterApproxRunResult], ClusterApproxSummary]:
    """Benchmark cluster-label approximation quality over multiple random seeds.

    Parameters
    ----------
    X:
        Feature matrix used for clustering and approximation.
    n_clusters:
        Number of clusters used by the default KMeans factory.
    seeds:
        Random seeds for train/test splits and model initialization.
    test_size:
        Fraction reserved for test fidelity estimation.
    clusterer_factory:
        Optional factory ``seed -> clusterer``. If omitted, KMeans is used.
    approximation_backend:
        Backend passed to ``ScoredRuleSetClusterer`` for pseudo-label approximation.
    approximation_backend_params:
        Backend parameters for the approximation model.
    transform_params:
        Optional transform params forwarded to ``ScoredRuleSetClassifier``.
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")

    run_seeds = list(seeds or [0, 1, 2])
    runs: list[ClusterApproxRunResult] = []

    for seed in run_seeds:
        X_train, X_test = train_test_split(X_arr, test_size=test_size, random_state=seed)

        if clusterer_factory is None:
            clusterer = KMeans(n_clusters=int(n_clusters), random_state=seed, n_init="auto")
        else:
            clusterer = clusterer_factory(seed)

        model = ScoredRuleSetClusterer(
            clusterer=clusterer,
            n_clusters=int(n_clusters),
            approximation_backend=approximation_backend,
            approximation_backend_params=approximation_backend_params,
            transform_params=transform_params,
            random_state=seed,
        )

        started = perf_counter()
        try:
            model.fit(X_train)
            fit_seconds = perf_counter() - started

            y_cluster_test = np.asarray(model.clusterer_.predict(X_test))
            y_ruleset_test = np.asarray(model.predict(X_test))
            ruleset = model.to_ruleset()
            n_atoms = int(sum(len(rule.atoms) for rule in ruleset.rules))

            runs.append(
                ClusterApproxRunResult(
                    seed=seed,
                    status="ok",
                    error=None,
                    fit_seconds=float(fit_seconds),
                    train_fidelity=float(model.fidelity_against_clusterer(X_train)),
                    test_fidelity=float(model.fidelity_against_clusterer(X_test)),
                    test_ari=float(adjusted_rand_score(y_cluster_test, y_ruleset_test)),
                    n_rules=len(ruleset.rules),
                    n_atoms=n_atoms,
                )
            )
        except Exception as exc:
            fit_seconds = perf_counter() - started
            runs.append(
                ClusterApproxRunResult(
                    seed=seed,
                    status="error",
                    error=str(exc),
                    fit_seconds=float(fit_seconds),
                    train_fidelity=None,
                    test_fidelity=None,
                    test_ari=None,
                    n_rules=None,
                    n_atoms=None,
                )
            )

    return runs, summarize_cluster_approximation_runs(runs)


def summarize_cluster_approximation_runs(runs: list[ClusterApproxRunResult]) -> ClusterApproxSummary:
    ok_runs = [r for r in runs if r.status == "ok"]

    def _stats(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        arr = np.asarray(values, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    train_mean, train_std = _stats([r.train_fidelity for r in ok_runs if r.train_fidelity is not None])
    test_mean, test_std = _stats([r.test_fidelity for r in ok_runs if r.test_fidelity is not None])
    ari_mean, ari_std = _stats([r.test_ari for r in ok_runs if r.test_ari is not None])
    rules_mean, rules_std = _stats([float(r.n_rules) for r in ok_runs if r.n_rules is not None])
    atoms_mean, atoms_std = _stats([float(r.n_atoms) for r in ok_runs if r.n_atoms is not None])

    return ClusterApproxSummary(
        n_runs=len(ok_runs),
        train_fidelity_mean=train_mean,
        train_fidelity_std=train_std,
        test_fidelity_mean=test_mean,
        test_fidelity_std=test_std,
        test_ari_mean=ari_mean,
        test_ari_std=ari_std,
        n_rules_mean=rules_mean,
        n_rules_std=rules_std,
        n_atoms_mean=atoms_mean,
        n_atoms_std=atoms_std,
    )


def format_cluster_approximation_runs_markdown(runs: list[ClusterApproxRunResult]) -> str:
    """Render per-seed cluster approximation runs as a compact markdown table."""
    columns = [
        ("seed", "seed"),
        ("status", "status"),
        ("fit_s", "fit_seconds"),
        ("train_fidelity", "train_fidelity"),
        ("test_fidelity", "test_fidelity"),
        ("test_ari", "test_ari"),
        ("n_rules", "n_rules"),
        ("n_atoms", "n_atoms"),
        ("error", "error"),
    ]
    headers = [title for title, _ in columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for run in runs:
        row = asdict(run)
        values: list[str] = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)
