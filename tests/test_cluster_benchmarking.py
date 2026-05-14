import numpy as np
from sklearn.datasets import make_blobs

from scoredrulesets import (
    benchmark_cluster_approximation,
    format_cluster_approximation_runs_markdown,
)


def test_cluster_benchmark_api_returns_summary_and_markdown_table():
    X, _ = make_blobs(
        n_samples=400,
        centers=4,
        n_features=5,
        cluster_std=0.9,
        random_state=42,
    )

    runs, summary = benchmark_cluster_approximation(
        X,
        n_clusters=4,
        seeds=[0, 1],
        approximation_backend="cart",
        approximation_backend_params={"max_depth": None},
    )

    assert len(runs) == 2
    assert summary.n_runs == 2
    assert summary.train_fidelity_mean is not None
    assert summary.test_fidelity_mean is not None
    assert summary.test_ari_mean is not None
    assert summary.n_rules_mean is not None
    assert summary.n_atoms_mean is not None

    # Well-separated blobs should be approximated almost perfectly.
    assert summary.train_fidelity_mean >= 0.999
    assert summary.test_fidelity_mean >= 0.95
    assert summary.test_ari_mean >= 0.95

    table = format_cluster_approximation_runs_markdown(runs)
    assert "| seed | status | fit_s |" in table
    assert "train_fidelity" in table
    assert "test_ari" in table

    # Ensure markdown contains one data row per run.
    assert table.count("| ok |") == 2

    # No NaNs should leak into successful runs.
    for run in runs:
        if run.status == "ok":
            assert run.train_fidelity is not None and np.isfinite(run.train_fidelity)
            assert run.test_fidelity is not None and np.isfinite(run.test_fidelity)
            assert run.test_ari is not None and np.isfinite(run.test_ari)
