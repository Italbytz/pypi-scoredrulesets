import numpy as np
import pytest
from sklearn.datasets import make_regression

from scoredrulesets import ScoredRuleSetRegressor


def _make_small_regression_dataset(seed: int = 0):
    X, y = make_regression(
        n_samples=120,
        n_features=6,
        n_informative=4,
        noise=8.0,
        random_state=seed,
    )
    return X, y


@pytest.mark.parametrize(
    ("backend", "backend_params"),
    [
        (
            "rulegp",
            {
                "atom_space_strategy": "hybrid",
                "max_generations": 20,
                "stagnation_generations": 6,
                "population_size": 30,
                "n_adaptations_per_gen": 10,
            },
        ),
        (
            "rulensga2",
            {
                "population_size": 20,
                "generations": 8,
                "max_rules": 5,
                "max_atoms_per_rule": 3,
            },
        ),
    ],
)
def test_regression_projection_backends_end_to_end(backend: str, backend_params: dict):
    X, y = _make_small_regression_dataset(seed=7)
    reg = ScoredRuleSetRegressor(
        backend=backend,
        backend_params=backend_params,
        target_bins=6,
        random_state=0,
    )

    try:
        reg.fit(X, y)
    except ImportError as exc:
        pytest.skip(f"Backend '{backend}' dependencies unavailable: {exc}")

    pred = reg.predict(X[:20])
    ruleset = reg.to_ruleset()

    assert pred.shape == (20,)
    assert np.isfinite(pred).all()
    assert float(np.std(pred)) > 0.0

    assert ruleset.task_type == "regression"
    assert ruleset.metadata is not None
    assert ruleset.metadata.get("source_backend") == backend
    assert "target_bin_centers" in ruleset.metadata
    assert len(ruleset.rules) > 0
