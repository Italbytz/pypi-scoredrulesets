from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split

from scoredrulesets import ScoredRuleSetClusterer


def test_kmeans_cart_cluster_wrapper_fits_and_predicts():
    X, _ = make_blobs(
        n_samples=500,
        centers=4,
        n_features=6,
        cluster_std=0.85,
        random_state=42,
    )
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)

    model = ScoredRuleSetClusterer(
        n_clusters=4,
        approximation_backend="cart",
        approximation_backend_params={"max_depth": None},
        random_state=0,
    )
    model.fit(X_train)

    # On training data this path should be near-lossless for CART with unlimited depth.
    train_fidelity = model.fidelity_against_clusterer(X_train)
    assert train_fidelity >= 0.999

    test_fidelity = model.fidelity_against_clusterer(X_test)
    assert test_fidelity >= 0.90

    pred = model.predict(X_test[:20])
    assert pred.shape == (20,)
    assert len(model.to_ruleset().rules) > 0


def test_cluster_ruleset_roundtrip(tmp_path: Path):
    X, _ = load_iris(return_X_y=True)
    model = ScoredRuleSetClusterer(
        n_clusters=3,
        approximation_backend="cart",
        random_state=7,
    )
    model.fit(X)

    path = tmp_path / "iris_cluster_ruleset.json"
    model.save_ruleset(path)

    loaded = ScoredRuleSetClusterer.from_ruleset_json(path)
    pred_a = model.predict(X[:30])
    pred_b = loaded.predict(X[:30])

    np.testing.assert_array_equal(pred_a, pred_b)
