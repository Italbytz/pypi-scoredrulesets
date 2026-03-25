import numpy as np
import pytest
import runpy
from pathlib import Path

from sklearn.datasets import load_iris, load_breast_cancer

from scoredrulesets import PittsburghRuleSetClassifier, ScoredRuleSetClassifier
from scoredrulesets.benchmarking.estimators import default_estimator_specs


def test_pittsburgh_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = PittsburghRuleSetClassifier(
        max_rules=4,
        candidate_pool_size=16,
        beam_width=5,
        max_iterations=8,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])
    ruleset = clf.to_ruleset()

    assert pred.shape == (10,)
    assert proba.shape == (10, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert any(rule.rule_id == "pittsburgh_default_prior" for rule in ruleset.rules)
    assert any(rule.rule_id and rule.rule_id.startswith("pittsburgh_rule_") for rule in ruleset.rules)
    assert ruleset.metadata["source"] == "pittsburgh"
    assert ruleset.metadata["selected_rule_count"] <= 4


def test_pittsburgh_estimator_handles_categorical_rules():
    X = np.array(
        [
            ["red", "small"],
            ["red", "large"],
            ["blue", "small"],
            ["blue", "large"],
            ["green", "small"],
            ["green", "large"],
            ["red", "small"],
            ["blue", "large"],
        ],
        dtype=object,
    )
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0])

    clf = PittsburghRuleSetClassifier(
        max_rules=3,
        min_samples_leaf=1,
        candidate_pool_size=12,
        beam_width=4,
        max_iterations=6,
        random_state=3,
    )
    clf.fit(X, y)

    ruleset = clf.to_ruleset()
    assert any(atom.op in {"==", "in"} for rule in ruleset.rules for atom in rule.atoms)



def test_sklearn_wrapper_pittsburgh_backend():
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="pittsburgh",
        backend_params={
            "max_rules": 4,
            "candidate_pool_size": 16,
            "beam_width": 5,
            "max_iterations": 8,
        },
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:6])
    proba = clf.predict_proba(X[:6])
    assert pred.shape == (6,)
    assert proba.shape == (6, 3)
    assert clf.to_ruleset().metadata["source"] == "pittsburgh"



def test_benchmarking_estimator_specs_include_pittsburgh():
    specs = default_estimator_specs()
    assert "wrapper_pittsburgh" in specs
    assert "wrapper_pittsburgh_fast" in specs
    assert "wrapper_pittsburgh_strong" in specs
    assert "wrapper_pittsburgh_diverse" in specs


def test_pittsburgh_benchmark_profiles_use_expected_backend_and_budget():
    specs = default_estimator_specs()

    strong = specs["wrapper_pittsburgh_strong"].factory()
    diverse = specs["wrapper_pittsburgh_diverse"].factory()

    assert strong.backend == "pittsburgh"
    assert strong.backend_params["beam_width"] >= 10
    assert strong.backend_params["candidate_pool_size"] >= 32

    assert diverse.backend == "pittsburgh"
    assert diverse.backend_params["max_rules"] >= 7
    assert diverse.backend_params["candidate_pool_size"] >= 36


def test_pittsburgh_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_pittsburgh_backend.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["profile"] == "default"
    assert result["pittsburgh"]["metadata"]["source"] == "pittsburgh"
    assert result["pittsburgh"]["n_rules"] > 0
    assert len(result["comparison"]) == 2
    assert {row["name"] for row in result["comparison"]} == {"cart", "gp"}


def test_pittsburgh_example_profile_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_pittsburgh_backend.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0, profile="strong")

    assert result["profile"] == "strong"
    meta = result["pittsburgh"]["metadata"]
    assert meta["source"] == "pittsburgh"
    assert meta["beam_width"] >= 6


def test_pittsburgh_wrapper_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_pittsburgh_wrapper.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["profile"] == "default"
    assert result["wrapper_pittsburgh"]["metadata"]["source"] == "pittsburgh"
    assert result["wrapper_pittsburgh"]["n_rules"] > 0
    assert len(result["comparison"]) == 2
    assert {row["name"] for row in result["comparison"]} == {"wrapper_cart_d2", "wrapper_cart_d4"}


def test_pittsburgh_wrapper_example_profile_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_pittsburgh_wrapper.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0, profile="strong")

    assert result["profile"] == "strong"
    meta = result["wrapper_pittsburgh"]["metadata"]
    assert meta["source"] == "pittsburgh"
    assert meta["beam_width"] >= 6


# ---------------------------------------------------------------------------
# Multi-class (OvR) Tests
# ---------------------------------------------------------------------------


class TestPittsburghOvRMulticlass:
    """Tests for the One-vs-Rest multiclass strategy."""

    def test_ovr_multiclass_iris(self):
        """OvR on Iris (3 classes): correct shapes, proba sums to 1, metadata."""
        X, y = load_iris(return_X_y=True)
        clf = PittsburghRuleSetClassifier(
            max_rules=4,
            candidate_pool_size=16,
            beam_width=5,
            max_iterations=8,
            random_state=0,
            multiclass_strategy="ovr",
        )
        clf.fit(X, y)

        pred = clf.predict(X)
        proba = clf.predict_proba(X)

        assert pred.shape == (len(y),)
        assert set(pred).issubset(set(y))
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

        ruleset = clf.to_ruleset()
        assert ruleset.metadata["multiclass_strategy"] == "ovr"
        assert ruleset.metadata["source"] == "pittsburgh"

        # At least some rules should carry ovr_class_index metadata
        ovr_rules = [
            r for r in ruleset.rules
            if r.metadata and r.metadata.get("ovr_class_index") is not None
        ]
        assert len(ovr_rules) > 0

        # Rules should cover multiple classes
        ovr_classes = {r.metadata["ovr_class_index"] for r in ovr_rules}
        assert len(ovr_classes) >= 2

    def test_ovr_binary_skips_decomposition(self):
        """With 2 classes, OvR should fall back to direct (no decomposition)."""
        X, y = load_breast_cancer(return_X_y=True)
        clf = PittsburghRuleSetClassifier(
            max_rules=3,
            candidate_pool_size=12,
            beam_width=4,
            max_iterations=6,
            random_state=0,
            multiclass_strategy="ovr",
        )
        clf.fit(X, y)

        pred = clf.predict(X[:10])
        proba = clf.predict_proba(X[:10])

        assert pred.shape == (10,)
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # Binary: no OvR decomposition, so no ovr_class_index in metadata
        ruleset = clf.to_ruleset()
        ovr_rules = [
            r for r in ruleset.rules
            if r.metadata and r.metadata.get("ovr_class_index") is not None
        ]
        assert len(ovr_rules) == 0

    def test_direct_unchanged_regression(self):
        """Default 'direct' strategy on Iris should work as before."""
        X, y = load_iris(return_X_y=True)
        clf = PittsburghRuleSetClassifier(
            max_rules=4,
            candidate_pool_size=16,
            beam_width=5,
            max_iterations=8,
            random_state=0,
            multiclass_strategy="direct",
        )
        clf.fit(X, y)

        pred = clf.predict(X[:10])
        proba = clf.predict_proba(X[:10])

        assert pred.shape == (10,)
        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert clf.to_ruleset().metadata["multiclass_strategy"] == "direct"

    def test_ovr_sequential_covering(self):
        """OvR combined with sequential covering on Iris."""
        X, y = load_iris(return_X_y=True)
        clf = PittsburghRuleSetClassifier(
            max_rules=4,
            candidate_pool_size=16,
            beam_width=5,
            max_iterations=8,
            random_state=0,
            multiclass_strategy="ovr",
            sequential_covering=True,
        )
        clf.fit(X, y)

        pred = clf.predict(X)
        proba = clf.predict_proba(X)

        assert pred.shape == (len(y),)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_ovr_via_wrapper(self):
        """OvR through the ScoredRuleSetClassifier wrapper."""
        X, y = load_iris(return_X_y=True)
        clf = ScoredRuleSetClassifier(
            backend="pittsburgh",
            backend_params={
                "max_rules": 4,
                "candidate_pool_size": 16,
                "beam_width": 5,
                "max_iterations": 8,
                "multiclass_strategy": "ovr",
            },
            random_state=0,
        )
        clf.fit(X, y)

        pred = clf.predict(X[:6])
        proba = clf.predict_proba(X[:6])
        assert pred.shape == (6,)
        assert proba.shape == (6, 3)
        assert clf.to_ruleset().metadata["multiclass_strategy"] == "ovr"

    def test_ovr_with_compaction(self):
        """OvR with post-hoc compaction enabled."""
        X, y = load_iris(return_X_y=True)
        clf = PittsburghRuleSetClassifier(
            max_rules=5,
            candidate_pool_size=20,
            beam_width=6,
            max_iterations=10,
            random_state=0,
            multiclass_strategy="ovr",
            enable_compaction=True,
        )
        clf.fit(X, y)

        pred = clf.predict(X)
        assert pred.shape == (len(y),)
        assert set(pred).issubset(set(y))

    def test_invalid_strategy_raises(self):
        """Unknown multiclass_strategy should raise ValueError."""
        X, y = load_iris(return_X_y=True)
        clf = PittsburghRuleSetClassifier(
            max_rules=3,
            random_state=0,
            multiclass_strategy="foo",
        )
        with pytest.raises(ValueError, match="multiclass_strategy"):
            clf.fit(X, y)

    def test_ovr_score_vector_length(self):
        """Each OvR rule should have a score vector of length n_classes."""
        X, y = load_iris(return_X_y=True)
        clf = PittsburghRuleSetClassifier(
            max_rules=3,
            candidate_pool_size=12,
            beam_width=4,
            max_iterations=6,
            random_state=0,
            multiclass_strategy="ovr",
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        n_classes = len(clf.classes_)
        for rule in ruleset.rules:
            assert len(rule.scores) == n_classes, (
                f"Rule {rule.rule_id} has {len(rule.scores)} scores, expected {n_classes}"
            )
