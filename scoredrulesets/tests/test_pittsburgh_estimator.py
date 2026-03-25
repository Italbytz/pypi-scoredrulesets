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
    assert "wrapper_pittsburgh_strong" in specs
    assert "wrapper_pittsburgh_ovr" in specs


def test_pittsburgh_benchmark_profiles_use_expected_backend_and_budget():
    specs = default_estimator_specs()

    base = specs["wrapper_pittsburgh"].factory()
    strong = specs["wrapper_pittsburgh_strong"].factory()
    ovr = specs["wrapper_pittsburgh_ovr"].factory()

    assert base.backend == "pittsburgh"
    assert base.backend_params["candidate_pool_size"] >= 48
    assert base.backend_params["max_rules"] >= 10

    assert strong.backend == "pittsburgh"
    assert strong.backend_params["beam_width"] >= 16
    assert strong.backend_params["candidate_pool_size"] >= 64
    assert strong.backend_params["sequential_covering"] is True

    assert ovr.backend == "pittsburgh"
    assert ovr.backend_params["multiclass_strategy"] == "ovr"


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


class TestPittsburghLowCardinality:
    """Tests for low-cardinality detection (Fix 1).

    Integer-encoded categorical features (≤ k unique values) should
    produce equality splits (== value) in addition to threshold splits.
    """

    @staticmethod
    def _monk_like_data(n=200, rng_seed=42):
        """Generate MONK-style data: all features are small integers."""
        rng = np.random.default_rng(rng_seed)
        # 6 features with cardinalities 3, 3, 2, 3, 4, 2  (like MONK)
        X = np.column_stack([
            rng.integers(1, 4, size=n),   # {1,2,3}
            rng.integers(1, 4, size=n),   # {1,2,3}
            rng.integers(1, 3, size=n),   # {1,2}
            rng.integers(1, 4, size=n),   # {1,2,3}
            rng.integers(1, 5, size=n),   # {1,2,3,4}
            rng.integers(1, 3, size=n),   # {1,2}
        ])
        # Label depends on single feature equality: class 1 iff f0 == 2
        y = (X[:, 0] == 2).astype(int)
        return X, y

    def test_equality_splits_generated(self):
        """Low-cardinality numeric features should produce == rules."""
        X, y = self._monk_like_data()
        clf = PittsburghRuleSetClassifier(
            max_rules=6,
            candidate_pool_size=32,
            beam_width=6,
            max_iterations=10,
            random_state=0,
            low_cardinality_threshold=10,
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        ops = [atom.op for rule in ruleset.rules for atom in rule.atoms]
        # Must contain at least one equality or "in" split
        assert any(op in ("==", "in") for op in ops), (
            f"Expected equality splits for low-cardinality features, got ops: {ops}"
        )

    def test_nonzero_atoms_on_low_cardinality(self):
        """Pittsburgh must not return 0 atoms on integer-encoded data."""
        X, y = self._monk_like_data()
        clf = PittsburghRuleSetClassifier(
            max_rules=6,
            candidate_pool_size=32,
            beam_width=6,
            max_iterations=10,
            random_state=0,
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        n_atoms = sum(len(r.atoms) for r in ruleset.rules)
        assert n_atoms > 0, "Expected at least 1 atom on low-cardinality data"

    def test_threshold_disabled_does_not_generate_equality(self):
        """With low_cardinality_threshold=0, no extra equality splits."""
        X, y = self._monk_like_data()
        clf = PittsburghRuleSetClassifier(
            max_rules=4,
            candidate_pool_size=16,
            beam_width=4,
            max_iterations=6,
            random_state=0,
            low_cardinality_threshold=0,
            enable_categorical_rules=False,
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        ops = [atom.op for rule in ruleset.rules for atom in rule.atoms]
        # Only numeric ops should appear (no == or in)
        assert all(op in ("<=", ">", "between") for op in ops), (
            f"Expected only numeric ops when low-cardinality disabled, got: {ops}"
        )

    def test_multiclass_low_cardinality(self):
        """Low-cardinality detection works with multiclass data."""
        rng = np.random.default_rng(123)
        X = np.column_stack([
            rng.integers(0, 4, size=300),
            rng.integers(0, 3, size=300),
            rng.integers(0, 5, size=300),
        ])
        y = X[:, 0] % 3  # 3 classes
        clf = PittsburghRuleSetClassifier(
            max_rules=6,
            candidate_pool_size=24,
            beam_width=5,
            max_iterations=8,
            random_state=0,
            low_cardinality_threshold=10,
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        n_atoms = sum(len(r.atoms) for r in ruleset.rules)
        assert n_atoms > 0, "Expected non-zero atoms on multiclass low-cardinality data"

    def test_high_cardinality_skips_equality(self):
        """Features above the threshold get only numeric splits."""
        rng = np.random.default_rng(42)
        # Feature with 50 unique values → above default threshold of 10
        X = rng.uniform(0, 10, size=(200, 3))
        y = (X[:, 0] > 5).astype(int)
        clf = PittsburghRuleSetClassifier(
            max_rules=4,
            candidate_pool_size=16,
            beam_width=4,
            max_iterations=6,
            random_state=0,
            low_cardinality_threshold=10,
            enable_categorical_rules=False,
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        ops = [atom.op for rule in ruleset.rules for atom in rule.atoms]
        assert all(op in ("<=", ">", "between") for op in ops), (
            f"High-cardinality numeric features should not get == splits, got: {ops}"
        )


