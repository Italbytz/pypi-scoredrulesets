import numpy as np
import pytest
import runpy
from pathlib import Path

from sklearn.datasets import load_iris, load_breast_cancer

from scoredrulesets import RuleLCSClassifier, ScoredRuleSetClassifier
from scoredrulesets.benchmarking.estimators import default_estimator_specs


def test_rulelcs_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = RuleLCSClassifier(
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
    assert any(rule.rule_id == "rulelcs_default_prior" for rule in ruleset.rules)
    assert any(
        rule.rule_id and rule.rule_id.startswith(("rulelcs_rule_", "rulelcs_ovr_", "rulelcs_conj_"))
        for rule in ruleset.rules
    )
    assert ruleset.metadata["source"] == "rulelcs"
    assert ruleset.metadata["selected_rule_count"] <= 4


def test_rulelcs_estimator_handles_categorical_rules():
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

    clf = RuleLCSClassifier(
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



def test_sklearn_wrapper_rulelcs_backend():
    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(
        backend="rulelcs",
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
    assert clf.to_ruleset().metadata["source"] == "rulelcs"



def test_benchmarking_estimator_specs_include_rulelcs():
    specs = default_estimator_specs()
    assert "wrapper_rulelcs" in specs
    assert "wrapper_rulelcs_strong" in specs


def test_rulelcs_benchmark_profiles_use_expected_backend_and_budget():
    specs = default_estimator_specs()

    base = specs["wrapper_rulelcs"].factory()
    strong = specs["wrapper_rulelcs_strong"].factory()

    assert base.backend == "rulelcs"
    assert base.backend_params["candidate_pool_size"] >= 48
    assert base.backend_params["max_rules"] >= 10

    assert strong.backend == "rulelcs"
    assert strong.backend_params["beam_width"] >= 16
    assert strong.backend_params["candidate_pool_size"] >= 64
    assert strong.backend_params["sequential_covering"] is True



def test_rulelcs_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "estimators"
        / "example_rulelcs_backend.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["profile"] == "default"
    assert result["rulelcs"]["metadata"]["source"] == "rulelcs"
    assert result["rulelcs"]["n_rules"] > 0
    assert len(result["comparison"]) == 1
    assert {row["name"] for row in result["comparison"]} == {"cart"}


def test_rulelcs_example_profile_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "estimators"
        / "example_rulelcs_backend.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0, profile="strong")

    assert result["profile"] == "strong"
    meta = result["rulelcs"]["metadata"]
    assert meta["source"] == "rulelcs"
    assert meta["beam_width"] >= 6


def test_rulelcs_wrapper_example_run_demo_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "estimators"
        / "example_rulelcs_wrapper.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0)

    assert result["dataset"] == "sklearn_iris"
    assert result["profile"] == "default"
    assert result["wrapper_rulelcs"]["metadata"]["source"] == "rulelcs"
    assert result["wrapper_rulelcs"]["n_rules"] > 0
    assert len(result["comparison"]) == 2
    assert {row["name"] for row in result["comparison"]} == {"wrapper_cart_d2", "wrapper_cart_d4"}


def test_rulelcs_wrapper_example_profile_smoke():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "estimators"
        / "example_rulelcs_wrapper.py"
    )
    module_globals = runpy.run_path(str(example_path))
    result = module_globals["run_demo"](random_state=0, profile="strong")

    assert result["profile"] == "strong"
    meta = result["wrapper_rulelcs"]["metadata"]
    assert meta["source"] == "rulelcs"
    assert meta["beam_width"] >= 6


# ---------------------------------------------------------------------------
# Multi-class (OvR) Tests
# ---------------------------------------------------------------------------


class TestRuleLCSOvRMulticlass:
    """Tests for the One-vs-Rest multiclass strategy."""

    def test_ovr_multiclass_iris(self):
        """OvR on Iris (3 classes): correct shapes, proba sums to 1, metadata."""
        X, y = load_iris(return_X_y=True)
        clf = RuleLCSClassifier(
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
        assert ruleset.metadata["source"] == "rulelcs"

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
        clf = RuleLCSClassifier(
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
        clf = RuleLCSClassifier(
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
        clf = RuleLCSClassifier(
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
            backend="rulelcs",
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
        clf = RuleLCSClassifier(
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
        clf = RuleLCSClassifier(
            max_rules=3,
            random_state=0,
            multiclass_strategy="foo",
        )
        with pytest.raises(ValueError, match="multiclass_strategy"):
            clf.fit(X, y)

    def test_ovr_score_vector_length(self):
        """Each OvR rule should have a score vector of length n_classes."""
        X, y = load_iris(return_X_y=True)
        clf = RuleLCSClassifier(
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


class TestRuleLCSLowCardinality:
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
        clf = RuleLCSClassifier(
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
        """RuleLCS must not return 0 atoms on integer-encoded data."""
        X, y = self._monk_like_data()
        clf = RuleLCSClassifier(
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
        clf = RuleLCSClassifier(
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
        clf = RuleLCSClassifier(
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
        clf = RuleLCSClassifier(
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


class TestRuleLCSSequentialCoveringStability:
    """Sequential covering must not degenerate on imbalanced or high-dim data.

    Regression tests for the bug where _default_scores_ was overwritten with
    the residual distribution after sequential covering.  This caused the model
    to predict only the residual-majority class on every sample when the
    residual became heavily skewed.
    """

    def test_sequential_covering_no_default_score_degeneration_imbalanced(self):
        """strong profile (sequential_covering=True) must not degenerate on
        imbalanced binary data (10 % minority).  macro-F1 should stay > 0.5."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        from scoredrulesets.benchmarking.datasets import generate_imbalanced_dataset

        X, y = generate_imbalanced_dataset(
            n_samples=600, n_features=10, n_informative=4,
            imbalance_ratio=0.1, random_state=0,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y,
        )
        clf = RuleLCSClassifier(
            max_rules=8, min_samples_leaf=3, candidate_pool_size=32, beam_width=8,
            max_iterations=16, validation_fraction=0.25,
            sequential_covering=True, enable_compaction=True, random_state=0,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

        # default_scores must still reflect the full training prior, not the residual
        default_rule = next(r for r in clf.to_ruleset().rules if not r.atoms)
        majority_score = max(default_rule.scores)
        assert majority_score < 0.95, (
            f"default_scores appear to reflect the skewed residual, not the full prior: "
            f"{default_rule.scores}"
        )
        assert f1 > 0.50, f"rulelcs_strong degenerated on imbalanced data: macro-F1={f1:.3f}"

    def test_sequential_covering_breast_cancer_no_collapse(self):
        """Regression: rulelcs_strong must not predict a single class on breast cancer."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score

        X, y = load_breast_cancer(return_X_y=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=1000, stratify=y,
        )
        clf = RuleLCSClassifier(
            max_rules=15, min_samples_leaf=3, candidate_pool_size=64, beam_width=16,
            max_iterations=32, validation_fraction=0.25, complexity_penalty=0.001,
            sequential_covering=True, enable_compaction=True, random_state=1000,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        # Must predict both classes (no single-class collapse)
        predicted_classes = set(y_pred.tolist())
        assert len(predicted_classes) >= 2, (
            f"rulelcs_strong collapsed to single class {predicted_classes} on breast_cancer"
        )
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        assert f1 > 0.70, f"rulelcs_strong degenerated on breast_cancer: macro-F1={f1:.3f}"

