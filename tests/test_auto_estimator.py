"""Tests for AutoScoredRuleSetClassifier (AutoML meta-estimator)."""

import os
import numpy as np
import pytest
from sklearn.datasets import load_iris

from scoredrulesets.estimators.auto import AutoScoredRuleSetClassifier


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y


class TestAutoEstimator:

    def test_basic_fit_predict(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "hs"],
            cv=3,
            scoring="f1_weighted",
            random_state=0,
        )
        clf.fit(X, y)

        assert clf.best_backend_ in ("cart", "hs")
        assert isinstance(clf.cv_results_, dict)
        assert len(clf.cv_results_) == 2
        assert clf.best_score_ > 0

        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(y))

    def test_to_ruleset(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        ruleset = clf.to_ruleset()
        assert ruleset is not None
        assert len(ruleset.rules) > 0

    def test_predict_proba(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "hs"],
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_with_preprocessing(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            cv=2,
            preprocessing={"feature_selection": "kbest", "k": 2},
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.best_backend_ == "cart"
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_per_backend_params(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "ruleplcs"],
            backend_params={
                "cart": {"max_depth": 2},
                "ruleplcs": {"max_rules": 3},
            },
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.best_backend_ in ("cart", "ruleplcs")

    def test_failing_backend_skipped(self, iris_data):
        """A backend that fails during CV should be skipped with a warning."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "nonexistent_backend"],
            cv=2,
            random_state=0,
        )
        with pytest.warns(UserWarning, match="failed during CV"):
            clf.fit(X, y)

        assert clf.best_backend_ == "cart"
        assert clf.cv_results_["nonexistent_backend"] == float("-inf")

    def test_default_backends(self, iris_data):
        """Default candidate_backends should work when no backends specified."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(cv=2, random_state=0)
        clf.fit(X, y)
        assert clf.best_backend_ in ("greedy_pareto", "cart", "hs", "ruleplcs", "rulenln")

    def test_rulenln_sweep_populates_archive(self, iris_data):
        """The neural backend should contribute top-k sweep variants to the archive."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["rulenln"],
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        spectrum = clf.get_pareto_spectrum()
        sweep_backends = [s["backend"] for s in spectrum if s["backend"].startswith("rulenln(")]
        # At least one non-dominated k-variant should be admitted alongside
        # the unmasked rulenln candidate.
        assert len(sweep_backends) >= 1
        for s in spectrum:
            if s["backend"].startswith("rulenln(k="):
                k = int(s["backend"].split("k=")[1].rstrip(")"))
                assert s["complexity"] <= k * 20 + 2  # k per rule (+ default)

    def test_rulenln_explicit_k_disables_sweep(self, iris_data):
        """If max_atoms_per_rule is pinned, no additional sweep variants appear."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["rulenln"],
            backend_params={"rulenln": {"max_atoms_per_rule": 3, "n_rules": 6, "epochs": 80}},
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)
        spectrum = clf.get_pareto_spectrum()
        assert all(s["backend"] == "rulenln" for s in spectrum)

    def test_pareto_fusion_and_spectrum(self, iris_data):
        """Test Master Pareto Archive spectrum, active model switching, and index card export."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["greedy_pareto", "cart"],
            preference="balanced",
            enable_pareto_fusion=True,
            cv=2,
            random_state=0,
        )
        clf.fit(X, y)

        spectrum = clf.get_pareto_spectrum()
        assert isinstance(spectrum, list)
        assert len(spectrum) >= 1

        # Check structure of spectrum entries
        first = spectrum[0]
        assert "complexity" in first
        assert "score" in first
        assert "backend" in first
        assert "ruleset" in first

        # Test index card export
        card = clf.export_index_card()
        assert isinstance(card, str)
        assert len(card) > 0

        # Test active model switching
        orig_ruleset = clf.to_ruleset()
        target_complexity = spectrum[-1]["complexity"]
        clf.set_active_model(target_complexity)
        assert clf.to_ruleset() is not None

        # Verify predictions work
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_preferences(self, iris_data):
        """Test different user intent profiles ('compact', 'balanced', 'accuracy')."""
        X, y = iris_data
        for pref in ("compact", "balanced", "accuracy"):
            clf = AutoScoredRuleSetClassifier(
                candidate_backends=["greedy_pareto", "cart"],
                preference=pref,
                enable_pareto_fusion=True,
                cv=2,
                random_state=0,
            )
            clf.fit(X, y)
            preds = clf.predict(X)
            assert len(preds) == len(y)

    def test_auto_regressor(self):
        """Test AutoScoredRuleSetRegressor on continuous target."""
        from sklearn.datasets import make_regression
        from scoredrulesets.estimators.auto import AutoScoredRuleSetRegressor

        X, y = make_regression(n_samples=60, n_features=4, noise=0.1, random_state=42)
        reg = AutoScoredRuleSetRegressor(
            candidate_backends=["greedy_cascaded", "cart"],
            cv=2,
            random_state=0,
        )
        reg.fit(X, y)

        assert reg.best_backend_ in ("greedy_cascaded", "cart")
        assert reg.ruleset_ is not None
        preds = reg.predict(X)
        assert len(preds) == len(y)

        spectrum = reg.get_pareto_spectrum()
        assert isinstance(spectrum, list)
        assert len(spectrum) >= 1

    def test_plot_pareto_front(self, tmp_path, iris_data):
        """Test Pareto front plotting and saving."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["greedy_pareto", "cart"],
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)
        out_file = str(tmp_path / "pareto.png")
        fig, ax = clf.plot_pareto_front(output_path=out_file)
        assert fig is not None
        assert os.path.exists(out_file)

    def test_probing_strategy(self, iris_data):
        """Test multi-fidelity probing with subsampling."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["greedy_pareto", "cart"],
            probing_strategy="subsample",
            probing_threshold_samples=50,
            probing_subsample=0.5,
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.best_backend_ in ("greedy_pareto", "cart")
        assert len(clf.predict(X)) == len(y)

    def test_auto_feature_selection_high_dimension(self):
        """Automated feature selection triggers on high-D datasets (D >= 50)."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=80,
            n_features=60,
            n_informative=6,
            random_state=42,
        )
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            feature_selection="auto",
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)

        # Feature selection should have automatically been activated
        assert clf.selected_feature_indices_ is not None
        assert len(clf.selected_feature_indices_) < 60
        assert clf.effective_preprocessing_.get("feature_selection") == "kbest"

        preds = clf.predict(X)
        assert len(preds) == len(y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_auto_feature_selection_small_dimension(self, iris_data):
        """Automated feature selection does not trigger on small-D datasets (D < 50)."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            feature_selection="auto",
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)

        assert clf.selected_feature_indices_ is None
        assert "feature_selection" not in clf.effective_preprocessing_
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_explicit_feature_selection_kbest_f(self):
        """User can specify explicit kbest_f feature selection and max_features."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=60,
            n_features=40,
            n_informative=5,
            random_state=42,
        )
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            feature_selection="kbest_f",
            max_features=6,
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)

        assert clf.selected_feature_indices_ is not None
        assert len(clf.selected_feature_indices_) == 6
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_disabled_feature_selection_high_dimension(self):
        """User can disable feature selection explicitly on high-D datasets."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=60,
            n_features=60,
            n_informative=5,
            random_state=42,
        )
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"],
            feature_selection="none",
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)

        assert clf.selected_feature_indices_ is None
        assert "feature_selection" not in clf.effective_preprocessing_
        preds = clf.predict(X)
        assert len(preds) == len(y)


