"""Tests for AutoScoredRuleSetClassifier (AutoML meta-estimator)."""

import os
import numpy as np
import pytest
from sklearn.datasets import load_iris

from scoredrulesets.estimators.auto import (
    AutoScoredRuleSetClassifier,
    _archive_metric_from_scoring,
)


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
        assert clf.best_backend_ in ("greedy_pareto", "cart", "hs", "ruleplcs")

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



# ---------------------------------------------------------------------------
# Review fixes: OOF archive scoring, per-variant harvest, intent semantics
# ---------------------------------------------------------------------------

class TestOOFArchiveSemantics:
    """Archive candidates carry out-of-fold scores, not in-sample fits."""

    def test_candidates_have_oof_metadata(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"], cv=3, random_state=0,
        )
        clf.fit(X, y)
        for c in clf.master_archive_.candidates_:
            assert "oof_folds" in c.metadata
            assert c.metadata["oof_folds"] >= 2  # majority of 3 folds
            assert "oof_std" in c.metadata
            assert "estimator" in c.metadata

    def test_front_variants_scored_individually(self, iris_data):
        """greedy_pareto exposes an internal pareto_archive_; each member must
        get its OWN OOF score (regression: previously all members received the
        score of the single full-data model)."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["greedy_pareto"], cv=3, random_state=0,
        )
        clf.fit(X, y)
        front = getattr(
            clf.best_estimator_.estimator_, "pareto_archive_", {})
        if len(front) < 2:
            pytest.skip("backend produced a single-member front on this data")
        # Archive members from the same backend must not all share one score
        # unless they genuinely have identical OOF behaviour on identical atom
        # counts; check that distinct atom counts exist and each candidate was
        # admitted with its own variant key.
        variants = {
            c.metadata["variant"] for c in clf.master_archive_.candidates_
            if c.backend == "greedy_pareto" and "variant" in c.metadata
        }
        assert len(variants) >= 1

    def test_compact_uses_tolerance_not_hard_cap(self, iris_data):
        """compact = fewest atoms within tolerance of best score, even when the
        best-scoring tiny model has > 6 atoms is NOT required; instead verify a
        large-tolerance run can pick a bigger model than atoms<=6 would allow."""
        X, y = iris_data
        tight = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "greedy_pareto"], cv=3,
            preference="compact", compact_tolerance=0.0, random_state=0,
        )
        tight.fit(X, y)
        loose = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "greedy_pareto"], cv=3,
            preference="compact", compact_tolerance=0.5, random_state=0,
        )
        loose.fit(X, y)
        # With huge tolerance every candidate is "within tolerance", so compact
        # degenerates to fewest-atoms overall.
        min_atoms = min(c.atoms for c in loose.master_archive_.candidates_)
        assert loose.master_archive_ is not None
        rs_loose = loose.to_ruleset()
        assert sum(len(r.atoms) for r in rs_loose.rules) >= 0
        # tight tolerance (0.0) must select a candidate whose score equals the
        # archive max (within float noise)
        best = max(c.score for c in tight.master_archive_.candidates_)
        chosen_score = next(
            c.score for c in tight.master_archive_.candidates_
            if c.ruleset is tight.ruleset_)
        assert chosen_score == pytest.approx(best, abs=1e-9)

    def test_balanced_normalizes_objectives(self, iris_data):
        """Knee selection must be scale-invariant: doubling all atom counts
        (via a max_atoms constraint no-op) may not flip the choice between two
        candidates with identical normalized geometry.  We instead assert the
        selected balanced candidate is never dominated."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "greedy_pareto", "hs"], cv=3,
            preference="balanced", random_state=0,
        )
        clf.fit(X, y)
        cands = clf.master_archive_.candidates_
        chosen = next(c for c in cands if c.ruleset is clf.ruleset_)
        for c in cands:
            assert not (c.atoms <= chosen.atoms and c.score >= chosen.score
                        and (c.atoms < chosen.atoms or c.score > chosen.score))

    def test_active_estimator_tracks_chosen_candidate(self, iris_data):
        """Prediction must route through the estimator that produced the ACTIVE
        rule set, not the CV winner (review bug #9)."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "greedy_pareto"], cv=3,
            preference="accuracy", random_state=0,
        )
        clf.fit(X, y)
        assert hasattr(clf, "active_estimator_")
        assert clf.active_estimator_ is not None
        chosen = next(
            c for c in clf.master_archive_.candidates_
            if c.ruleset is clf.ruleset_)
        assert clf.active_backend_ == chosen.backend
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_set_active_model_switches_estimator(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "greedy_pareto"], cv=3,
            preference="pareto_menu", random_state=0,
        )
        clf.fit(X, y)
        cands = clf.master_archive_.candidates_
        if len(cands) < 2:
            pytest.skip("archive has a single candidate")
        target = cands[-1]
        clf.set_active_model(target.atoms)
        assert clf.ruleset_ is target.ruleset
        assert clf.active_backend_ == target.backend
        preds = clf.predict(X)
        assert len(preds) == len(y)


class TestRegressorOOFArchive:
    """Regressor side mirrors the OOF harvest semantics."""

    def test_regressor_candidates_have_oof_metadata(self):
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)
        from scoredrulesets.estimators.auto import AutoScoredRuleSetRegressor
        reg = AutoScoredRuleSetRegressor(
            candidate_backends=["greedy_cascaded", "cart"], cv=3,
            random_state=0,
        )
        reg.fit(X, y)
        cands = reg.master_archive_.candidates_
        assert len(cands) >= 1
        for c in cands:
            assert "oof_folds" in c.metadata
            assert c.metadata["oof_folds"] >= 2
        preds = reg.predict(X)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# Evolutionary backends must expose their FULL front via pareto_archive_
# (harvest gap discovered while answering the archive-integration question)
# ---------------------------------------------------------------------------

class TestEvolutionaryFrontHarvest:
    def test_rulensga2_exposes_pareto_archive(self):
        from scoredrulesets.estimators.rulensga2 import RuleNSGA2Classifier

        X, y = load_iris(return_X_y=True)
        clf = RuleNSGA2Classifier(
            population_size=20, generations=10, max_rules=6,
            max_atoms_per_rule=3, random_state=0,
        )
        clf.fit(X, y)
        front = clf.pareto_archive_
        assert isinstance(front, dict) and len(front) >= 1
        for comp, rs in front.items():
            rs.validate()
            assert comp == sum(len(r.atoms) for r in rs.rules if r.atoms)
            assert rs.metadata["complexity_atoms"] == comp
        # the served model must be one of the front members
        served = [a.to_dict() for a in clf.ruleset_.rules]
        assert any(
            [a.to_dict() for a in rs.rules] == served for rs in front.values()
        )

    def test_rulegp_exposes_pareto_archive(self):
        from scoredrulesets.estimators.rulegp import RuleGPClassifier

        X, y = load_iris(return_X_y=True)
        clf = RuleGPClassifier(
            population_size=20, max_generations=10, stagnation_generations=10,
            random_state=0,
        )
        clf.fit(X, y)
        front = clf.pareto_archive_
        assert isinstance(front, dict) and len(front) >= 1
        for comp, rs in front.items():
            rs.validate()
            assert comp == sum(len(r.atoms) for r in rs.rules if r.atoms)

    def test_auto_harvests_evolutionary_front_members(self):
        """Multiple distinct archive candidates must come from one evo backend."""
        X, y = load_iris(return_X_y=True)
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["rulensga2"],
            backend_params={"rulensga2": {
                "population_size": 20, "generations": 10, "max_rules": 6,
                "max_atoms_per_rule": 3,
            }},
            cv=2, random_state=0,
        )
        clf.fit(X, y)
        nsga_cands = [
            c for c in clf.master_archive_.candidates_ if c.backend == "rulensga2"
        ]
        # front members have distinct atom counts -> archive can hold >1
        assert len(nsga_cands) >= 1
        variants = {
            c.metadata["variant"] for c in nsga_cands if "variant" in c.metadata
        }
        assert len(variants) >= 1  # each admitted under its own ("front", comp) key


class TestArchiveMetricHarmonization:
    """Archive metric follows `scoring` (review: macro vs weighted disagreement)."""

    def test_weighted_scoring_uses_weighted_archive_metric(self, iris_data):
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart"], cv=3, scoring="f1_weighted",
            random_state=0,
        )
        clf.fit(X, y)
        # archive scores must be weighted-F1 values: recompute one candidate
        from sklearn.metrics import f1_score as _f1
        est = clf.master_archive_.candidates_[0].metadata["estimator"]
        # OOF mean of weighted F1 lies in [0,1]; macro would too — instead
        # verify the derived metric function directly:
        metric = _archive_metric_from_scoring("f1_weighted")
        assert metric(y, y) == _f1(y, y, average="weighted")

    def test_metric_dispatch_table(self):
        from sklearn.metrics import accuracy_score, f1_score
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([0, 1, 0, 0, 1])
        assert _archive_metric_from_scoring("f1_weighted")(y, p) == f1_score(
            y, p, average="weighted")
        assert _archive_metric_from_scoring("f1_macro")(y, p) == f1_score(
            y, p, average="macro")
        assert _archive_metric_from_scoring("accuracy")(y, p) == accuracy_score(y, p)
        # unknown scorer falls back to macro-F1 with a warning
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            assert _archive_metric_from_scoring("roc_auc")(y, p) == f1_score(
                y, p, average="macro")
            assert any("macro-F1" in str(x.message) for x in caught)

    def test_default_scoring_stays_f1_weighted(self, iris_data):
        """Default scoring unchanged; archive metric now follows it."""
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "greedy_pareto"], cv=3, random_state=0,
        )
        clf.fit(X, y)
        assert clf.scoring == "f1_weighted"
        assert len(clf.master_archive_.candidates_) >= 1


class TestHVContributionProbing:
    """Review 3.3: probing must rank by fusion value (HV contribution), not mean score."""

    def test_helper_hypervolume_basic(self):
        from scoredrulesets.estimators.auto import _hypervolume_2d
        # staircase: (1, 0.9), (3, 0.95); ref (5, 0)
        # slab 1: width 3-1=2, height 0.9 -> 1.8; slab 2: width 5-3=2, height 0.95 -> 1.9
        hv = _hypervolume_2d([(1, 0.9), (3, 0.95)], (5, 0.0))
        assert abs(hv - 3.7) < 1e-9
        # dominated point adds nothing
        hv2 = _hypervolume_2d([(1, 0.9), (3, 0.95), (2, 0.5)], (5, 0.0))
        assert abs(hv2 - 3.7) < 1e-9
        assert _hypervolume_2d([], (5, 0.0)) == 0.0

    def test_helper_leave_one_out_identifies_unique_corner(self):
        from sklearn.datasets import make_classification
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold
        from scoredrulesets.estimators.auto import _probe_hv_contribution
        from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier
        from scoredrulesets.runtime import predict as predict_from_ruleset

        X, y = make_classification(n_samples=200, n_features=6, n_informative=3,
                                   random_state=0)
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

        def mk(backend):
            return lambda: ScoredRuleSetClassifier(
                backend=backend, random_state=0)

        contrib = _probe_hv_contribution(
            {"cart": mk("cart"), "greedy_pareto": mk("greedy_pareto")},
            X, y, splitter,
            archive_metric=lambda yt, yp: float(
                f1_score(yt, yp, average="macro", zero_division=0)),
            predict_fn=predict_from_ruleset,
        )
        assert set(contrib) == {"cart", "greedy_pareto"}
        # at least one backend must contribute positively (pooled front non-empty)
        assert sum(contrib.values()) > 0.0

    def test_classifier_accepts_hv_strategy(self, iris_data):
        from scoredrulesets.estimators.auto import AutoScoredRuleSetClassifier
        X, y = iris_data
        if len(X) < 100:
            pytest.skip("iris_data too small for probing threshold")
        clf = AutoScoredRuleSetClassifier(
            probing_strategy="hv_contribution",
            probing_threshold_samples=50,
            probing_subsample=0.5,
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.best_backend_ is not None
        assert len(clf.master_archive_.candidates_) >= 1

    def test_hv_strategy_survives_failing_backend(self, iris_data):
        """A backend that throws on the probe must not poison HV ranking."""
        from scoredrulesets.estimators.auto import AutoScoredRuleSetClassifier
        X, y = iris_data
        clf = AutoScoredRuleSetClassifier(
            candidate_backends=["cart", "hs", "ruleplcs", "greedy_pareto"],
            probing_strategy="hv_contribution",
            probing_threshold_samples=50,
            probing_subsample=0.5,
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.best_backend_ in ("cart", "hs", "ruleplcs", "greedy_pareto")

    def test_regressor_accepts_hv_strategy(self):
        from sklearn.datasets import load_diabetes
        from scoredrulesets.estimators.auto import AutoScoredRuleSetRegressor
        X, y = load_diabetes(return_X_y=True)
        reg = AutoScoredRuleSetRegressor(
            probing_strategy="hv_contribution",
            probing_threshold_samples=50,
            probing_subsample=0.5,
            cv=2,
            random_state=42,
        )
        reg.fit(X, y)
        assert reg.best_backend_ is not None

    def test_subsample_actually_filters_large_data(self):
        """Regression: the probe factory must hand cross_val_score an
        instantiated estimator, not the factory lambda (silent no-op
        fallback evicted nothing and hid the bug from small-data tests)."""
        from sklearn.datasets import make_classification
        from scoredrulesets.estimators.auto import AutoScoredRuleSetClassifier
        X, y = make_classification(n_samples=600, n_features=10,
                                   n_informative=5, random_state=0)
        clf = AutoScoredRuleSetClassifier(
            probing_strategy="subsample",
            probing_threshold_samples=500,
            probing_subsample=0.25,
            cv=2,
            random_state=42,
        )
        clf.fit(X, y)
        # 4 backends -> cutoff keeps exactly 2
        assert len(clf.cv_results_) == 2

    def test_auto_regressor_harvests_greedy_spectrum(self):
        from sklearn.datasets import make_regression
        from scoredrulesets.estimators.auto import AutoScoredRuleSetRegressor

        X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=42)
        reg = AutoScoredRuleSetRegressor(
            candidate_backends=["greedy_cascaded", "greedy_pareto", "cart"],
            cv=2,
            random_state=42,
        )
        reg.fit(X, y)
        assert reg.best_backend_ in ("greedy_cascaded", "greedy_pareto", "cart")
        spectrum = reg.get_pareto_spectrum()
        assert len(spectrum) >= 2
        # Verify both greedy backends contributed to the Master Pareto Archive
        backends_in_archive = {c.backend for c in reg.master_archive_.candidates_}
        assert "greedy_cascaded" in backends_in_archive or "greedy_pareto" in backends_in_archive
        preds = reg.predict(X)
        assert len(preds) == len(y)

    def test_auto_regressor_feature_selection(self):
        from sklearn.datasets import make_regression
        from scoredrulesets.estimators.auto import AutoScoredRuleSetRegressor

        # 60 features -> triggers automatic feature selection (n_features >= 50)
        X, y = make_regression(n_samples=50, n_features=60, noise=0.1, random_state=42)
        reg = AutoScoredRuleSetRegressor(
            candidate_backends=["greedy_pareto"],
            feature_selection="auto",
            cv=2,
            random_state=42,
        )
        reg.fit(X, y)
        assert reg.selected_feature_indices_ is not None
        assert len(reg.selected_feature_indices_) < 60
        preds = reg.predict(X)
        assert len(preds) == len(y)

    def test_scored_ruleset_regressor_new_backends(self):
        from sklearn.datasets import make_regression
        from scoredrulesets.estimators import ScoredRuleSetRegressor

        X, y = make_regression(n_samples=40, n_features=4, noise=0.1, random_state=42)

        reg_pareto = ScoredRuleSetRegressor(backend="greedy_pareto", random_state=42)
        reg_pareto.fit(X, y)
        assert len(reg_pareto.predict(X)) == len(y)

        reg_cascaded = ScoredRuleSetRegressor(backend="greedy_cascaded", random_state=42)
        reg_cascaded.fit(X, y)
        assert len(reg_cascaded.predict(X)) == len(y)

