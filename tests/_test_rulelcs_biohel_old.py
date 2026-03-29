"""Tests for BioHEL-inspired enhancements in RulePLCSClassifier.

Covers:
- Sequential Covering (IRL)
- Token Competition
- Post-hoc Rule Compaction
- Windowing (stochastic evaluation)
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification

from scoredrulesets import RulePLCSClassifier


# ── helpers ──────────────────────────────────────────────────────────
def _iris():
    return load_iris(return_X_y=True)


def _synthetic_binary(n=200, seed=42):
    return make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=seed,
    )


# ── Sequential Covering (IRL) ───────────────────────────────────────

class TestSequentialCovering:

    def test_basic_fit_predict(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            sequential_covering=True,
            max_rules=4,
            random_state=0,
        )
        clf.fit(X, y)

        pred = clf.predict(X[:10])
        proba = clf.predict_proba(X[:10])
        assert pred.shape == (10,)
        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_metadata_records_sequential_covering(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            sequential_covering=True,
            max_rules=3,
            random_state=0,
        )
        clf.fit(X, y)
        meta = clf.to_ruleset().metadata
        assert meta["sequential_covering"] is True
        assert meta["source"] == "ruleplcs"

    def test_sequential_produces_rules(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            sequential_covering=True,
            max_rules=5,
            random_state=0,
        )
        clf.fit(X, y)
        rs = clf.to_ruleset()
        non_default = [r for r in rs.rules if r.atoms]
        assert len(non_default) >= 1

    def test_sequential_binary(self):
        X, y = _synthetic_binary()
        clf = RulePLCSClassifier(
            sequential_covering=True,
            max_rules=3,
            min_samples_leaf=3,
            random_state=7,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (200,)


# ── Token Competition ────────────────────────────────────────────────

class TestTokenCompetition:

    def test_fit_with_token_competition(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            token_competition_weight=0.15,
            max_rules=4,
            beam_width=5,
            max_iterations=8,
            random_state=0,
        )
        clf.fit(X, y)
        pred = clf.predict(X[:5])
        assert pred.shape == (5,)

    def test_metadata_records_token_competition_weight(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            token_competition_weight=0.2,
            max_rules=3,
            random_state=0,
        )
        clf.fit(X, y)
        meta = clf.to_ruleset().metadata
        assert meta["token_competition_weight"] == pytest.approx(0.2)

    def test_zero_weight_matches_baseline(self):
        """With weight=0 the token competition term is inactive."""
        X, y = _iris()
        base = RulePLCSClassifier(
            max_rules=3, beam_width=4, max_iterations=6, random_state=0,
        )
        base.fit(X, y)

        tc = RulePLCSClassifier(
            token_competition_weight=0.0,
            max_rules=3, beam_width=4, max_iterations=6, random_state=0,
        )
        tc.fit(X, y)

        # Same rules selected (deterministic, same seed, weight=0)
        base_ids = sorted(r.rule_id for r in base.to_ruleset().rules)
        tc_ids = sorted(r.rule_id for r in tc.to_ruleset().rules)
        assert base_ids == tc_ids


# ── Post-hoc Rule Compaction ─────────────────────────────────────────

class TestRuleCompaction:

    def test_compaction_reduces_or_keeps_rules(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            enable_compaction=True,
            max_rules=6,
            candidate_pool_size=24,
            beam_width=6,
            max_iterations=10,
            random_state=0,
        )
        clf.fit(X, y)
        rs = clf.to_ruleset()
        non_default = [r for r in rs.rules if r.atoms]
        # Compaction should keep at least 1 non-default rule on Iris
        assert len(non_default) >= 1
        # And the total should be <= max_rules
        assert len(non_default) <= 6

    def test_compaction_metadata(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            enable_compaction=True,
            max_rules=4,
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.to_ruleset().metadata["compaction_enabled"] is True

    def test_compaction_preserves_accuracy(self):
        """Compaction should not significantly degrade accuracy."""
        X, y = _iris()
        no_compact = RulePLCSClassifier(
            enable_compaction=False,
            max_rules=5, random_state=0,
        )
        no_compact.fit(X, y)
        acc_before = (no_compact.predict(X) == y).mean()

        compact = RulePLCSClassifier(
            enable_compaction=True,
            max_rules=5, random_state=0,
        )
        compact.fit(X, y)
        acc_after = (compact.predict(X) == y).mean()

        # Compaction should not drop accuracy by more than 5pp
        assert acc_after >= acc_before - 0.05


# ── Windowing ────────────────────────────────────────────────────────

class TestWindowing:

    def test_windowing_runs_without_error(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            window_fraction=0.5,
            max_rules=4,
            beam_width=5,
            max_iterations=8,
            random_state=0,
        )
        clf.fit(X, y)
        pred = clf.predict(X[:5])
        assert pred.shape == (5,)

    def test_windowing_metadata(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            window_fraction=0.3,
            max_rules=3,
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.to_ruleset().metadata["window_fraction"] == pytest.approx(0.3)

    def test_full_window_is_default(self):
        X, y = _iris()
        clf = RulePLCSClassifier(max_rules=3, random_state=0)
        clf.fit(X, y)
        assert clf.to_ruleset().metadata["window_fraction"] == pytest.approx(1.0)


# ── Combined features ───────────────────────────────────────────────

class TestCombinedFeatures:

    def test_all_biohel_features_together(self):
        X, y = _iris()
        clf = RulePLCSClassifier(
            sequential_covering=True,
            token_competition_weight=0.1,
            enable_compaction=True,
            window_fraction=0.6,
            max_rules=4,
            random_state=0,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        rs = clf.to_ruleset()
        assert pred.shape == (150,)
        assert rs.metadata["sequential_covering"] is True
        assert rs.metadata["compaction_enabled"] is True

    def test_sequential_with_compaction_binary(self):
        X, y = _synthetic_binary(n=300, seed=123)
        clf = RulePLCSClassifier(
            sequential_covering=True,
            enable_compaction=True,
            max_rules=5,
            min_samples_leaf=3,
            random_state=42,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        acc = (pred == y).mean()
        # Should achieve reasonable accuracy on this easy synthetic dataset
        assert acc > 0.6

