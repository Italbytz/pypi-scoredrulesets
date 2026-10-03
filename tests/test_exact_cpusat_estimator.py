"""Tests for the exact CP-SAT certified-compact backend."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import scoredrulesets.estimators._time_budget as time_budget_module
from scoredrulesets.estimators.exact_cpusat import (
    ExactCPSATClassifier,
    enumerate_candidates,
)
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier
from scoredrulesets.schema import ScoredRuleSet

import importlib.util

HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
needs_ortools = pytest.mark.skipif(
    not HAS_ORTOOLS, reason="exact backend requires 'scoredrulesets[exact]'"
)


class TestPython314Guard:
    """The version guard must fail fast (never hang) on Python 3.14+.

    Runs on every interpreter: it monkeypatches sys.version_info, so it does
    not need a working solver and cannot deadlock.
    """

    def test_raises_import_error_on_py314(self, monkeypatch):
        import sys as _sys
        from scoredrulesets.estimators import exact_cpusat as mod

        monkeypatch.setattr(mod.sys, "version_info", (3, 14, 0))
        clf = ExactCPSATClassifier(max_width=1, max_rules=1, max_atoms_per_class=1)
        with pytest.raises(ImportError, match="deadlock"):
            clf.fit(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 1]))

    def test_guard_passes_on_py313(self, monkeypatch):
        pytest.importorskip("ortools")
        from scoredrulesets.estimators import exact_cpusat as mod

        monkeypatch.setattr(mod.sys, "version_info", (3, 13, 0))
        # No ImportError from the guard; proceeds to ortools import (present
        # here because the module-level importorskip already ran).
        clf = ExactCPSATClassifier(
            max_width=1, max_rules=1, max_atoms_per_class=1,
            n_bins=2, time_limit_per_solve=2.0, random_state=0,
        )
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        clf.fit(X, y)
        assert clf.ruleset_ is not None


# ---------------------------------------------------------------------------
# Stage 1: enumeration unit tests (no solver needed)
# ---------------------------------------------------------------------------

class TestEnumerateCandidates:
    def test_width_one_and_two(self):
        # 2 features x 2 literals, samples: 00, 01, 10, 11
        P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        cands = enumerate_candidates(P, max_width=2, max_candidates=1000)
        masks = {c.mask for c in cands}
        # single literals present
        assert any(c.width == 1 for c in cands)
        # conjunction x0 & x1 covers exactly sample 3
        assert (1 << 3) in masks

    def test_empty_conjunction_dropped(self):
        # x and ¬x on the same feature can never co-occur
        P = np.array([[1.0, 0.0], [0.0, 1.0]])  # cols: (x<=t), (x>t)
        cands = enumerate_candidates(P, max_width=2, max_candidates=1000)
        for c in cands:
            assert c.mask != 0
            assert not (0 in c.lits and 1 in c.lits)

    def test_dedup_keeps_narrowest(self):
        # duplicate columns -> same coverage, only narrowest width kept per mask
        P = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cands = enumerate_candidates(P, max_width=2, max_candidates=1000)
        by_mask: dict[int, list[int]] = {}
        for c in cands:
            by_mask.setdefault(c.mask, []).append(c.width)
        for widths in by_mask.values():
            assert len(widths) == 1  # one candidate per distinct coverage

    def test_max_candidates_cap(self):
        rng = np.random.default_rng(0)
        P = rng.integers(0, 2, size=(60, 12)).astype(float)
        cands = enumerate_candidates(P, max_width=2, max_candidates=50)
        assert len(cands) <= 50


# ---------------------------------------------------------------------------
# Estimator end-to-end
# ---------------------------------------------------------------------------

def _binary_split(random_state: int = 42):
    X, y = load_wine(return_X_y=True)
    mask = y < 2
    return train_test_split(
        X[mask], y[mask], test_size=0.3, random_state=random_state,
        stratify=y[mask],
    )


@needs_ortools
class TestExactCPSATClassifier:
    def test_fit_predict_binary(self):
        X_train, X_test, y_train, y_test = _binary_split()
        clf = ExactCPSATClassifier(
            max_width=2, max_rules=2, max_atoms_per_class=3,
            n_bins=2, max_features=6, time_limit_per_solve=2.0, random_state=0,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        assert pred.shape == y_test.shape
        f1 = f1_score(y_test, pred, average="macro")
        print(f"\n[exact] Wine-binary F1={f1:.4f}")
        assert f1 > 0.5

    def test_ruleset_valid_and_certified(self):
        X_train, _, y_train, _ = _binary_split()
        clf = ExactCPSATClassifier(
            max_width=1, max_rules=2, max_atoms_per_class=2,
            n_bins=2, max_features=6, time_limit_per_solve=2.0, random_state=0,
        )
        clf.fit(X_train, y_train)
        rs = clf.to_ruleset()
        assert isinstance(rs, ScoredRuleSet)
        rs.validate()
        assert rs.metadata["certified"] is True
        # max_rules and the atom budget are PER CLASS: binary -> 2 classes,
        # so up to 2*max_rules non-default rules and 2*budget atoms overall.
        non_default = [r for r in rs.rules if r.atoms]
        assert len(non_default) <= 2 * 2
        assert sum(len(r.atoms) for r in non_default) <= 2 * 2
        # per-class budgets respected
        for cls in range(2):
            cls_rules = [r for r in non_default if r.scores[cls] == 1.0]
            assert len(cls_rules) <= 2
            assert sum(len(r.atoms) for r in cls_rules) <= 2

    def test_one_hot_scores(self):
        X_train, _, y_train, _ = _binary_split()
        clf = ExactCPSATClassifier(
            max_width=1, max_rules=2, max_atoms_per_class=2,
            n_bins=2, max_features=6, time_limit_per_solve=2.0, random_state=0,
        )
        clf.fit(X_train, y_train)
        rs = clf.to_ruleset()
        for rule in rs.rules:
            if rule.atoms:
                assert sorted(rule.scores) == [0.0, 1.0] or sorted(
                    rule.scores) == [0.0, 0.0]

    def test_multiclass_iris(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y)
        clf = ExactCPSATClassifier(
            max_width=2, max_rules=2, max_atoms_per_class=3,
            n_bins=2, max_features=6, time_limit_per_solve=2.0, random_state=0,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        assert set(pred).issubset(set(y))
        f1 = f1_score(y_test, pred, average="macro")
        print(f"\n[exact] Iris F1={f1:.4f}")
        assert f1 > 0.3

    def test_pareto_schedule_builds_archive(self):
        X_train, _, y_train, _ = _binary_split()
        clf = ExactCPSATClassifier(
            max_width=2, max_rules=2, max_atoms_per_class=2,
            pareto_schedule=[1, 2], n_bins=2, max_features=6,
            time_limit_per_solve=2.0, random_state=0,
        )
        clf.fit(X_train, y_train)
        assert hasattr(clf, "pareto_archive_")
        assert len(clf.pareto_archive_) >= 1  # budgets 1,2 -> distinct sizes
        for atoms, rs in clf.pareto_archive_.items():
            rs.validate()
            assert atoms == sum(len(r.atoms) for r in rs.rules)

    def test_max_features_subsample(self):
        X_train, X_test, y_train, y_test = _binary_split()
        clf = ExactCPSATClassifier(
            max_width=1, max_rules=1, max_atoms_per_class=1,
            n_bins=2, max_features=5, time_limit_per_solve=2.0, random_state=0,
        )
        clf.fit(X_train, y_train)
        # rules reference ORIGINAL feature indices from the subsample
        rs = clf.to_ruleset()
        feats = {int(a.feature[1:]) for r in rs.rules for a in r.atoms}
        assert feats.issubset(set(clf._pool_features_.tolist()))
        # predict consumes full-width X (atoms resolve against f0..f12)
        pred = clf.predict(X_test)
        assert pred.shape == y_test.shape

    def test_raises_on_setup_timeout(self, monkeypatch):
        from scoredrulesets import FitBudgetExceededError

        X_train, _, y_train, _ = _binary_split()
        ticks = iter(float(i) for i in range(2000))
        monkeypatch.setattr(time_budget_module.time, "monotonic", lambda: next(ticks))
        clf = ExactCPSATClassifier(
            max_width=1, max_rules=1, max_atoms_per_class=1,
            max_fit_seconds=0.5, random_state=0,
        )
        with pytest.raises(FitBudgetExceededError):
            clf.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Wrapper integration
# ---------------------------------------------------------------------------

@needs_ortools
class TestExactWrapper:
    def test_wrapper_backend_exact(self):
        X_train, X_test, y_train, y_test = _binary_split()
        clf = ScoredRuleSetClassifier(
            backend="exact",
            backend_params={
                "max_width": 1, "max_rules": 2, "max_atoms_per_class": 2,
                "n_bins": 2, "max_features": 6, "time_limit_per_solve": 2.0,
            },
            random_state=0,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        assert pred.shape == y_test.shape
        rs = clf.to_ruleset()
        assert rs.metadata["backend"] == "exact_cpusat"


class TestExactDispatch:
    """Dispatcher error message — runs without ortools installed."""

    def test_unknown_backend_error_mentions_exact(self):
        with pytest.raises(ValueError, match="'exact'"):
            ScoredRuleSetClassifier(backend="does_not_exist").fit(
                np.array([[0.0]]), np.array([0])
            )
