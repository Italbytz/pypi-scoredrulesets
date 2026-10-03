"""Unit tests for GreedyParetoClassifier and GreedyCascadedRegressor."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression

from scoredrulesets.estimators.greedy_pareto import GreedyCascadedRegressor, GreedyParetoClassifier


def test_greedy_pareto_classifier_iris():
    X, y = load_iris(return_X_y=True)
    clf = GreedyParetoClassifier(max_complexity=8, beam_width=3, preference="balanced")
    clf.fit(X, y)

    # Check that ruleset is fitted
    ruleset = clf.to_ruleset()
    assert ruleset is not None
    assert len(ruleset.rules) >= 1

    # Check Pareto archive
    assert hasattr(clf, "pareto_archive_")
    assert len(clf.pareto_archive_) >= 1

    # Predictions
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(clf.classes_))

    # Probabilities
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)


def test_greedy_pareto_classifier_preferences():
    X, y = make_classification(
        n_samples=80, n_features=6, n_informative=4, n_classes=2, random_state=42
    )

    clf_compact = GreedyParetoClassifier(max_complexity=8, preference="compact").fit(X, y)
    clf_accuracy = GreedyParetoClassifier(max_complexity=8, preference="accuracy").fit(X, y)

    atoms_compact = sum(len(r.atoms) for r in clf_compact.to_ruleset().rules)
    atoms_accuracy = sum(len(r.atoms) for r in clf_accuracy.to_ruleset().rules)

    assert atoms_compact <= atoms_accuracy


def test_greedy_cascaded_regressor():
    X, y = make_regression(n_samples=100, n_features=6, noise=0.1, random_state=42)
    reg = GreedyCascadedRegressor(k_stage1=3, k_stage2=3, beam_width=4)
    reg.fit(X, y)

    ruleset = reg.to_ruleset()
    assert ruleset is not None
    assert ruleset.task_type == "regression"
    assert len(ruleset.rules) >= 2

    # Check predictions
    preds = reg.predict(X)
    assert len(preds) == len(y)
    # Basic sanity: correlation with true target should be positive
    corr = np.corrcoef(y, preds)[0, 1]
    assert corr > 0.5


def test_archive_pairing_invariant():
    """Every archived rule set must reproduce its recorded train_macro_f1.

    Regression guard for the Op-B pairing bug: new_rules were sorted while
    new_masks kept insertion order, so rule<->weight vectors were scrambled
    for multi-rule candidates whose sorted order differed from insertion
    order.  The stored ScoredRuleSet then predicted differently from the
    F1 recorded in its metadata.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import f1_score

    from scoredrulesets.runtime import predict as predict_from_ruleset

    X, y = load_breast_cancer(return_X_y=True)
    clf = GreedyParetoClassifier(random_state=0)
    clf.fit(X, y)
    assert clf.pareto_archive_, "empty archive"
    for comp, rs in clf.pareto_archive_.items():
        recorded = rs.metadata.get("train_macro_f1")
        assert recorded is not None
        actual = f1_score(y, predict_from_ruleset(rs, X), average="macro")
        assert abs(actual - recorded) < 0.01, (
            f"archive[{comp}]: stored ruleset reproduces F1 {actual:.4f} "
            f"but metadata records {recorded:.4f} (rule<->weight pairing bug)"
        )
