import numpy as np
from sklearn.datasets import load_iris

from scoredrulesets import NativeScoredRuleSetClassifier


def test_native_estimator_fit_predict_and_ruleset():
    X, y = load_iris(return_X_y=True)
    clf = NativeScoredRuleSetClassifier(aggregation="softmax_sum")
    clf.fit(X, y)

    pred = clf.predict(X[:7])
    proba = clf.predict_proba(X[:7])
    ruleset = clf.to_ruleset()

    assert pred.shape == (7,)
    assert proba.shape == (7, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert len(ruleset.rules) == 1
    assert ruleset.rules[0].rule_id == "native_default_prior"

