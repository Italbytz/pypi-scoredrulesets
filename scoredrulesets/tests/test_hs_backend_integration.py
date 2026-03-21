import pytest
from sklearn.datasets import load_iris

from scoredrulesets import ScoredRuleSetClassifier


@pytest.mark.hs
def test_hs_backend_fit_if_available():
    pytest.importorskip("imodels")

    X, y = load_iris(return_X_y=True)
    clf = ScoredRuleSetClassifier(backend="hs", random_state=0)

    try:
        clf.fit(X, y)
    except ImportError as exc:
        pytest.skip(f"HS-Klasse in imodels nicht verfuegbar: {exc}")

    pred = clf.predict(X[:5])
    assert pred.shape == (5,)
    assert len(clf.to_ruleset().rules) > 0

