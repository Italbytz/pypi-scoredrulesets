from sklearn.utils.estimator_checks import parametrize_with_checks

from scoredrulesets import ScoredRuleSetClassifier


@parametrize_with_checks(
    [
        ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 2},
            random_state=0,
        )
    ]
)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)

