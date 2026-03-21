from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..estimators.gp_native import GeneticScoredRuleSetClassifier
from ..estimators.native import NativeScoredRuleSetClassifier
from ..estimators.sklearn_wrapper import ScoredRuleSetClassifier


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    factory: Callable[[], object]


def default_estimator_specs() -> dict[str, EstimatorSpec]:
    return {
        "wrapper_cart": EstimatorSpec(
            name="wrapper_cart",
            factory=lambda: ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
                random_state=0,
            ),
        ),
        "wrapper_hs": EstimatorSpec(
            name="wrapper_hs",
            factory=lambda: ScoredRuleSetClassifier(backend="hs", random_state=0),
        ),
        "native": EstimatorSpec(
            name="native",
            factory=lambda: NativeScoredRuleSetClassifier(
                max_rules=8,
                min_samples_leaf=3,
            ),
        ),
        "gp": EstimatorSpec(
            name="gp",
            factory=lambda: GeneticScoredRuleSetClassifier(
                population_size=25,
                generations=10,
                max_rules=8,
                validation_fraction=0.2,
                early_stopping_rounds=3,
                random_state=0,
            ),
        ),
    }

