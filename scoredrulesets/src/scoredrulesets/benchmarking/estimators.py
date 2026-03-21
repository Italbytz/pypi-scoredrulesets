from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..estimators.gp_native import GeneticScoredRuleSetClassifier
from ..estimators.native import NativeScoredRuleSetClassifier
from ..estimators.sklearn_wrapper import ScoredRuleSetClassifier
from ..estimators.tree_transform import TreeTransformParams


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
        "wrapper_cart_pruned_λ1.5": EstimatorSpec(
            name="wrapper_cart_pruned_λ1.5",
            factory=lambda: ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
                transform_params={
                    "prune_atoms": True,
                    "prune_lambda": 1.5,
                },
                random_state=0,
            ),
        ),
        "wrapper_cart_pruned_λ2.0": EstimatorSpec(
            name="wrapper_cart_pruned_λ2.0",
            factory=lambda: ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
                transform_params={
                    "prune_atoms": True,
                    "prune_lambda": 2.0,
                },
                random_state=0,
            ),
        ),
        "wrapper_cart_pruned_λ3.0": EstimatorSpec(
            name="wrapper_cart_pruned_λ3.0",
            factory=lambda: ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
                transform_params={
                    "prune_atoms": True,
                    "prune_lambda": 3.0,
                },
                random_state=0,
            ),
        ),
        "wrapper_hs": EstimatorSpec(
            name="wrapper_hs",
            factory=lambda: ScoredRuleSetClassifier(backend="hs", random_state=0),
        ),
        "wrapper_hs_pruned_λ2.0": EstimatorSpec(
            name="wrapper_hs_pruned_λ2.0",
            factory=lambda: ScoredRuleSetClassifier(
                backend="hs",
                transform_params={
                    "prune_atoms": True,
                    "prune_lambda": 2.0,
                },
                random_state=0,
            ),
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



