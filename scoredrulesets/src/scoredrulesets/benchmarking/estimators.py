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
        # GP-Nomenklatur:
        # - `gp` ist der empfohlene starke Default fuer Benchmarks (residual covering).
        # - `gp_fast` behaelt die fruehere, schnelle Light-Konfiguration.
        # - `gp_diverse` behaelt den vorherigen starken Default mit diverser Endauswahl.
        "gp_fast": EstimatorSpec(
            name="gp_fast",
            factory=lambda: GeneticScoredRuleSetClassifier(
                population_size=25,
                generations=10,
                max_rules=8,
                validation_fraction=0.2,
                early_stopping_rounds=3,
                random_state=0,
            ),
        ),
        "gp": EstimatorSpec(
            name="gp",
            factory=lambda: GeneticScoredRuleSetClassifier(
                population_size=100,
                generations=60,
                max_rules=5,
                max_atoms_per_rule=3,
                selection_mode="pareto",
                final_rule_selection="diverse",
                evolution_fitness_mode="residual_covering",
                evolution_context_size=3,
                residual_focus_weight=0.35,
                validation_fraction=0.25,
                early_stopping_rounds=12,
                complexity_penalty=0.05,
                class_balance_weight=0.35,
                random_state=0,
            ),
        ),
        "gp_diverse": EstimatorSpec(
            name="gp_diverse",
            factory=lambda: GeneticScoredRuleSetClassifier(
                population_size=100,
                generations=60,
                max_rules=5,
                max_atoms_per_rule=3,
                selection_mode="pareto",
                final_rule_selection="diverse",
                validation_fraction=0.25,
                early_stopping_rounds=12,
                complexity_penalty=0.05,
                class_balance_weight=0.35,
                random_state=0,
            ),
        ),
        "gp_residual": EstimatorSpec(
            name="gp_residual",
            factory=lambda: GeneticScoredRuleSetClassifier(
                population_size=100,
                generations=60,
                max_rules=5,
                max_atoms_per_rule=3,
                selection_mode="pareto",
                final_rule_selection="diverse",
                evolution_fitness_mode="residual_covering",
                evolution_context_size=3,
                residual_focus_weight=0.35,
                validation_fraction=0.25,
                early_stopping_rounds=12,
                complexity_penalty=0.05,
                class_balance_weight=0.35,
                random_state=0,
            ),
        ),
        "gp_contrib": EstimatorSpec(
            name="gp_contrib",
            factory=lambda: GeneticScoredRuleSetClassifier(
                population_size=100,
                generations=60,
                max_rules=5,
                max_atoms_per_rule=3,
                selection_mode="pareto",
                final_rule_selection="contribution",
                validation_fraction=0.25,
                early_stopping_rounds=12,
                complexity_penalty=0.05,
                class_balance_weight=0.35,
                random_state=0,
            ),
        ),
        "wrapper_rulekit": EstimatorSpec(
            name="wrapper_rulekit",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulekit",
                backend_params={},
                random_state=0,
            ),
        ),
        "wrapper_exstracs": EstimatorSpec(
            name="wrapper_exstracs",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={},
                random_state=0,
            ),
        ),
        "wrapper_exstracs_shrink_conservative": EstimatorSpec(
            name="wrapper_exstracs_shrink_conservative",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={},
                exstracs_params={
                    "conservative_prune": True,
                },
                random_state=0,
            ),
        ),
        "wrapper_exstracs_shrink_aggressive": EstimatorSpec(
            name="wrapper_exstracs_shrink_aggressive",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={},
                exstracs_params={
                    "aggressive_prune": True,
                    "max_f1_loss": 0.01,  # 1% F1-Verlust akzeptabel
                },
                random_state=0,
            ),
        ),
        "wrapper_exstracs_shrink_filter": EstimatorSpec(
            name="wrapper_exstracs_shrink_filter",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={},
                exstracs_params={
                    "filter_weak_rules": True,
                    "min_fitness_percentile": 0.2,  # Behalte top 80%
                },
                random_state=0,
            ),
        ),
        "wrapper_exstracs_shrink_all": EstimatorSpec(
            name="wrapper_exstracs_shrink_all",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={},
                exstracs_params={
                    "conservative_prune": True,
                    "filter_weak_rules": True,
                    "consolidate_similar": True,
                    "aggressive_prune": True,
                    "max_f1_loss": 0.01,
                },
                random_state=0,
            ),
        ),
    }
