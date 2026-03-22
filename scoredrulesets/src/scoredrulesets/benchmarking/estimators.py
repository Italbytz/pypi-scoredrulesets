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
    # ...existing code...
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
        "wrapper_pittsburgh": EstimatorSpec(
            name="wrapper_pittsburgh",
            factory=lambda: ScoredRuleSetClassifier(
                backend="pittsburgh",
                backend_params={
                    "max_rules": 5,
                    "candidate_pool_size": 20,
                    "beam_width": 6,
                    "max_iterations": 12,
                    "validation_fraction": 0.2,
                    "complexity_penalty": 0.01,
                },
                random_state=0,
            ),
        ),
        "wrapper_pittsburgh_fast": EstimatorSpec(
            name="wrapper_pittsburgh_fast",
            factory=lambda: ScoredRuleSetClassifier(
                backend="pittsburgh",
                backend_params={
                    "max_rules": 4,
                    "candidate_pool_size": 12,
                    "beam_width": 4,
                    "max_iterations": 6,
                    "validation_fraction": 0.15,
                    "complexity_penalty": 0.012,
                },
                random_state=0,
            ),
        ),
        "wrapper_pittsburgh_strong": EstimatorSpec(
            name="wrapper_pittsburgh_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="pittsburgh",
                backend_params={
                    "max_rules": 6,
                    "candidate_pool_size": 32,
                    "beam_width": 10,
                    "max_iterations": 20,
                    "validation_fraction": 0.25,
                    "complexity_penalty": 0.008,
                },
                random_state=0,
            ),
        ),
        "wrapper_pittsburgh_diverse": EstimatorSpec(
            name="wrapper_pittsburgh_diverse",
            factory=lambda: ScoredRuleSetClassifier(
                backend="pittsburgh",
                backend_params={
                    "max_rules": 7,
                    "min_samples_leaf": 3,
                    "candidate_pool_size": 36,
                    "beam_width": 10,
                    "max_iterations": 18,
                    "validation_fraction": 0.25,
                    "complexity_penalty": 0.01,
                },
                random_state=0,
            ),
        ),
        "wrapper_michigan": EstimatorSpec(
            name="wrapper_michigan",
            factory=lambda: ScoredRuleSetClassifier(
                backend="michigan",
                backend_params={
                    "population_size": 60,
                    "epochs": 8,
                    "max_atoms_per_rule": 2,
                    "learning_rate": 0.08,
                    "mutation_rate": 0.08,
                    "covering_probability": 0.12,
                },
                random_state=0,
            ),
        ),
        "wrapper_michigan_fast": EstimatorSpec(
            name="wrapper_michigan_fast",
            factory=lambda: ScoredRuleSetClassifier(
                backend="michigan",
                backend_params={
                    "population_size": 36,
                    "epochs": 5,
                    "max_atoms_per_rule": 2,
                    "learning_rate": 0.1,
                    "mutation_rate": 0.05,
                    "covering_probability": 0.10,
                },
                random_state=0,
            ),
        ),
        "wrapper_michigan_strong": EstimatorSpec(
            name="wrapper_michigan_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="michigan",
                backend_params={
                    "population_size": 100,
                    "epochs": 14,
                    "max_atoms_per_rule": 3,
                    "learning_rate": 0.07,
                    "mutation_rate": 0.1,
                    "covering_probability": 0.15,
                    "min_rule_fitness": 0.015,
                },
                random_state=0,
            ),
        ),
        "wrapper_michigan_compact": EstimatorSpec(
            name="wrapper_michigan_compact",
            factory=lambda: ScoredRuleSetClassifier(
                backend="michigan",
                backend_params={
                    "population_size": 90,
                    "epochs": 14,
                    "max_atoms_per_rule": 2,
                    "learning_rate": 0.07,
                    "mutation_rate": 0.08,
                    "covering_probability": 0.14,
                    "min_rule_fitness": 0.03,
                    "max_final_rules": 20,
                },
                random_state=0,
            ),
        ),
        # GP naming:
        # - `gp` is the recommended strong benchmark default (residual covering).
        # - `gp_fast` keeps the earlier lightweight fast configuration.
        # - `gp_diverse` keeps the previous strong default with diverse final selection.
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
        "wrapper_logicgp": EstimatorSpec(
            name="wrapper_logicgp",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "flcw_macro",
                    "max_generations": 50,
                    "stagnation_generations": 20,
                    "population_size": 30,
                    "n_adaptations_per_gen": 6,
                    "n_bins": 5,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_logicgp_fast": EstimatorSpec(
            name="wrapper_logicgp_fast",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "flcw_macro",
                    "max_generations": 20,
                    "stagnation_generations": 10,
                    "population_size": 15,
                    "n_adaptations_per_gen": 6,
                    "n_bins": 4,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_logicgp_rlcw_macro": EstimatorSpec(
            name="wrapper_logicgp_rlcw_macro",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "rlcw_macro",
                    "max_generations": 500,
                    "stagnation_generations": 100,
                    "population_size": 50,
                    "n_adaptations_per_gen": 12,
                    "n_bins": 5,
                    "min_max_weight": 0.1,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_logicgp_rlcw_micro": EstimatorSpec(
            name="wrapper_logicgp_rlcw_micro",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "rlcw_micro",
                    "max_generations": 500,
                    "stagnation_generations": 100,
                    "population_size": 50,
                    "n_adaptations_per_gen": 12,
                    "n_bins": 5,
                    "min_max_weight": 0.1,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_logicgp_rlcw_fast": EstimatorSpec(
            name="wrapper_logicgp_rlcw_fast",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "rlcw_macro",
                    "max_generations": 100,
                    "stagnation_generations": 30,
                    "population_size": 30,
                    "n_adaptations_per_gen": 12,
                    "n_bins": 4,
                    "min_max_weight": 0.2,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_rulefit": EstimatorSpec(
            name="wrapper_rulefit",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulefit",
                backend_params={
                    "max_rules": 50,
                    "tree_size": 4,
                },
                random_state=0,
            ),
        ),
        "wrapper_rulefit_compact": EstimatorSpec(
            name="wrapper_rulefit_compact",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulefit",
                backend_params={
                    "max_rules": 20,
                    "tree_size": 3,
                },
                random_state=0,
            ),
        ),
    }
