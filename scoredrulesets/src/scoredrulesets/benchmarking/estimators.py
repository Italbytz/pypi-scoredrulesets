from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..estimators.nln import NeuralLogicNetClassifier
from ..estimators.rulekit_native import RuleKitNativeClassifier
from ..estimators.sklearn_wrapper import ScoredRuleSetClassifier


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    factory: Callable[[], object]


# Global mutable registry – external scripts (e.g. benchmark_logicgp_variants)
# can add specs here at runtime.  `default_estimator_specs()` merges
# these into its result so that `run_benchmarks()` picks them up.
ESTIMATOR_SPECS: dict[str, EstimatorSpec] = {}


def default_estimator_specs() -> dict[str, EstimatorSpec]:
    # ...existing code...
    specs = {
        "wrapper_cart": EstimatorSpec(
            name="wrapper_cart",
            factory=lambda: ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
                random_state=0,
            ),
        ),
        "wrapper_cart_pruned": EstimatorSpec(
            name="wrapper_cart_pruned",
            factory=lambda: ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": 4},
                transform_params={
                    "prune_atoms": True,
                },
                random_state=0,
            ),
        ),
        "wrapper_hs": EstimatorSpec(
            name="wrapper_hs",
            factory=lambda: ScoredRuleSetClassifier(backend="hs", random_state=0),
        ),
        "wrapper_hs_pruned": EstimatorSpec(
            name="wrapper_hs_pruned",
            factory=lambda: ScoredRuleSetClassifier(
                backend="hs",
                transform_params={
                    "prune_atoms": True,
                },
                random_state=0,
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
        "wrapper_rulekit": EstimatorSpec(
            name="wrapper_rulekit",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulekit",
                backend_params={},
                random_state=0,
            ),
        ),
        # -- RuleKit pure-Python re-implementation (no JVM required) --
        "wrapper_rulekit_native": EstimatorSpec(
            name="wrapper_rulekit_native",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulekit_native",
                backend_params={
                    "max_rules": 20,
                    "max_conditions": 5,
                    "min_samples_leaf": 5,
                    "enable_pruning": True,
                },
                random_state=0,
            ),
        ),
        "wrapper_rulekit_native_fast": EstimatorSpec(
            name="wrapper_rulekit_native_fast",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulekit_native",
                backend_params={
                    "max_rules": 10,
                    "max_conditions": 3,
                    "min_samples_leaf": 5,
                    "enable_pruning": True,
                },
                random_state=0,
            ),
        ),
        "wrapper_rulekit_native_strong": EstimatorSpec(
            name="wrapper_rulekit_native_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulekit_native",
                backend_params={
                    "max_rules": 30,
                    "max_conditions": 7,
                    "min_samples_leaf": 3,
                    "enable_pruning": True,
                    "pruning_fraction": 0.25,
                },
                random_state=0,
            ),
        ),
        "rulekit_native": EstimatorSpec(
            name="rulekit_native",
            factory=lambda: RuleKitNativeClassifier(
                max_rules=20,
                max_conditions=5,
                min_samples_leaf=5,
                enable_pruning=True,
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
        # -- ExSTraCS pruned: conservative atom pruning (≤1.5% F1 loss, 20–77% reduction) --
        "wrapper_exstracs_pruned": EstimatorSpec(
            name="wrapper_exstracs_pruned",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={},
                exstracs_params={
                    "conservative_prune": True,
                },
                random_state=0,
            ),
        ),
        # -- ExSTraCS compact: interval merge + conservative (0–6% F1 loss, 29–98% reduction) --
        "wrapper_exstracs_compact": EstimatorSpec(
            name="wrapper_exstracs_compact",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={},
                exstracs_params={
                    "interval_merge": True,
                    "interval_merge_iou_threshold": 0.3,
                    "conservative_prune": True,
                },
                random_state=0,
            ),
        ),
        "wrapper_logicgp": EstimatorSpec(
            name="wrapper_logicgp",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "rlcw",
                    "f1_averaging": "micro",
                    "max_generations": 200,
                    "stagnation_generations": 40,
                    "population_size": 50,
                    "n_adaptations_per_gen": 12,
                    "n_bins": 5,
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
                    "trainer": "rlcw",
                    "f1_averaging": "macro",
                    "max_generations": 200,
                    "stagnation_generations": 40,
                    "population_size": 50,
                    "n_adaptations_per_gen": 12,
                    "n_bins": 5,
                    "max_fit_seconds": 240,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_logicgp_flcw": EstimatorSpec(
            name="wrapper_logicgp_flcw",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "flcw",
                    "f1_averaging": "micro",
                    "max_generations": 200,
                    "stagnation_generations": 40,
                    "population_size": 50,
                    "n_adaptations_per_gen": 8,
                    "n_bins": 5,
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
        # ------------------------------------------------------------------
        # Neural Logic Network (NLN) – differentiable logic rule learning
        # ------------------------------------------------------------------
        "wrapper_nln": EstimatorSpec(
            name="wrapper_nln",
            factory=lambda: ScoredRuleSetClassifier(
                backend="nln",
                backend_params={
                    "n_rules": 12,
                    "n_bins": 5,
                    "learning_rate": 0.3,
                    "l1_conj": 0.002,
                    "l1_score": 0.001,
                    "epochs": 300,
                    "early_stopping_rounds": 30,
                    "atom_threshold": 0.1,
                },
                random_state=0,
            ),
        ),
        "wrapper_nln_fast": EstimatorSpec(
            name="wrapper_nln_fast",
            factory=lambda: ScoredRuleSetClassifier(
                backend="nln",
                backend_params={
                    "n_rules": 8,
                    "n_bins": 4,
                    "learning_rate": 0.4,
                    "l1_conj": 0.005,
                    "l1_score": 0.003,
                    "epochs": 100,
                    "early_stopping_rounds": 15,
                    "atom_threshold": 0.1,
                },
                random_state=0,
            ),
        ),
        "wrapper_nln_strong": EstimatorSpec(
            name="wrapper_nln_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="nln",
                backend_params={
                    "n_rules": 20,
                    "n_bins": 6,
                    "learning_rate": 0.2,
                    "l1_conj": 0.001,
                    "l1_score": 0.0005,
                    "epochs": 500,
                    "early_stopping_rounds": 40,
                    "atom_threshold": 0.08,
                },
                random_state=0,
            ),
        ),
        "nln_native": EstimatorSpec(
            name="nln_native",
            factory=lambda: NeuralLogicNetClassifier(
                n_rules=12,
                n_bins=5,
                learning_rate=0.3,
                l1_conj=0.002,
                l1_score=0.001,
                epochs=300,
                early_stopping_rounds=30,
                atom_threshold=0.1,
                random_state=0,
            ),
        ),
        # ------------------------------------------------------------------
        # Multiplexer-optimierte Varianten (Ziel: F1=1.0 auf vollem Datensatz)
        # ------------------------------------------------------------------
        "wrapper_logicgp_mux": EstimatorSpec(
            name="wrapper_logicgp_mux",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "rlcw",
                    "f1_averaging": "micro",
                    "max_generations": 10_000,
                    "stagnation_generations": 500,
                    "population_size": 50,
                    "n_adaptations_per_gen": 12,
                    "n_bins": 2,  # binaere Features → 2 Bins
                    "validation_fraction": 0.0,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_logicgp_mux_rlcw_macro": EstimatorSpec(
            name="wrapper_logicgp_mux_rlcw_macro",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "rlcw",
                    "f1_averaging": "macro",
                    "max_generations": 10_000,
                    "stagnation_generations": 500,
                    "population_size": 80,
                    "n_adaptations_per_gen": 12,
                    "n_bins": 2,
                    "min_max_weight": 0.1,
                    "validation_fraction": 0.0,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_cart_mux": EstimatorSpec(
            name="wrapper_cart_mux",
            factory=lambda: ScoredRuleSetClassifier(
                backend="cart",
                backend_params={"max_depth": None},  # unbegrenzte Tiefe
                random_state=0,
            ),
        ),
        # -- ruleGP: full-ruleset GP with NSGA-II --
        "wrapper_rulegp": EstimatorSpec(
            name="wrapper_rulegp",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulegp",
                backend_params={
                    "population_size": 60,
                    "generations": 100,
                    "max_rules": 8,
                    "max_atoms_per_rule": 4,
                    "enable_compaction": True,
                },
                random_state=0,
            ),
        ),
    }

    # Merge any dynamically registered specs (external benchmarks etc.)
    specs.update(ESTIMATOR_SPECS)
    return specs

