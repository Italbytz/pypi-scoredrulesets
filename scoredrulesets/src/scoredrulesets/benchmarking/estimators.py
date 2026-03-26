from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
        "wrapper_rulelcs": EstimatorSpec(
            name="wrapper_rulelcs",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulelcs",
                backend_params={
                    "max_rules": 10,
                    "min_samples_leaf": 3,
                    "candidate_pool_size": 48,
                    "beam_width": 12,
                    "max_iterations": 24,
                    "validation_fraction": 0.25,
                    "complexity_penalty": 0.002,
                },
                random_state=0,
            ),
        ),
        "wrapper_rulelcs_strong": EstimatorSpec(
            name="wrapper_rulelcs_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulelcs",
                backend_params={
                    "max_rules": 15,
                    "min_samples_leaf": 3,
                    "candidate_pool_size": 64,
                    "beam_width": 16,
                    "max_iterations": 32,
                    "validation_fraction": 0.25,
                    "complexity_penalty": 0.001,
                    "sequential_covering": True,
                    "enable_compaction": True,
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
        "wrapper_rulekit_native": EstimatorSpec(
            name="wrapper_rulekit_native",
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
        "wrapper_exstracs": EstimatorSpec(
            name="wrapper_exstracs",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={"learning_iterations": 10000},
                random_state=0,
            ),
        ),
        # -- ExSTraCS pruned: conservative atom pruning (≤1.5% F1 loss, 20–77% reduction) --
        "wrapper_exstracs_pruned": EstimatorSpec(
            name="wrapper_exstracs_pruned",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={"learning_iterations": 10000},
                exstracs_params={
                    "conservative_prune": True,
                },
                random_state=0,
            ),
        ),
        # -- ExSTraCS + Lossy Rule Compaction (LRC): interval merge + conservative pruning (0–6% F1 loss, 29–98% reduction) --
        "wrapper_exstracs_compact": EstimatorSpec(
            name="wrapper_exstracs_compact",
            factory=lambda: ScoredRuleSetClassifier(
                backend="exstracs",
                backend_params={"learning_iterations": 10000},
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
        "wrapper_logicgp_strong": EstimatorSpec(
            name="wrapper_logicgp_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="logicgp",
                backend_params={
                    "trainer": "rlcw",
                    "f1_averaging": "macro",
                    "max_generations": 500,
                    "stagnation_generations": 80,
                    "population_size": 120,
                    "n_adaptations_per_gen": 20,
                    "n_bins": 5,
                    "max_fit_seconds": 240,
                    "random_state": 0,
                },
                random_state=0,
            ),
        ),
        "wrapper_rulenln_strong": EstimatorSpec(
            name="wrapper_rulenln_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulenln",
                backend_params={
                    "n_rules": 20,
                    "n_bins": 6,
                    "learning_rate": 0.15,
                    "l1_conj": 0.0003,
                    "l1_score": 0.00015,
                    "epochs": 600,
                    "early_stopping_rounds": 50,
                    "atom_threshold": 0.06,
                },
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
        "wrapper_rulegp_strong": EstimatorSpec(
            name="wrapper_rulegp_strong",
            factory=lambda: ScoredRuleSetClassifier(
                backend="rulegp",
                backend_params={
                    "population_size": 150,
                    "generations": 250,
                    "max_rules": 12,
                    "max_atoms_per_rule": 5,
                    "tournament_size": 4,
                    "early_stopping_rounds": 30,
                    "enable_compaction": True,
                },
                random_state=0,
            ),
        ),
    }

    # Merge any dynamically registered specs (external benchmarks etc.)
    specs.update(ESTIMATOR_SPECS)
    return specs

