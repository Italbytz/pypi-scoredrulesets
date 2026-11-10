"""logicGP and GPAS - ScoredRuleSet estimator wrappers.

Based on the paper:
  Nunkesser (GECCO'25): "logicGP -- A Framework for Literal Based
  Classification with a Focus on Software Architecture and Open Source
  Implementation"
  DOI: 10.1145/3712255.3734300

and:
  Nunkesser et al. (2007): "Detecting high-order interactions of
  single nucleotide polymorphisms using genetic programming"
  Bioinformatics, 23(24):3280–3288.

The core evolutionary algorithms and bitset-parallel engines are implemented in
and imported from ``evolutionary_ml.logicgp``. This module provides the
scikit-learn BaseRuleSetEstimator interface producing validated ``ScoredRuleSet``
models (``self.ruleset_``).
"""

from __future__ import annotations

from typing import Any, Callable, Union
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from evolutionary_ml.logicgp import (
    Fitness as _Fitness,
    FitnessRLCW as _FitnessRLCW,
    GPASClassifier as _BaseGPASClassifier,
    LogicGPClassifier as _BaseLogicGPClassifier,
    Monomial as _Monomial,
    Polynomial as _Polynomial,
    SetLiteral as _SetLiteral,
    crossover as _crossover,
    discretize_features as _discretize_features,
    final_model_selection as _final_model_selection,
    generate_literals as _generate_literals,
    generate_singleton_literals as _generate_singleton_literals,
    mut_delete_literal as _mut_delete_literal,
    mut_delete_literal_min2 as _mut_delete_literal_min2,
    mut_delete_monomial as _mut_delete_monomial,
    mut_insert_literal as _mut_insert_literal,
    mut_insert_literal_distinct_feature as _mut_insert_literal_distinct_feature,
    mut_insert_monomial as _mut_insert_monomial,
    mut_insert_monomial_pair as _mut_insert_monomial_pair,
    mut_replace_literal as _mut_replace_literal,
    mut_replace_literal_distinct_feature as _mut_replace_literal_distinct_feature,
    pareto_front as _pareto_front,
    pareto_front_rlcw as _pareto_front_rlcw,
    select_model_best_f1 as _select_model_best_f1,
    select_model_shortest_zero_train_mcr as _select_model_shortest_zero_train_mcr,
    tournament_trim as _tournament_trim,
)
from evolutionary_ml.logicgp.evaluator import (
    compute_weights_fallback as _compute_weights,
    evaluate_fitness_fallback as _evaluate_fitness,
    evaluate_fitness_rlcw_fallback as _evaluate_fitness_rlcw,
)

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from ._time_budget import (
    FitBudgetExceededError,
    deadline_reached,
    resolve_deadline,
)
from .atom_space import LogicGPEncodingStrategy
from .base import BaseRuleSetEstimator

LiteralGenerator = Callable[[np.ndarray], list[_SetLiteral]]
ModelSelector = Callable[..., _Polynomial]
FitnessEvaluator = Callable[..., Union[_Fitness, _FitnessRLCW]]


class LogicGPClassifier(_BaseLogicGPClassifier, BaseRuleSetEstimator):
    """logicGP Python implementation (FLCW and RLCW variants) producing ScoredRuleSet."""

    def __init__(
        self,
        trainer: str = "rlcw",
        f1_averaging: str = "micro",
        max_generations: int = 10_000,
        stagnation_generations: int = 500,
        n_bins: int = 5,
        min_max_weight: float = 0.0,
        min_improvement_pct: float = 0.01,
        population_size: int | None = None,
        n_adaptations_per_gen: int = 6,
        tournament_size: int = 5,
        max_model_size: int | None = None,
        validation_fraction: float = 0.0,
        literal_generator: str | LiteralGenerator = "full",
        model_selection: str | ModelSelector = "paper",
        fitness_evaluator: str | FitnessEvaluator | None = None,
        max_fit_seconds: float | None = None,
        feature_encoding_strategy: LogicGPEncodingStrategy = "auto_low_cardinality",
        random_state: int | None = None,
    ):
        super().__init__(
            trainer=trainer,
            f1_averaging=f1_averaging,
            max_generations=max_generations,
            stagnation_generations=stagnation_generations,
            n_bins=n_bins,
            min_max_weight=min_max_weight,
            min_improvement_pct=min_improvement_pct,
            population_size=population_size,
            n_adaptations_per_gen=n_adaptations_per_gen,
            tournament_size=tournament_size,
            max_model_size=max_model_size,
            validation_fraction=validation_fraction,
            literal_generator=literal_generator,
            model_selection=model_selection,
            fitness_evaluator=fitness_evaluator,
            max_fit_seconds=max_fit_seconds,
            feature_encoding_strategy=feature_encoding_strategy,
            random_state=random_state,
        )

    def fit(self, X, y):
        super().fit(X, y)
        n_classes = len(self.classes_)
        self.ruleset_ = self._poly_to_ruleset(self.best_poly_, n_classes)
        self.ruleset_.validate()
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr: np.ndarray = np.asarray(check_array(X, dtype=None))
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}."
            )
        X_disc, _, _ = _discretize_features(
            X_arr,
            fitted_binners=self._binners_,
            cat_masks=self._cat_masks_,
            strategy=self.feature_encoding_strategy,
        )
        return predict_from_ruleset(self.ruleset_, X_disc)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr: np.ndarray = np.asarray(check_array(X, dtype=None))
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}."
            )
        X_disc, _, _ = _discretize_features(
            X_arr,
            fitted_binners=self._binners_,
            cat_masks=self._cat_masks_,
            strategy=self.feature_encoding_strategy,
        )
        return predict_proba_from_ruleset(self.ruleset_, X_disc)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

    def extract_interactions(
        self,
        *,
        min_occurrences: int = 10,
        min_ratio: float = 0.1,
        out_dot=None,
        out_csv=None,
    ) -> dict:
        from scoredrulesets.analysis.logicgp_interactions import extract_interactions
        return extract_interactions(
            self,
            min_occurrences=min_occurrences,
            min_ratio=min_ratio,
            out_dot=out_dot,
            out_csv=out_csv,
        )

    def extract_pareto_interactions(
        self,
        *,
        feature_names=None,
        min_occurrences: int = 1,
        min_ratio: float = 0.05,
        backbone_threshold: float = 0.5,
        out_dot=None,
        out_csv=None,
    ):
        from scoredrulesets.analysis.pareto_interactions import extract_pareto_interactions
        return extract_pareto_interactions(
            self,
            feature_names=feature_names,
            min_occurrences=min_occurrences,
            min_ratio=min_ratio,
            backbone_threshold=backbone_threshold,
            out_dot=out_dot,
            out_csv=out_csv,
        )

    def _poly_to_ruleset(self, poly: _Polynomial, n_classes: int) -> ScoredRuleSet:
        rules: list[Rule] = []

        rules.append(Rule(
            atoms=[],
            scores=poly.default_weights.tolist(),
            rule_id="logicgp_default",
            metadata={"source": "logicgp", "kind": "default"},
        ))

        for i, mon in enumerate(poly.monomials):
            atoms = self._monomial_to_atoms(mon)
            rules.append(Rule(
                atoms=atoms,
                scores=mon.weights.tolist(),
                rule_id=f"logicgp_{i}",
                metadata={"source": "logicgp", "kind": "monomial"},
            ))

        feature_names = self.feature_names_in_.tolist()
        use_rlcw, f1_avg = self._resolve_trainer_config()

        return ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=feature_names,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            rules=rules,
            metadata={
                "source": "logicgp",
                "trainer": "rlcw" if use_rlcw else "flcw",
                "f1_averaging": f1_avg,
                "n_monomials": len(poly.monomials),
                "model_size": poly.size,
                "max_generations": self.max_generations,
                "n_bins": self.n_bins,
                "feature_encoding_strategy": self.feature_encoding_strategy,
                "population_size": self.population_size,
                "n_adaptations_per_gen": self.n_adaptations_per_gen,
                "max_model_size": self.max_model_size,
                "literal_generator": self._strategy_name(self.literal_generator),
                "model_selection": self._strategy_name(self.model_selection),
            },
        )

    def _monomial_to_atoms(self, mon: _Monomial) -> list[Atom]:
        atoms: list[Atom] = []
        for lit in mon.literals:
            fname = (
                self.feature_names_in_[lit.feature_idx]
                if lit.feature_idx < len(self.feature_names_in_)
                else f"f{lit.feature_idx}"
            )
            cat_list = sorted(lit.category_set, key=lambda x: (str(type(x)), x))
            if len(cat_list) == 1:
                atoms.append(Atom(feature=str(fname), op="==", value=cat_list[0]))
            else:
                atoms.append(Atom(feature=str(fname), op="in", value=cat_list))
        return atoms

    @staticmethod
    def _strategy_name(strategy) -> str:
        if isinstance(strategy, str):
            return strategy
        if callable(strategy):
            return getattr(strategy, "__name__", repr(strategy))
        return repr(strategy)


class GPASClassifier(LogicGPClassifier):
    """GPAS -- Genetic Programming for Association Studies producing ScoredRuleSet."""

    def __init__(
        self,
        max_generations: int = 10_000,
        stagnation_generations: int = 500,
        n_bins: int = 5,
        min_improvement_pct: float = 0.01,
        population_size: int | None = None,
        n_adaptations_per_gen: int = 6,
        tournament_size: int = 5,
        max_model_size: int | None = None,
        validation_fraction: float = 0.0,
        max_fit_seconds: float | None = None,
        model_selection: str | ModelSelector = "paper",
        feature_encoding_strategy: LogicGPEncodingStrategy = "auto_low_cardinality",
        random_state: int | None = None,
    ):
        super().__init__(
            trainer="flcw",
            f1_averaging="micro",
            max_generations=max_generations,
            stagnation_generations=stagnation_generations,
            n_bins=n_bins,
            min_max_weight=0.0,
            min_improvement_pct=min_improvement_pct,
            population_size=population_size,
            n_adaptations_per_gen=n_adaptations_per_gen,
            tournament_size=tournament_size,
            max_model_size=max_model_size,
            validation_fraction=validation_fraction,
            literal_generator="full",
            model_selection=model_selection,
            fitness_evaluator=None,
            max_fit_seconds=max_fit_seconds,
            feature_encoding_strategy=feature_encoding_strategy,
            random_state=random_state,
        )

    def fit(self, X, y):
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(
                f"GPASClassifier requires exactly 2 classes, got {len(labels)}. "
                "Use LogicGPClassifier for multi-class problems."
            )
        return super().fit(X, y)

    _init_population = _BaseGPASClassifier._init_population
    _build_mutation_ops = _BaseGPASClassifier._build_mutation_ops
