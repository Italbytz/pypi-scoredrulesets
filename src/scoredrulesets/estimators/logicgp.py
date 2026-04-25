"""
logicGP - Python implementation (logicGP-FLCW variant)
=======================================================
Based on the paper:
  Nunkesser (GECCO'25): "logicGP -- A Framework for Literal Based
  Classification with a Focus on Software Architecture and Open Source
  Implementation"
  DOI: 10.1145/3712255.3734300

and the C# reference repository:
  https://github.com/Italbytz/nuget-adapters-algorithms-ea

This module implements the logicGP FLCW macro/micro variant for
multi-class classification with categorical or discretized
continuous features.

Prediction model
----------------
  M_hat = sum_i  w_i  * prod_j ( 1 if l_{i,j}(x) else 0 )
    Y_hat = M_hat  if any monomial fired, else w_0
  G_hat = argmax_c Y_hat[c]

Weights (FLCW)
---------------
    w_i[c] = P(class=c | monomial i fires)  (= relative class frequency
                        among training instances where monomial i is true)
    w_0[c] = relative class frequency among instances where
                        NO monomial fires

GP algorithm
--------------
  Initialpopulation: ein Individuum pro Literal
  Pro Generation:
    - 2 Eltern -> Crossover -> 1 Kind
    - 5 Mutationstypen, je 1 Individuum -> 5 neue Individuen
        - Reproduction: all current individuals
        - Selection: Pareto-dominance selection (keep non-dominated)
    Fitness: per-class recall + model size (number of distinct literals)
    Termination: max_generations or stagnation
    Final model selection: best model per size, then selection strategy
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Union

import numpy as np
from sklearn.metrics import f1_score as _f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .atom_space import LogicGPEncodingStrategy
from .atom_space import discretize_logicgp_features
from .base import BaseRuleSetEstimator


# ---------------------------------------------------------------------------
# SetLiteral
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SetLiteral:
    """X_feature in category_set."""
    feature_idx: int
    category_set: frozenset
    all_categories: tuple  # sorted tuple of all categories for this feature

    def evaluate(self, value: Any) -> bool:
        return value in self.category_set

    def __repr__(self) -> str:
        return f"f{self.feature_idx} in {set(self.category_set)}"


# ---------------------------------------------------------------------------
# Monomial and Polynomial
# ---------------------------------------------------------------------------

@dataclass
class _Monomial:
    """Conjunction of SetLiterals with class weight vector."""
    literals: list[_SetLiteral]
    weights: np.ndarray  # shape (n_classes,), computed from training dist

    @property
    def size(self) -> int:
        return len(self.literals)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Returns weight vector if all literals fire, else zeros."""
        for lit in self.literals:
            if not lit.evaluate(x[lit.feature_idx]):
                return np.zeros_like(self.weights)
        return self.weights.copy()

    def fires(self, x: np.ndarray) -> bool:
        return all(lit.evaluate(x[lit.feature_idx]) for lit in self.literals)

    def fires_mask(self, X: np.ndarray) -> np.ndarray:
        mask = np.ones(X.shape[0], dtype=bool)
        for lit in self.literals:
            col = X[:, lit.feature_idx]
            # Vectorized: np.isin statt Python-Loop pro Sample
            cats = np.array(sorted(lit.category_set))
            mask &= np.isin(col, cats)
        return mask

    def clone(self) -> "_Monomial":
        return _Monomial(
            literals=list(self.literals),
            weights=self.weights.copy(),
        )


@dataclass
class _Polynomial:
    """
        Weighted polynomial: list of monomials + default weights w_0.
        Prediction:
            M_hat = sum of monomial weights for fired monomials
            Y_hat = M_hat if any monomial fired, else w_0
      G_hat = argmax(Y_hat)
    """
    monomials: list[_Monomial]
    default_weights: np.ndarray  # w_0, shape (n_classes,)

    @property
    def size(self) -> int:
        """Number of distinct literals in the polynomial."""
        all_lits: set[_SetLiteral] = set()
        for m in self.monomials:
            all_lits.update(m.literals)
        return len(all_lits)

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(self.default_weights)
        any_fired = False
        for mon in self.monomials:
            scores = mon.evaluate(x)
            if np.any(scores > 0):
                result += scores
                any_fired = True
        return result if any_fired else self.default_weights.copy()

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        n_classes = len(self.default_weights)
        scores = np.tile(self.default_weights, (n, 1))  # (n, n_classes)
        any_fired = np.zeros(n, dtype=bool)
        accum = np.zeros((n, n_classes), dtype=float)
        for mon in self.monomials:
            mask = mon.fires_mask(X)  # (n,)
            if mask.any():
                any_fired |= mask
                accum[mask] += mon.weights
        scores[any_fired] = accum[any_fired]
        return np.argmax(scores, axis=1)

    def clone(self) -> "_Polynomial":
        return _Polynomial(
            monomials=[m.clone() for m in self.monomials],
            default_weights=self.default_weights.copy(),
        )


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

@dataclass
class _Fitness:
    """
    Multi-objective fitness: per-class recall + model size.
    Objective[c] = Recall_c = TP_c / max(N_c, 1)
    Size = number of distinct literals (minimize)
    """
    objectives: np.ndarray  # shape (n_classes,), recall per class
    size: int

    def dominates(self, other: "_Fitness") -> bool:
        """True iff self dominates other (at least as good, better in >=1 objective)."""
        if self.size > other.size:
            # size must be <= for dominance
            return False
        # per-class recall must be >=
        if np.any(self.objectives < other.objectives - 1e-12):
            return False
        # at least one component must be strictly better
        return (self.size < other.size) or np.any(self.objectives > other.objectives + 1e-12)

    @property
    def consolidated(self) -> float:
        """Consolidated fitness metric for final ranking."""
        return float(np.mean(self.objectives))


@dataclass
class _FitnessRLCW:
    """
    3-objective fitness for the RLCW trainer (Restricted Literals Computed Weights).

    Objective 1 (max_recall):   max per-class recall
    Objective 2 (mean_other):   mean of remaining per-class recalls (excluding argmax class)
    Objective 3 (size):         model size (minimize)

        Domination: A dominates B only if
            - A.best_class == B.best_class  (class-bound dominance)
      - A.size <= B.size
      - A.max_recall >= B.max_recall
      - A.mean_other >= B.mean_other
    - at least one component is strictly better
    """
    max_recall: float    # obj1: max(per_class_recalls)
    mean_other: float    # obj2: mean of remaining per-class recalls
    size: int            # obj3: model size (minimize)
    best_class: int      # argmax of per-class recalls (for class-bound comparison)

    def dominates(self, other: "_FitnessRLCW") -> bool:
        """True iff self dominates other (class-bound Pareto dominance)."""
        if self.best_class != other.best_class:
            return False
        if self.size > other.size:
            return False
        if self.max_recall < other.max_recall - 1e-12:
            return False
        if self.mean_other < other.mean_other - 1e-12:
            return False
        return (
            self.size < other.size
            or self.max_recall > other.max_recall + 1e-12
            or self.mean_other > other.mean_other + 1e-12
        )

    @property
    def consolidated(self) -> float:
        """Consolidated fitness metric for final ranking."""
        return (self.max_recall + self.mean_other) / 2.0


# ---------------------------------------------------------------------------
# Weight computation (FLCW)
# ---------------------------------------------------------------------------

def _compute_weights(
    poly: _Polynomial,
    X_disc: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
) -> None:
    """
    Compute weights in place for the FLCW variant (class-balanced).

    w_i[c] = (count_c_in_firing / N_c)  normalisiert auf Summe 1.
    This corresponds to normalized per-class recall of the monomial and
    prevents majority classes from dominating the weights.
    w_0 is analogous for instances where NO monomial fires.
    """
    n = X_disc.shape[0]
    any_fired = np.zeros(n, dtype=bool)
    # Class counts for balancing.
    class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)  # Division durch 0 vermeiden

    for mon in poly.monomials:
        mask = mon.fires_mask(X_disc)
        any_fired |= mask
        counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
        total = counts.sum()
        if total > 0:
            # Class-balanced: share of covered instances per class.
            balanced = counts / class_counts
            bal_total = balanced.sum()
            mon.weights = balanced / bal_total if bal_total > 0 else np.ones(n_classes) / n_classes
        else:
            mon.weights = np.ones(n_classes, dtype=float) / n_classes

    # Default weights: where no monomial fired.
    no_fire_mask = ~any_fired
    if no_fire_mask.any():
        counts0 = np.bincount(y_idx[no_fire_mask], minlength=n_classes).astype(float)
        balanced0 = counts0 / class_counts
        bal_total0 = balanced0.sum()
        poly.default_weights = balanced0 / bal_total0 if bal_total0 > 0 else np.ones(n_classes) / n_classes
    else:
        poly.default_weights = np.ones(n_classes, dtype=float) / n_classes


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------

def _evaluate_fitness(
    poly: _Polynomial,
    X_disc: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
) -> _Fitness:
    preds = poly.predict_classes(X_disc)
    recalls = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        mask_c = y_idx == c
        n_c = mask_c.sum()
        if n_c > 0:
            recalls[c] = float((preds[mask_c] == c).sum()) / float(n_c)
    return _Fitness(objectives=recalls, size=poly.size)


def _evaluate_fitness_rlcw(
    poly: _Polynomial,
    X_disc: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
) -> _FitnessRLCW:
    """
    Compute RLCW fitness with 3 objectives:
    1. max per-class recall
    2. mean of remaining per-class recalls
    3. model size (minimize)
    """
    preds = poly.predict_classes(X_disc)
    recalls = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        mask_c = y_idx == c
        n_c = int(mask_c.sum())
        if n_c > 0:
            recalls[c] = float((preds[mask_c] == c).sum()) / float(n_c)
    best_class = int(np.argmax(recalls))
    obj1 = float(recalls[best_class])
    if n_classes > 1:
        other_idx = [i for i in range(n_classes) if i != best_class]
        obj2 = float(np.mean(recalls[other_idx]))
    else:
        obj2 = 0.0
    return _FitnessRLCW(
        max_recall=obj1,
        mean_other=obj2,
        size=poly.size,
        best_class=best_class,
    )


# ---------------------------------------------------------------------------
# Pareto-Selektion
# ---------------------------------------------------------------------------

def _pareto_front(
    individuals: list[tuple[_Polynomial, _Fitness]],
) -> list[tuple[_Polynomial, _Fitness]]:
    """Keep only non-dominated individuals."""
    front: list[tuple[_Polynomial, _Fitness]] = []
    for poly, fit in individuals:
        dominated = False
        for _, other_fit in individuals:
            if other_fit.dominates(fit):
                dominated = True
                break
        if not dominated:
            front.append((poly, fit))
    return front


def _pareto_front_rlcw(
    individuals: list[tuple[_Polynomial, _FitnessRLCW]],
) -> list[tuple[_Polynomial, _FitnessRLCW]]:
    """Keep only non-dominated individuals (class-bound RLCW dominance)."""
    front: list[tuple[_Polynomial, _FitnessRLCW]] = []
    for poly, fit in individuals:
        dominated = False
        for _, other_fit in individuals:
            if other_fit.dominates(fit):
                dominated = True
                break
        if not dominated:
            front.append((poly, fit))
    return front


def _tournament_trim(
    individuals: list,
    n_keep: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> list:
    """
    Trim a population to n_keep individuals using tournament selection.
    Used when the Pareto front exceeds the target population size.
    """
    if len(individuals) <= n_keep:
        return individuals

    selected = []
    # Work with indices to avoid numpy array comparison pitfalls with remove().
    remaining_idx = list(range(len(individuals)))

    for _ in range(n_keep):
        if not remaining_idx:
            break
        k = min(tournament_size, len(remaining_idx))
        contest_positions = rng.choice(len(remaining_idx), size=k, replace=False)
        contest_idx = [remaining_idx[p] for p in contest_positions]

        winner_orig_idx = contest_idx[0]
        _, w_fit = individuals[winner_orig_idx]

        for orig_idx in contest_idx[1:]:
            _, c_fit = individuals[orig_idx]
            if c_fit.dominates(w_fit):
                winner_orig_idx = orig_idx
                w_fit = c_fit
            elif not w_fit.dominates(c_fit) and c_fit.consolidated > w_fit.consolidated:
                winner_orig_idx = orig_idx
                w_fit = c_fit

        selected.append(individuals[winner_orig_idx])
        remaining_idx.remove(winner_orig_idx)

    return selected


# ---------------------------------------------------------------------------
# Mutation and crossover operators
# ---------------------------------------------------------------------------

def _crossover(
    poly_a: _Polynomial,
    poly_b: _Polynomial,
    rng: np.random.Generator,
) -> _Polynomial:
    """
    Crossover: take poly_b as base and replace one random monomial
    with one random monomial from poly_a.
    """
    offspring = poly_b.clone()
    if not poly_a.monomials or not offspring.monomials:
        return offspring
    src_mon = rng.choice(poly_a.monomials)  # type: ignore[arg-type]
    tgt_idx = int(rng.integers(0, len(offspring.monomials)))
    offspring.monomials[tgt_idx] = src_mon.clone()
    return offspring


def _mut_insert_literal(
    poly: _Polynomial,
    all_literals: list[_SetLiteral],
    rng: np.random.Generator,
) -> _Polynomial:
    """Insert a random literal into a random monomial."""
    if not all_literals:
        return poly
    result = poly.clone()
    mon_idx = int(rng.integers(0, len(result.monomials)))
    new_lit = all_literals[int(rng.integers(0, len(all_literals)))]
    result.monomials[mon_idx].literals.append(new_lit)
    return result


def _mut_delete_literal(
    poly: _Polynomial,
    rng: np.random.Generator,
) -> _Polynomial:
    """Delete a random literal from a random monomial."""
    result = poly.clone()
    # Only monomials with more than one literal are eligible.
    candidates = [i for i, m in enumerate(result.monomials) if len(m.literals) > 1]
    if not candidates:
        return result
    mon_idx = candidates[int(rng.integers(0, len(candidates)))]
    lit_idx = int(rng.integers(0, len(result.monomials[mon_idx].literals)))
    result.monomials[mon_idx].literals.pop(lit_idx)
    return result


def _mut_replace_literal(
    poly: _Polynomial,
    all_literals: list[_SetLiteral],
    rng: np.random.Generator,
) -> _Polynomial:
    """Replace a random literal with a new random literal."""
    if not all_literals:
        return poly
    result = poly.clone()
    mon_idx = int(rng.integers(0, len(result.monomials)))
    if not result.monomials[mon_idx].literals:
        return result
    lit_idx = int(rng.integers(0, len(result.monomials[mon_idx].literals)))
    result.monomials[mon_idx].literals[lit_idx] = all_literals[int(rng.integers(0, len(all_literals)))]
    return result


def _mut_insert_monomial(
    poly: _Polynomial,
    all_literals: list[_SetLiteral],
    n_classes: int,
    rng: np.random.Generator,
) -> _Polynomial:
    """Add a new monomial (single literal)."""
    if not all_literals:
        return poly
    result = poly.clone()
    new_lit = all_literals[int(rng.integers(0, len(all_literals)))]
    new_mon = _Monomial(
        literals=[new_lit],
        weights=np.ones(n_classes, dtype=float) / n_classes,
    )
    result.monomials.append(new_mon)
    return result


def _mut_delete_monomial(
    poly: _Polynomial,
    rng: np.random.Generator,
) -> _Polynomial:
    """Delete a random monomial."""
    if len(poly.monomials) <= 1:
        return poly
    result = poly.clone()
    idx = int(rng.integers(0, len(result.monomials)))
    result.monomials.pop(idx)
    return result


# ---------------------------------------------------------------------------
# Literal generation (search space)
# ---------------------------------------------------------------------------

def _generate_literals(X_disc: np.ndarray) -> list[_SetLiteral]:
    """
    Generate all non-trivial subset literals (SetLiterals) for each
    feature in X_disc. This matches the FLCW search space (full literals).
    """
    n_features = X_disc.shape[1]
    literals: list[_SetLiteral] = []
    for feat_idx in range(n_features):
        col = X_disc[:, feat_idx]
        cats = sorted(set(col.tolist()))
        k = len(cats)
        if k < 2:
            continue  # No literal possible for constant features.
        all_cats = tuple(cats)
        power_set_count = 1 << k
        for bitmask in range(1, power_set_count - 1):  # all non-trivial subsets
            cat_set = frozenset(cats[i] for i in range(k) if (bitmask >> i) & 1)
            literals.append(_SetLiteral(
                feature_idx=feat_idx,
                category_set=cat_set,
                all_categories=all_cats,
            ))
    return literals


def _generate_singleton_literals(X_disc: np.ndarray) -> list[_SetLiteral]:
    """
    Generates only singleton literals (one category per literal) for each
    feature. This is a more restricted search space than the full power-set
    approach.  Useful for datasets with many categories per feature.
    """
    n_features = X_disc.shape[1]
    literals: list[_SetLiteral] = []
    for feat_idx in range(n_features):
        col = X_disc[:, feat_idx]
        cats = sorted(set(col.tolist()))
        if len(cats) < 2:
            continue
        all_cats = tuple(cats)
        for cat in cats:
            literals.append(_SetLiteral(
                feature_idx=feat_idx,
                category_set=frozenset([cat]),
                all_categories=all_cats,
            ))
    return literals


# ---------------------------------------------------------------------------
# Configurable strategy types (Protocol-based for extensibility)
# ---------------------------------------------------------------------------

# Literal generator: X_disc -> list of _SetLiteral
LiteralGenerator = Callable[[np.ndarray], list[_SetLiteral]]

# Fitness evaluator: (poly, X_disc, y_idx, n_classes) -> _Fitness | _FitnessRLCW
FitnessEvaluator = Callable[[_Polynomial, np.ndarray, np.ndarray, int], Any]

# Model selector: (candidates, min_improvement) -> _Polynomial
#   candidates: list of (poly, fitness, f1_score)
ModelSelector = Callable[[list, float], _Polynomial]


# ---------------------------------------------------------------------------
# Alternative model selection: best F1 (ignoring model size trade-off)
# ---------------------------------------------------------------------------

def _select_model_best_f1(
    candidates: list[tuple[_Polynomial, Any, float]],
    min_improvement: float = 0.01,
) -> _Polynomial:
    """Select the candidate with the highest F1 score (ties broken by smallest size)."""
    if not candidates:
        raise ValueError("No candidates available.")
    best = max(candidates, key=lambda x: (x[2], -x[1].size))
    return best[0]


# ---------------------------------------------------------------------------
# Final model selection (from paper: Algorithm 1)
# ---------------------------------------------------------------------------

def _final_model_selection(
    candidates: list[tuple[_Polynomial, _Fitness, float]],  # (poly, fit, val_accuracy)
    min_improvement: float = 0.01,
) -> _Polynomial:
    """
    Implements the model selection strategy from the logicGP paper:
    1. Keep best model per size (consolidation)
    2. Discard models with less than 1% improvement vs smaller ones
    3. Choose the largest model with maximal accuracy
    """
    if not candidates:
        raise ValueError("No candidates available.")

    # Step 1: best model per size
    best_per_size: dict[int, tuple[_Polynomial, _Fitness, float]] = {}
    for poly, fit, acc in candidates:
        s = fit.size
        if s not in best_per_size or acc > best_per_size[s][2]:
            best_per_size[s] = (poly, fit, acc)

    sorted_by_size = sorted(best_per_size.values(), key=lambda x: x[1].size)

    # Step 2: discard models without sufficient improvement vs smaller ones
    filtered: list[tuple[_Polynomial, _Fitness, float]] = []
    for poly, fit, acc in sorted_by_size:
        smaller_max_acc = max(
            (a for p, f, a in filtered if f.size < fit.size),
            default=0.0,
        )
        if acc >= smaller_max_acc * (1.0 + min_improvement) or not filtered:
            filtered.append((poly, fit, acc))

    if not filtered:
        filtered = sorted_by_size

    # Step 3: pick model with maximal accuracy (ties: largest model)
    best = max(filtered, key=lambda x: (x[2], x[1].size))
    return best[0]


# ---------------------------------------------------------------------------
# Discretization of continuous features
# ---------------------------------------------------------------------------

def _discretize_features(
    X: np.ndarray,
    n_bins: int = 5,
    fitted_binners: list | None = None,
    cat_masks: np.ndarray | None = None,
    strategy: LogicGPEncodingStrategy = "auto_low_cardinality",
) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Discretize continuous features into bin indices.
    Categorical features (non-numeric or <= n_bins unique values) stay unchanged.
    Returns (X_disc, fitted_binners, cat_mask).
    cat_mask[i] = True means feature i was treated as categorical (no binner).
    """
    return discretize_logicgp_features(
        X,
        n_bins=n_bins,
        fitted_binners=fitted_binners,
        cat_masks=cat_masks,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Main class: LogicGPClassifier
# ---------------------------------------------------------------------------

class LogicGPClassifier(BaseRuleSetEstimator):
    """
    logicGP Python implementation (FLCW and RLCW variants).

    Supports two trainer families:

    **FLCW** (Full Literals Computed Weights, original variant):
      - ``trainer="flcw"``
    - all non-trivial subset literals as search space
    - n_classes Pareto objectives (per-class recall + size)

    **RLCW** (Restricted Literals Computed Weights, more efficient variant):
      - ``trainer="rlcw"`` (Standard)
    - restricted search space via ``min_max_weight``
    - 3 Pareto objectives (max-recall, mean-other-recall, size)
    - class-bound Pareto dominance (encourages class diversity)
    - optional population-size limit with tournament selection

    Parameters
    ----------
    trainer : str
        Trainer variant: ``"rlcw"`` (default) or ``"flcw"``.
    f1_averaging : str
        F1 averaging for model selection: ``"micro"`` (default)
        or ``"macro"``.
    max_generations : int
        Maximum number of GP generations.
    stagnation_generations : int
        Stop after this many generations without improvement.
    n_bins : int
        Number of bins for discretizing continuous features.
    min_max_weight : float
        Filter literals whose maximal class weight is <= this threshold.
        0.0 = no filter. Recommended for RLCW: 0.1-0.3.
    min_improvement_pct : float
        Minimum improvement percentage for model selection (default: 0.01 = 1%).
    population_size : int or None
        Maximum population size after selection. If ``None``, keep the
        full Pareto front (default FLCW behavior).
        Recommended for RLCW: 30-100.
    n_adaptations_per_gen : int
        Number of new individuals per generation. Default is 6
        (1 crossover + 5 mutations). For RLCW, a higher value
        (e.g. 12-24) can improve convergence.
    tournament_size : int
        Tournament size for tournament selection (only active when
        ``population_size`` gesetzt und Pareto-Front > ``population_size``).
    max_model_size : int or None
        Maximum model size (number of literals). If set, individuals
        with larger model size are ignored (search phase of the
        two-phase model selection algorithm from logicGP-RLCW).
    validation_fraction : float
        Fraction of training data reserved as validation set for final
        model selection. Selection then uses actual F1 on the validation set
        (according to ``f1_averaging``).
        ``0`` (default) disables the split: selection is then based on
        training F1, which is often more reliable than ``consolidated``
        (mean recall) and avoids data loss.
        Values > 0 are only recommended for large datasets (n >= 200).
    max_fit_seconds : float or None
        Maximum runtime for the GP loop in seconds. If set, evolution is
        stopped cleanly after timeout and the best model so far is returned.
        ``None`` (default) disables the time limit. In benchmarks, use a value
        below the outer timeout (e.g. 240 with a 300s timeout).
    random_state : int or None
    """

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
        self.trainer = trainer
        self.f1_averaging = f1_averaging
        self.max_generations = max_generations
        self.stagnation_generations = stagnation_generations
        self.n_bins = n_bins
        self.min_max_weight = min_max_weight
        self.min_improvement_pct = min_improvement_pct
        self.population_size = population_size
        self.n_adaptations_per_gen = n_adaptations_per_gen
        self.tournament_size = tournament_size
        self.max_model_size = max_model_size
        self.validation_fraction = validation_fraction
        self.literal_generator = literal_generator
        self.model_selection = model_selection
        self.fitness_evaluator = fitness_evaluator
        self.max_fit_seconds = max_fit_seconds
        self.feature_encoding_strategy = feature_encoding_strategy
        self.random_state = random_state


    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = np.asarray(
            [f"f{i}" for i in range(self.n_features_in_)], dtype=object
        )
        self.classes_ = unique_labels(y_valid)
        n_classes = len(self.classes_)

        self._rng_ = np.random.default_rng(self.random_state)

        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y_valid], dtype=int)

        # Discretization
        X_disc, self._binners_, self._cat_masks_ = _discretize_features(
            X_valid,
            n_bins=self.n_bins,
            strategy=self.feature_encoding_strategy,
        )

        # ----- Validation split for improved final model selection -----
        use_val = (
            self.validation_fraction > 0
            and X_disc.shape[0] >= 30          # too few samples -> no split
            and len(np.unique(y_idx)) >= 2     # at least 2 classes
        )
        if use_val:
            try:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=self.validation_fraction,
                    random_state=(self.random_state if self.random_state is not None else 0),
                )
                train_idx, val_idx = next(sss.split(X_disc, y_idx))
                X_disc_train = X_disc[train_idx]
                y_idx_train = y_idx[train_idx]
                X_disc_val = X_disc[val_idx]
                y_idx_val = y_idx[val_idx]
            except ValueError:
                # Stratification failed (too few samples per class).
                use_val = False

        if not use_val:
            X_disc_train = X_disc
            y_idx_train = y_idx
            X_disc_val = None
            y_idx_val = None

        # Suchraum: Literale via konfigurierbaren Generator
        lit_gen_fn = self._resolve_literal_generator()
        all_literals = lit_gen_fn(X_disc_train)
        if self.min_max_weight > 0 and len(all_literals) > 0:
            all_literals = self._filter_literals(
                all_literals, X_disc_train, y_idx_train, n_classes
            )
        if not all_literals:
            raise ValueError(
                "No literals can be generated. Please check features or n_bins."
            )

        # Initial population: one individual per literal + seeds.
        population = self._init_population(
            all_literals, n_classes, X_disc_train, y_idx_train
        )

        # GP loop (on training split)
        best_poly, final_population = self._run_gp(
            population, all_literals, X_disc_train, y_idx_train, n_classes,
            X_val=X_disc_val, y_val=y_idx_val,
        )
        self._final_population = final_population

        # Recompute weights on the full dataset.
        if use_val:
            _compute_weights(best_poly, X_disc, y_idx, n_classes)

        # Create scored ruleset.
        self.ruleset_ = self._poly_to_ruleset(best_poly, n_classes)
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
        """Extract pairwise feature interactions from the final GP population.

        Delegates to :func:`scoredrulesets.analysis.extract_interactions`.
        Must be called after :meth:`fit`.

        Parameters
        ----------
        min_occurrences:
            Minimum co-occurrence count to keep an interaction edge.
        min_ratio:
            Minimum ratio of pair count to individual feature count.
        out_dot:
            Optional path for a GraphViz DOT output file.
        out_csv:
            Optional path for a CSV output file.

        Returns
        -------
        dict with keys ``edges``, ``feature_counts``, ``pair_counts``.
        """
        from scoredrulesets.analysis.logicgp_interactions import extract_interactions
        return extract_interactions(
            self,
            min_occurrences=min_occurrences,
            min_ratio=min_ratio,
            out_dot=out_dot,
            out_csv=out_csv,
        )

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _resolve_literal_generator(self) -> LiteralGenerator:
        """Resolve the literal_generator parameter to a callable."""
        gen = self.literal_generator
        if callable(gen) and not isinstance(gen, str):
            return gen
        if isinstance(gen, str):
            _GENERATORS: dict[str, LiteralGenerator] = {
                "full": _generate_literals,
                "singleton": _generate_singleton_literals,
            }
            if gen in _GENERATORS:
                return _GENERATORS[gen]
            raise ValueError(
                f"Unknown literal_generator '{gen}'. "
                f"Choose from {list(_GENERATORS.keys())} or pass a callable."
            )
        raise ValueError(
            f"literal_generator must be a string or callable, got {type(gen)}."
        )

    def _resolve_model_selector(self) -> ModelSelector:
        """Resolve the model_selection parameter to a callable."""
        sel = self.model_selection
        if callable(sel) and not isinstance(sel, str):
            return sel
        if isinstance(sel, str):
            _SELECTORS: dict[str, ModelSelector] = {
                "paper": _final_model_selection,
                "best_f1": _select_model_best_f1,
            }
            if sel in _SELECTORS:
                return _SELECTORS[sel]
            raise ValueError(
                f"Unknown model_selection '{sel}'. "
                f"Choose from {list(_SELECTORS.keys())} or pass a callable."
            )

    def _resolve_trainer_config(self) -> tuple[bool, str]:
        """Resolve trainer + f1_averaging into (use_rlcw, f1_average).

        Returns
        -------
        use_rlcw : bool
            True for RLCW trainer, False for FLCW.
        f1_average : str
            ``"micro"`` or ``"macro"``.
        """
        trainer = self.trainer.lower().strip()
        f1_avg = self.f1_averaging.lower().strip()

        if trainer not in ("rlcw", "flcw"):
            raise ValueError(
                f"Unknown trainer '{self.trainer}'. "
                f"Choose 'rlcw' or 'flcw'."
            )
        if f1_avg not in ("micro", "macro"):
            raise ValueError(
                f"Unknown f1_averaging '{self.f1_averaging}'. "
                f"Choose 'micro' or 'macro'."
            )

        use_rlcw = trainer == "rlcw"
        return use_rlcw, f1_avg

    def _resolve_fitness_evaluator(self) -> FitnessEvaluator | None:
        """Resolve the fitness_evaluator parameter to a callable, or None for auto."""
        fe = self.fitness_evaluator
        if fe is None:
            return None
        if callable(fe) and not isinstance(fe, str):
            return fe
        if isinstance(fe, str):
            _EVALUATORS: dict[str, FitnessEvaluator] = {
                "flcw": _evaluate_fitness,
                "rlcw": _evaluate_fitness_rlcw,
            }
            if fe in _EVALUATORS:
                return _EVALUATORS[fe]
            raise ValueError(
                f"Unknown fitness_evaluator '{fe}'. "
                f"Choose from {list(_EVALUATORS.keys())} or pass a callable."
            )
        raise ValueError(
            f"fitness_evaluator must be a string, callable, or None, got {type(fe)}."
        )

    @staticmethod
    def _strategy_name(strategy) -> str:
        """Return a human-readable name for a strategy parameter."""
        if isinstance(strategy, str):
            return strategy
        if callable(strategy):
            return getattr(strategy, "__name__", repr(strategy))
        return repr(strategy)

    def _filter_literals(
        self,
        literals: list[_SetLiteral],
        X_disc: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
    ) -> list[_SetLiteral]:
        """Filter out literals whose maximal class weight is too small."""
        filtered = []
        for lit in literals:
            col = X_disc[:, lit.feature_idx]
            cats = np.array(sorted(lit.category_set))
            mask = np.isin(col, cats)
            if mask.sum() == 0:
                continue
            counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
            total = counts.sum()
            if total > 0:
                weights = counts / total
                if weights.max() > self.min_max_weight:
                    filtered.append(lit)
        return filtered if filtered else literals

    def _init_population(
        self,
        all_literals: list[_SetLiteral],
        n_classes: int,
        X_disc: np.ndarray | None = None,
        y_idx: np.ndarray | None = None,
    ) -> list[_Polynomial]:
        """Create initial population: one individual per literal + seeds.

        In addition to standard single-literal individuals, this creates
        class-discriminative seed individuals:
        1. For each class: best 2-literal conjunction (different features)
        2. Multi-monomial individuals: one monomial per class (multi-class)
        This can accelerate convergence, especially for multi-class problems.
        """
        population = []
        # Standard: one individual per literal.
        for lit in all_literals:
            mon = _Monomial(
                literals=[lit],
                weights=np.ones(n_classes, dtype=float) / n_classes,
            )
            poly = _Polynomial(
                monomials=[mon],
                default_weights=np.ones(n_classes, dtype=float) / n_classes,
            )
            population.append(poly)

        # Class-discriminative seeding (only if X_disc/y_idx are available).
        if X_disc is not None and y_idx is not None and n_classes >= 2:
            population.extend(
                self._seed_class_discriminative(
                    all_literals, n_classes, X_disc, y_idx
                )
            )
        return population

    def _seed_class_discriminative(
        self,
        all_literals: list[_SetLiteral],
        n_classes: int,
        X_disc: np.ndarray,
        y_idx: np.ndarray,
    ) -> list[_Polynomial]:
        """Create class-discriminative seed individuals.

        For each class, the literal with the best combination of
        purity (class fraction among firing samples) and coverage
        (fraction of correctly covered class instances) is determined.
        This is used to build multi-literal monomials and multi-monomial polynomials.
        """
        seeds: list[_Polynomial] = []
        n_samples = X_disc.shape[0]
        unif = np.ones(n_classes, dtype=float) / n_classes

        # Evaluate discriminative strength of each literal per class.
        lit_fire_masks: list[np.ndarray] = []
        for lit in all_literals:
            col = X_disc[:, lit.feature_idx]
            cats = np.array(sorted(lit.category_set))
            mask = np.isin(col, cats)
            lit_fire_masks.append(mask)

        # Find best literal per class (by purity * coverage).
        best_per_class: list[list[tuple[int, float]]] = [[] for _ in range(n_classes)]
        for lit_idx, mask in enumerate(lit_fire_masks):
            n_fire = int(mask.sum())
            if n_fire == 0:
                continue
            for c in range(n_classes):
                class_mask = y_idx == c
                n_class = int(class_mask.sum())
                if n_class == 0:
                    continue
                tp = int((mask & class_mask).sum())
                purity = tp / n_fire
                coverage = tp / n_class
                score = purity * coverage
                best_per_class[c].append((lit_idx, score))

        # Sort and keep top-5 per class.
        top_k = 5
        class_top_lits: list[list[int]] = []
        for c in range(n_classes):
            best_per_class[c].sort(key=lambda x: x[1], reverse=True)
            class_top_lits.append([idx for idx, _ in best_per_class[c][:top_k]])

        # Seed 1: for each class, one 2-literal monomial (different features).
        for c in range(n_classes):
            if len(class_top_lits[c]) < 2:
                continue
            for i in range(min(3, len(class_top_lits[c]))):
                for j in range(i + 1, min(4, len(class_top_lits[c]))):
                    lit_i = all_literals[class_top_lits[c][i]]
                    lit_j = all_literals[class_top_lits[c][j]]
                    if lit_i.feature_idx != lit_j.feature_idx:
                        mon = _Monomial(
                            literals=[lit_i, lit_j],
                            weights=unif.copy(),
                        )
                        poly = _Polynomial(
                            monomials=[mon],
                            default_weights=unif.copy(),
                        )
                        seeds.append(poly)
                        break  # one seed per class is enough

        # Seed 2: multi-monomial polynomial - one monomial per class.
        if n_classes >= 2 and all(len(cl) > 0 for cl in class_top_lits):
            monomials = []
            for c in range(n_classes):
                best_lit = all_literals[class_top_lits[c][0]]
                monomials.append(_Monomial(
                    literals=[best_lit],
                    weights=unif.copy(),
                ))
            poly = _Polynomial(
                monomials=monomials,
                default_weights=unif.copy(),
            )
            seeds.append(poly)

            # Auch eine Variante mit den zweitbesten Literalen
            if all(len(cl) >= 2 for cl in class_top_lits):
                monomials2 = []
                for c in range(n_classes):
                    lit2 = all_literals[class_top_lits[c][1]]
                    monomials2.append(_Monomial(
                        literals=[lit2],
                        weights=unif.copy(),
                    ))
                poly2 = _Polynomial(
                    monomials=monomials2,
                    default_weights=unif.copy(),
                )
                seeds.append(poly2)

        return seeds

    def _run_gp(
        self,
        population: list[_Polynomial],
        all_literals: list[_SetLiteral],
        X_disc: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[_Polynomial, list[tuple[_Polynomial, _Fitness | _FitnessRLCW]]]:
        """
        Run the main GP loop and return the best polynomial.
        Supports FLCW and RLCW variants based on ``self.trainer``.

        Final model selection evaluates candidates based on
        actual F1 (instead of ``fit.consolidated`` = mean recall).
        ``f1_averaging`` determines micro or macro averaging.
        If ``X_val``/``y_val`` are provided: F1 on validation set.
        Otherwise: F1 on training data (no data loss, more reliable
        for small datasets).
        """
        use_rlcw, f1_average = self._resolve_trainer_config()
        has_val = X_val is not None and y_val is not None

        # Resolve configurable strategies.
        resolved_fe = self._resolve_fitness_evaluator()
        if resolved_fe is not None:
            evaluate_fn = resolved_fe
            use_rlcw = (evaluate_fn is _evaluate_fitness_rlcw)
        else:
            evaluate_fn = _evaluate_fitness_rlcw if use_rlcw else _evaluate_fitness
        pareto_fn = _pareto_front_rlcw if use_rlcw else _pareto_front
        select_model_fn = self._resolve_model_selector()

        # ------------------------------------------------------------------
        # Compute weights and evaluate initial population.
        # ------------------------------------------------------------------
        for poly in population:
            _compute_weights(poly, X_disc, y_idx, n_classes)

        evaluated: list = [
            (poly, evaluate_fn(poly, X_disc, y_idx, n_classes))
            for poly in population
        ]
        evaluated = pareto_fn(evaluated)

        # Limit population size (RLCW with population_size).
        if self.population_size is not None and len(evaluated) > self.population_size:
            evaluated = _tournament_trim(
                evaluated, self.population_size, self.tournament_size, self._rng_
            )

        best_consolidated = max(f.consolidated for _, f in evaluated)
        stagnation_count = 0
        all_candidates: list = [
            (poly, fit, fit.consolidated) for poly, fit in evaluated
        ]

        # ------------------------------------------------------------------
        # Elitism: track the best individual by macro-F1 to prevent
        # regression and speed up convergence, especially in multi-class tasks.
        # ------------------------------------------------------------------
        labels = list(range(n_classes))
        eval_X_elite = X_val if has_val else X_disc
        eval_y_elite = y_val if has_val else y_idx

        elite_poly: _Polynomial | None = None
        elite_fit = None
        elite_f1 = -1.0

        for poly, fit in evaluated:
            preds = poly.predict_classes(eval_X_elite)
            f1 = float(_f1_score(eval_y_elite, preds, average=f1_average, labels=labels))
            if f1 > elite_f1:
                elite_f1 = f1
                elite_poly = poly.clone()
                elite_fit = fit

        # Mutation operators as a list for cyclic traversal.
        _MUT_OPS = [
            lambda p: _mut_insert_literal(p, all_literals, self._rng_),
            lambda p: _mut_delete_literal(p, self._rng_),
            lambda p: _mut_replace_literal(p, all_literals, self._rng_),
            lambda p: _mut_insert_monomial(p, all_literals, n_classes, self._rng_),
            lambda p: _mut_delete_monomial(p, self._rng_),
        ]
        n_adapt = max(1, self.n_adaptations_per_gen)

        # Time budget for early stopping (max_fit_seconds).
        _gp_start_time = time.monotonic()
        _time_budget = self.max_fit_seconds  # None = no limit

        for _gen in range(self.max_generations):
            # Check time budget.
            if _time_budget is not None:
                elapsed = time.monotonic() - _gp_start_time
                if elapsed >= _time_budget:
                    break

            # ------------------------------------------------------------------
            # Generate new individuals (n_adaptations_per_gen items).
            # ------------------------------------------------------------------
            new_polys: list[_Polynomial] = []

            for i in range(n_adapt):
                if i == 0 and len(evaluated) >= 2:
                    # Crossover: first child is crossover offspring.
                    p_a, p_b = self._select_two_parents(evaluated)
                    new_polys.append(_crossover(p_a, p_b, self._rng_))
                else:
                    # Mutation: cycle through all 5 operator types.
                    parent = self._select_parent(evaluated)
                    new_polys.append(_MUT_OPS[(i - 1) % 5](parent))

            # ------------------------------------------------------------------
            # Fitness for new individuals.
            # ------------------------------------------------------------------
            new_evaluated: list = []
            for poly in new_polys:
                if not poly.monomials:
                    continue
                # Size constraint (RLCW search phase).
                if self.max_model_size is not None and poly.size > self.max_model_size:
                    continue
                _compute_weights(poly, X_disc, y_idx, n_classes)
                fit = evaluate_fn(poly, X_disc, y_idx, n_classes)
                new_evaluated.append((poly, fit))

            # ------------------------------------------------------------------
            # Pareto selection + optional population-size cap.
            # ------------------------------------------------------------------
            combined = evaluated + new_evaluated
            evaluated = pareto_fn(combined)

            if self.population_size is not None and len(evaluated) > self.population_size:
                evaluated = _tournament_trim(
                    evaluated, self.population_size, self.tournament_size, self._rng_
                )

            # ------------------------------------------------------------------
            # Elitism: ensure elite individual stays in the population.
            # ------------------------------------------------------------------
            if elite_poly is not None and elite_fit is not None:
                # Check whether elite is still in the population.
                elite_ids = {id(p) for p, _ in evaluated}
                if id(elite_poly) not in elite_ids:
                    evaluated.append((elite_poly, elite_fit))

            # Elite update: check whether a new best individual exists.
            for poly, fit in new_evaluated:
                preds = poly.predict_classes(eval_X_elite)
                f1 = float(_f1_score(eval_y_elite, preds, average=f1_average, labels=labels))
                if f1 > elite_f1:
                    elite_f1 = f1
                    elite_poly = poly.clone()
                    elite_fit = fit

            # ------------------------------------------------------------------
            # Stagnation tracking (always on consolidated, for efficiency).
            # ------------------------------------------------------------------
            current_best = max(f.consolidated for _, f in evaluated)
            if current_best > best_consolidated + 1e-10:
                best_consolidated = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1

            all_candidates.extend(
                (poly, fit, fit.consolidated) for poly, fit in evaluated
            )

            if stagnation_count >= self.stagnation_generations:
                break

        # ------------------------------------------------------------------
        # Final model selection
        # ------------------------------------------------------------------
        # Evaluate candidates using actual macro-F1 instead of
        # ``consolidated`` (mean recall), da F1 besser mit der realen
        # predictive quality correlates better.
        # With active validation split: F1 on validation set (generalization).
        # Without split: F1 on training data (no data loss).
        eval_X = X_val if has_val else X_disc
        eval_y = y_val if has_val else y_idx
        labels = list(range(n_classes))

        seen_ids: set[int] = set()
        f1_candidates: list[tuple[_Polynomial, _Fitness | _FitnessRLCW, float]] = []

        # 1) Finale Population (aktuellste und relevanteste Individuen)
        for poly, fit in evaluated:
            if id(poly) not in seen_ids:
                seen_ids.add(id(poly))
                preds = poly.predict_classes(eval_X)
                f1 = float(_f1_score(eval_y, preds, average=f1_average, labels=labels))
                f1_candidates.append((poly, fit, f1))

        # 2) Historische Top-Kandidaten (nach consolidated) re-evaluieren
        top_hist = sorted(all_candidates, key=lambda x: x[2], reverse=True)
        n_extra = 0
        for poly, fit, _ in top_hist:
            if id(poly) in seen_ids:
                continue
            seen_ids.add(id(poly))
            preds = poly.predict_classes(eval_X)
            f1 = float(_f1_score(eval_y, preds, average=f1_average, labels=labels))
            f1_candidates.append((poly, fit, f1))
            n_extra += 1
            if n_extra >= 50:
                break

        return select_model_fn(f1_candidates, self.min_improvement_pct), evaluated

    def _select_two_parents(self, evaluated: list) -> tuple[_Polynomial, _Polynomial]:
        """Select two parents via dominance tournament."""
        idx_a, idx_b = self._rng_.choice(len(evaluated), size=2, replace=False)
        poly_a, fit_a = evaluated[idx_a]
        poly_b, fit_b = evaluated[idx_b]
        if fit_a.dominates(fit_b):
            return poly_a, poly_b
        if fit_b.dominates(fit_a):
            return poly_b, poly_a
        return poly_a, poly_b

    def _select_parent(self, evaluated: list) -> _Polynomial:
        """Random parent selection from the current population."""
        idx = int(self._rng_.integers(0, len(evaluated)))
        return evaluated[idx][0]

    def _poly_to_ruleset(self, poly: _Polynomial, n_classes: int) -> ScoredRuleSet:
        """Convert a _Polynomial to a ScoredRuleSet (weights already computed)."""
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

        _use_rlcw, _f1_avg = self._resolve_trainer_config()

        return ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=feature_names,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            rules=rules,
            metadata={
                "source": "logicgp",
                "trainer": "rlcw" if _use_rlcw else "flcw",
                "f1_averaging": _f1_avg,
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
        """Convert monomial literals to an atom list for ScoredRuleSet."""
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


# ---------------------------------------------------------------------------
# GPAS classifier (binary specialization of logicGP)
# ---------------------------------------------------------------------------

class GPASClassifier(LogicGPClassifier):
    """
    GPAS -- Genetic Programming for Association Studies.

    Specialization of logicGP for binary classification of categorical
    (e.g. SNP) data, based on:

      Nunkesser et al. (2007): "Detecting high-order interactions of
      single nucleotide polymorphisms using genetic programming"
      Bioinformatics, 23(24):3280–3288.

    Key differences to :class:`LogicGPClassifier`:

    - **Binary classification only** -- ``fit`` raises ``ValueError`` for
      more than 2 classes.
    - **Size-2 initialization** -- initial population consists of monomials
      with exactly 2 literals each, matching the
      ``RandomInitialization { Size = 2 }`` strategy from the C# reference
      implementation.
    - **Fixed trainer**: ``trainer="flcw"`` and ``f1_averaging="micro"``
      (binary micro-accuracy equals overall accuracy).

    Parameters
    ----------
    max_generations : int
        Maximum number of GP generations.
    stagnation_generations : int
        Stop after this many generations without improvement.
    n_bins : int
        Number of bins for discretizing continuous features.
    min_improvement_pct : float
        Minimum improvement percentage for model selection.
    population_size : int or None
        Maximum population size after Pareto selection. ``None`` keeps the
        full Pareto front.
    n_adaptations_per_gen : int
        Number of new individuals per generation (1 crossover + mutations).
    tournament_size : int
        Tournament size for population trimming when ``population_size`` is set.
    max_model_size : int or None
        Maximum number of literals allowed in a model.
    validation_fraction : float
        Fraction of training data reserved for final model selection.
        ``0`` (default) disables the split.
    max_fit_seconds : float or None
        Time budget for the GP loop in seconds. ``None`` = no limit.
    feature_encoding_strategy : str
        Encoding strategy for features; passed to the underlying discretizer.
    random_state : int or None
        Random seed for reproducibility.
    """

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
            model_selection="paper",
            fitness_evaluator=None,
            max_fit_seconds=max_fit_seconds,
            feature_encoding_strategy=feature_encoding_strategy,
            random_state=random_state,
        )

    def fit(self, X, y):
        labels = unique_labels(y)
        if len(labels) != 2:
            raise ValueError(
                f"GPASClassifier requires exactly 2 classes, got {len(labels)}. "
                "Use LogicGPClassifier for multi-class problems."
            )
        return super().fit(X, y)

    def _init_population(
        self,
        all_literals: list[_SetLiteral],
        n_classes: int,
        X_disc: np.ndarray | None = None,
        y_idx: np.ndarray | None = None,
    ) -> list[_Polynomial]:
        """
        GPAS initialization: each individual is a single monomial with
        exactly 2 literals (``RandomInitialization { Size = 2 }``).

        For each literal ``l_i``, one polynomial is created whose monomial
        pairs ``l_i`` with a randomly drawn literal from a *different*
        feature (falling back to any other literal when unavoidable).
        """
        population: list[_Polynomial] = []
        n_lits = len(all_literals)
        if n_lits < 2:
            # Degenerate case: fall back to the parent initializer.
            return super()._init_population(all_literals, n_classes, X_disc, y_idx)

        # Group literal indices by feature index for efficient sampling.
        feature_to_lits: dict[int, list[int]] = {}
        for idx, lit in enumerate(all_literals):
            feature_to_lits.setdefault(lit.feature_idx, []).append(idx)

        unif = np.ones(n_classes, dtype=float) / n_classes

        for i, lit_i in enumerate(all_literals):
            other_features = [f for f in feature_to_lits if f != lit_i.feature_idx]
            if other_features:
                feat_j = int(self._rng_.choice(other_features))
                j = int(self._rng_.choice(feature_to_lits[feat_j]))
            else:
                candidates = [k for k in range(n_lits) if k != i]
                j = int(self._rng_.choice(candidates))

            mon = _Monomial(
                literals=[lit_i, all_literals[j]],
                weights=unif.copy(),
            )
            poly = _Polynomial(
                monomials=[mon],
                default_weights=unif.copy(),
            )
            population.append(poly)

        # Class-discriminative seeding (inherited strategy).
        if X_disc is not None and y_idx is not None:
            population.extend(
                self._seed_class_discriminative(
                    all_literals, n_classes, X_disc, y_idx
                )
            )

        return population











