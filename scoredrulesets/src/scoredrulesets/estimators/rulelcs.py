"""ruleLCS – BioHEL-inspired iterative rule learning classifier.

A simplified reimplementation of the BioHEL algorithm (Bacardit & Butz, 2009)
as a scikit-learn compatible estimator.

Key elements from BioHEL:
* **Iterative Rule Learning (IRL)** – sequential covering with GA-based rule
  discovery.  Each GA run produces *one* rule; the rule set is built
  incrementally by removing correctly covered instances after each rule.
* **Hyperrectangle representation with attribute list** – variable-length
  rules that combine interval predicates for numeric features and subset
  predicates for categoricals.  Attributes not expressed are "don't-care".
* **MDL-based fitness** – balances rule precision, recall (coverage), and
  theory length (complexity) with a dynamically relaxed MDL weight.
* **Tournament selection (WOR)** – selection without replacement.
* **Attribute-level crossover** – variable-length crossover that exchanges
  attribute blocks between parents.
* **Generalize / Specialize operators** – probabilistic structural mutation
  that adds or removes attributes from a rule.
* **Smart initialisation** – rules are seeded around randomly sampled
  training instances with class-balanced sampling.
* **ILAS windowing** – incremental learning with alternating strata for
  efficient fitness evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Predicate:
    """One condition on a single feature."""
    feature_idx: int
    is_numeric: bool
    # numeric: (lo, hi) interval
    lo: float = 0.0
    hi: float = 0.0
    # categorical: set of allowed values
    allowed: set[int] = field(default_factory=set)

    def matches_value(self, val: float) -> bool:
        if self.is_numeric:
            return self.lo <= val <= self.hi
        return int(round(val)) in self.allowed


@dataclass
class _Individual:
    """One classifier (rule) in the GA population."""
    predicates: list[_Predicate]
    class_value: int
    # fitness components
    fitness: float = np.inf
    accuracy2: float = 0.0  # precision = TP / (TP + FP)
    recall: float = 0.0
    theory_length: float = 0.0
    exceptions_length: float = 0.0

    def matches(self, x: np.ndarray) -> bool:
        for p in self.predicates:
            if not p.matches_value(x[p.feature_idx]):
                return False
        return True


def _matches_mask(ind: _Individual, X: np.ndarray) -> np.ndarray:
    """Return boolean mask of instances matched by *ind* (vectorised)."""
    mask = np.ones(X.shape[0], dtype=bool)
    for p in ind.predicates:
        col = X[:, p.feature_idx]
        if p.is_numeric:
            mask &= (col >= p.lo) & (col <= p.hi)
        else:
            rounded = np.rint(col).astype(np.intp)
            allowed_arr = np.array(sorted(p.allowed), dtype=np.intp)
            mask &= np.isin(rounded, allowed_arr)
        if not mask.any():
            break
    return mask


# ---------------------------------------------------------------------------
# BioHEL GA core
# ---------------------------------------------------------------------------

def _smart_init(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    feature_info: list[dict],
    target_class: int,
    prob_include: float,
    prob_one: float,
) -> _Individual:
    """Create one individual seeded around a random training instance."""
    # Pick a random instance of the target class
    class_mask = y == target_class
    class_indices = np.where(class_mask)[0]
    if len(class_indices) == 0:
        class_indices = np.arange(len(y))
    idx = rng.choice(class_indices)
    inst = X[idx]

    predicates: list[_Predicate] = []
    for fi in range(n_features):
        if rng.random() > prob_include:
            continue  # don't-care
        info = feature_info[fi]
        if info["numeric"]:
            domain = info["max"] - info["min"]
            if domain <= 0:
                continue
            size = rng.uniform(0.25, 0.75) * domain
            center = inst[fi]
            lo = max(info["min"], center - size / 2)
            hi = min(info["max"], center + size / 2)
            predicates.append(_Predicate(
                feature_idx=fi, is_numeric=True, lo=lo, hi=hi,
            ))
        else:
            vals = info["values"]
            allowed: set[int] = set()
            v = int(round(inst[fi]))
            allowed.add(v)
            for ov in vals:
                if ov != v and rng.random() < prob_one:
                    allowed.add(ov)
            if len(allowed) < len(vals):
                predicates.append(_Predicate(
                    feature_idx=fi, is_numeric=False, allowed=allowed,
                ))
            # if all values allowed → don't-care

    return _Individual(predicates=predicates, class_value=target_class)


def _compute_theory_length(ind: _Individual, feature_info: list[dict]) -> float:
    """Compute normalised theory length ∈ [0, 1].

    For each expressed attribute:
      numeric: 1 - (hi - lo) / domain_size
      categorical: num_excluded / num_values
    Average over *all* features (unexpressed = 0 contribution).
    """
    if len(feature_info) == 0:
        return 0.0
    total = 0.0
    for p in ind.predicates:
        info = feature_info[p.feature_idx]
        if p.is_numeric:
            domain = info["max"] - info["min"]
            if domain > 0:
                total += 1.0 - (p.hi - p.lo) / domain
            # else: single-value domain → 0
        else:
            n_vals = len(info["values"])
            if n_vals > 0:
                total += (n_vals - len(p.allowed)) / n_vals
    return total / len(feature_info)


def _evaluate(
    ind: _Individual,
    X: np.ndarray,
    y: np.ndarray,
    feature_info: list[dict],
    coverage_break: float,
    coverage_ratio: float,
    mdl_weight: float,
    use_mdl: bool,
) -> None:
    """Compute MDL-based fitness for one individual."""
    total_positive = int(np.sum(y == ind.class_value))
    match_mask = _matches_mask(ind, X)
    matched_y = y[match_mask]
    tp = int(np.sum(matched_y == ind.class_value))
    fp = int(match_mask.sum()) - tp
    matched = tp + fp
    # Precision
    if matched > 0:
        ind.accuracy2 = tp / matched
    else:
        ind.accuracy2 = 0.0

    # Recall
    if total_positive > 0:
        ind.recall = tp / total_positive
    else:
        ind.recall = 0.0

    # Coverage penalty
    min_recall = coverage_break / 3.0
    if ind.recall < min_recall:
        coverage = 0.0
    elif ind.recall < coverage_break:
        coverage = coverage_ratio * ind.recall / coverage_break
    else:
        coverage = coverage_ratio + (1.0 - coverage_ratio) * min(
            1.0, (ind.recall - coverage_break) / max(1e-12, 1.0 - coverage_break)
        )

    acc_error = 1.0 - ind.accuracy2
    coverage_penalty = 1.0 - coverage
    ind.exceptions_length = acc_error + coverage_penalty

    # Theory length
    ind.theory_length = _compute_theory_length(ind, feature_info)

    # MDL fitness (minimise)
    if use_mdl:
        ind.fitness = ind.theory_length * mdl_weight + ind.exceptions_length
    else:
        ind.fitness = ind.exceptions_length


def _crossover_hyperrect_list(
    rng: np.random.Generator,
    p1: _Individual,
    p2: _Individual,
    feature_info: list[dict],
) -> tuple[_Individual, _Individual]:
    """Attribute-level crossover for variable-length rules."""
    if len(p1.predicates) == 0 or len(p2.predicates) == 0:
        # Can't crossover empty rules — just swap classes
        c1_preds = [_clone_pred(p) for p in p1.predicates]
        c2_preds = [_clone_pred(p) for p in p2.predicates]
        cls1 = p2.class_value if rng.random() < 0.5 else p1.class_value
        cls2 = p1.class_value if rng.random() < 0.5 else p2.class_value
        return (
            _Individual(predicates=c1_preds, class_value=cls1),
            _Individual(predicates=c2_preds, class_value=cls2),
        )

    pos1 = rng.integers(0, len(p1.predicates))
    att1 = p1.predicates[pos1].feature_idx

    # Find position in p2 with closest attribute index
    pos2 = 0
    best_dist = abs(p2.predicates[0].feature_idx - att1)
    for j in range(1, len(p2.predicates)):
        d = abs(p2.predicates[j].feature_idx - att1)
        if d < best_dist:
            best_dist = d
            pos2 = j

    c1_preds = [_clone_pred(p) for p in p1.predicates[:pos1]] + \
               [_clone_pred(p) for p in p2.predicates[pos2:]]
    c2_preds = [_clone_pred(p) for p in p2.predicates[:pos2]] + \
               [_clone_pred(p) for p in p1.predicates[pos1:]]

    # Deduplicate attributes (keep first occurrence)
    c1_preds = _dedup_predicates(c1_preds)
    c2_preds = _dedup_predicates(c2_preds)

    cls1 = p1.class_value if rng.random() < 0.5 else p2.class_value
    cls2 = p2.class_value if rng.random() < 0.5 else p1.class_value

    return (
        _Individual(predicates=c1_preds, class_value=cls1),
        _Individual(predicates=c2_preds, class_value=cls2),
    )


def _dedup_predicates(preds: list[_Predicate]) -> list[_Predicate]:
    seen: set[int] = set()
    out: list[_Predicate] = []
    for p in preds:
        if p.feature_idx not in seen:
            seen.add(p.feature_idx)
            out.append(p)
    return out


def _clone_pred(p: _Predicate) -> _Predicate:
    return _Predicate(
        feature_idx=p.feature_idx,
        is_numeric=p.is_numeric,
        lo=p.lo,
        hi=p.hi,
        allowed=set(p.allowed),
    )


def _clone_individual(ind: _Individual) -> _Individual:
    return _Individual(
        predicates=[_clone_pred(p) for p in ind.predicates],
        class_value=ind.class_value,
        fitness=ind.fitness,
        accuracy2=ind.accuracy2,
        recall=ind.recall,
        theory_length=ind.theory_length,
        exceptions_length=ind.exceptions_length,
    )


def _mutate(
    rng: np.random.Generator,
    ind: _Individual,
    feature_info: list[dict],
    classes: np.ndarray,
    default_class: int | None,
) -> None:
    """Mutate one individual in-place."""
    # 10% chance: class mutation
    if rng.random() < 0.1:
        available = [c for c in classes if c != ind.class_value]
        if default_class is not None:
            available = [c for c in available if c != default_class]
        if available:
            ind.class_value = int(rng.choice(available))
        return

    # 90% chance: attribute mutation
    if len(ind.predicates) == 0:
        return
    pidx = rng.integers(0, len(ind.predicates))
    p = ind.predicates[pidx]
    info = feature_info[p.feature_idx]

    if p.is_numeric:
        domain = info["max"] - info["min"]
        offset = rng.uniform(-0.5, 0.5) * domain
        if rng.random() < 0.5:
            p.lo = np.clip(p.lo + offset, info["min"], info["max"])
        else:
            p.hi = np.clip(p.hi + offset, info["min"], info["max"])
        if p.lo > p.hi:
            p.lo, p.hi = p.hi, p.lo
    else:
        vals = list(info["values"])
        if vals:
            v = int(rng.choice(vals))
            if v in p.allowed:
                if len(p.allowed) > 1:
                    p.allowed.discard(v)
            else:
                p.allowed.add(v)


def _generalize_specialize(
    rng: np.random.Generator,
    ind: _Individual,
    feature_info: list[dict],
    prob_generalize: float,
    prob_specialize: float,
) -> None:
    """Apply BioHEL's generalize/specialize structural operators."""
    # Generalize: remove a random attribute (make it don't-care)
    if len(ind.predicates) > 0 and rng.random() < prob_generalize:
        idx = rng.integers(0, len(ind.predicates))
        del ind.predicates[idx]

    # Specialize: add a new random attribute
    if rng.random() < prob_specialize:
        expressed = {p.feature_idx for p in ind.predicates}
        unexpressed = [fi for fi in range(len(feature_info)) if fi not in expressed]
        if unexpressed:
            fi = int(rng.choice(unexpressed))
            info = feature_info[fi]
            if info["numeric"]:
                domain = info["max"] - info["min"]
                if domain > 0:
                    size = rng.uniform(0.25, 0.75) * domain
                    center = rng.uniform(info["min"], info["max"])
                    lo = max(info["min"], center - size / 2)
                    hi = min(info["max"], center + size / 2)
                    ind.predicates.append(_Predicate(
                        feature_idx=fi, is_numeric=True, lo=lo, hi=hi,
                    ))
            else:
                vals = info["values"]
                allowed = {int(rng.choice(list(vals)))}
                for v in vals:
                    if v not in allowed and rng.random() < 0.5:
                        allowed.add(v)
                if len(allowed) < len(vals):
                    ind.predicates.append(_Predicate(
                        feature_idx=fi, is_numeric=False, allowed=allowed,
                    ))


def _tournament_select_wor(
    rng: np.random.Generator,
    population: list[_Individual],
    tournament_size: int,
    n_select: int,
) -> list[_Individual]:
    """Tournament selection without replacement."""
    selected: list[_Individual] = []
    pop_size = len(population)
    for _ in range(n_select):
        contenders = rng.choice(pop_size, size=min(tournament_size, pop_size), replace=False)
        best_idx = contenders[0]
        for ci in contenders[1:]:
            if population[ci].fitness < population[best_idx].fitness:
                best_idx = ci
        selected.append(_clone_individual(population[best_idx]))
    return selected


def _is_majority_rule(
    ind: _Individual,
    X: np.ndarray,
    y: np.ndarray,
    coverage_break: float,
) -> bool:
    """Check BioHEL's majority condition for a rule."""
    match_mask = _matches_mask(ind, X)
    matched_y = y[match_mask]
    if len(matched_y) == 0:
        return False

    classes, counts_arr = np.unique(matched_y, return_counts=True)
    best_pos = int(np.argmax(counts_arr))
    best_class = int(classes[best_pos])
    if best_class != ind.class_value:
        return False

    # Check no tie
    if len(counts_arr) > 1:
        sorted_c = np.sort(counts_arr)[::-1]
        if sorted_c[0] == sorted_c[1]:
            return False

    # Coverage check
    total_positive = int(np.sum(y == ind.class_value))
    if total_positive == 0:
        return False
    tp = int(counts_arr[best_pos]) if best_class == ind.class_value else 0
    recall = tp / total_positive
    return recall >= coverage_break / 3.0


def _ilas_strata(
    rng: np.random.Generator,
    y: np.ndarray,
    n_strata: int,
) -> list[np.ndarray]:
    """Create ILAS strata indices with class-stratified round-robin."""
    if n_strata <= 1:
        return [np.arange(len(y))]
    strata: list[list[int]] = [[] for _ in range(n_strata)]
    classes = np.unique(y)
    for c in classes:
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)
        for j, idx in enumerate(c_indices):
            strata[j % n_strata].append(int(idx))
    return [np.array(s) for s in strata]


def _run_ga(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    feature_info: list[dict],
    classes: np.ndarray,
    default_class: int | None,
    *,
    population_size: int,
    n_iterations: int,
    tournament_size: int,
    crossover_prob: float,
    mutation_prob: float,
    prob_generalize: float,
    prob_specialize: float,
    prob_include: float,
    prob_one: float,
    coverage_break: float,
    coverage_ratio: float,
    mdl_initial_ratio: float,
    mdl_activate_iter: int,
    mdl_relax_factor: float,
    n_strata: int,
) -> _Individual:
    """Run one GA to find a single rule."""
    # Determine target classes (exclude default class)
    target_classes = [int(c) for c in classes]
    if default_class is not None:
        target_classes = [c for c in target_classes if c != default_class]
    if not target_classes:
        target_classes = [int(classes[0])]

    # Create ILAS strata
    strata = _ilas_strata(rng, y, n_strata)

    # Initialise population
    population: list[_Individual] = []
    for _ in range(population_size):
        tc = int(rng.choice(target_classes))
        ind = _smart_init(rng, X, y, len(feature_info), feature_info,
                          tc, prob_include, prob_one)
        population.append(ind)

    # Evaluate on first stratum
    stratum_idx = strata[0]
    X_w, y_w = X[stratum_idx], y[stratum_idx]
    for ind in population:
        _evaluate(ind, X_w, y_w, feature_info, coverage_break, coverage_ratio,
                  0.0, False)

    # Track best
    best = _clone_individual(min(population, key=lambda i: i.fitness))
    best_exceptions = best.exceptions_length
    no_improve_count = 0

    # MDL weight
    mdl_weight = 0.0
    mdl_active = False

    for iteration in range(n_iterations):
        # Windowing: pick stratum
        stratum_idx = strata[iteration % len(strata)]
        if iteration == n_iterations - 1:
            # Last iteration: use all data
            stratum_idx = np.arange(len(y))
        X_w, y_w = X[stratum_idx], y[stratum_idx]

        # Selection
        offspring = _tournament_select_wor(rng, population, tournament_size,
                                           population_size)

        # Crossover
        i = 0
        while i + 1 < len(offspring):
            if rng.random() < crossover_prob:
                c1, c2 = _crossover_hyperrect_list(rng, offspring[i], offspring[i + 1],
                                                    feature_info)
                offspring[i] = c1
                offspring[i + 1] = c2
            i += 2

        # Mutation + generalize/specialize
        for ind in offspring:
            if rng.random() < mutation_prob:
                _mutate(rng, ind, feature_info, classes, default_class)
            _generalize_specialize(rng, ind, feature_info,
                                   prob_generalize, prob_specialize)

        # Evaluate offspring
        for ind in offspring:
            _evaluate(ind, X_w, y_w, feature_info, coverage_break,
                      coverage_ratio, mdl_weight, mdl_active)

        # Elitism: inject best into population if better than worst
        population = offspring
        worst_idx = max(range(len(population)), key=lambda j: population[j].fitness)
        _evaluate(best, X_w, y_w, feature_info, coverage_break,
                  coverage_ratio, mdl_weight, mdl_active)
        if best.fitness < population[worst_idx].fitness:
            population[worst_idx] = _clone_individual(best)

        # Update best
        cur_best = min(population, key=lambda i: i.fitness)
        if cur_best.fitness < best.fitness:
            best = _clone_individual(cur_best)

        # MDL weight management
        if iteration == mdl_activate_iter and not mdl_active:
            # Activate MDL weight
            mdl_active = True
            if best.theory_length > 1e-12:
                mdl_weight = (mdl_initial_ratio / (1.0 - mdl_initial_ratio)) * \
                             (best.exceptions_length / best.theory_length)
            else:
                mdl_weight = 0.1
            # Re-evaluate everyone with MDL
            for ind in population:
                _evaluate(ind, X_w, y_w, feature_info, coverage_break,
                          coverage_ratio, mdl_weight, True)
            best = _clone_individual(min(population, key=lambda i: i.fitness))
            best_exceptions = best.exceptions_length
            no_improve_count = 0

        if mdl_active:
            cur_exc = min(population, key=lambda i: i.exceptions_length).exceptions_length
            if cur_exc < best_exceptions - 1e-9:
                best_exceptions = cur_exc
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 10:
                    mdl_weight *= mdl_relax_factor
                    no_improve_count = 0
                    # Re-evaluate with relaxed weight
                    for ind in population:
                        _evaluate(ind, X_w, y_w, feature_info, coverage_break,
                                  coverage_ratio, mdl_weight, True)
                    best = _clone_individual(min(population, key=lambda i: i.fitness))

    # Final: re-evaluate best on all data using exceptions only
    _evaluate(best, X, y, feature_info, coverage_break, coverage_ratio, 0.0, False)
    return best


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------

class RuleLCSClassifier(BaseRuleSetEstimator):
    """BioHEL-inspired iterative rule learning classifier.

    Learns one rule at a time via a Genetic Algorithm, then removes covered
    instances (sequential covering).  Uses an MDL-based fitness function,
    hyperrectangle representation with attribute lists, tournament selection
    without replacement, attribute-level crossover, and generalize/specialize
    structural mutation.

    Parameters
    ----------
    population_size : int
        Number of individuals in the GA population.
    n_iterations : int
        Number of GA generations per rule learning attempt.
    n_repetitions : int
        Number of independent GA runs per rule; the best rule is kept.
    tournament_size : int
        Tournament size for selection.
    crossover_prob : float
        Probability of crossover per pair.
    mutation_prob : float
        Probability of mutation per individual.
    prob_generalize : float
        Probability of removing an attribute (generalize operator).
    prob_specialize : float
        Probability of adding an attribute (specialize operator).
    prob_include : float
        Probability that an attribute is included during smart init.
    prob_one : float
        Probability that a categorical value is included during init.
    coverage_break : float
        Minimal recall threshold for the coverage penalty.
    coverage_ratio : float
        Scaling parameter for the coverage penalty curve.
    mdl_initial_ratio : float
        Initial theory-length ratio for MDL weight initialisation.
    mdl_activate_iter : int
        Iteration at which MDL weight is activated.
    mdl_relax_factor : float
        Factor by which MDL weight is relaxed upon stagnation.
    max_rules : int
        Maximum number of rules to learn (excluding default rule).
    max_consecutive_fails : int
        Stop if this many consecutive GA runs fail the majority check.
    n_strata : int
        Number of ILAS windowing strata (1 = no windowing).
    default_class_policy : str
        "major" = use majority class as default, "auto" = determine from
        remaining after IRL, "disabled" = no default class during GA.
    low_cardinality_threshold : int
        Numeric features with ≤ this many unique values are additionally
        treated as categorical.
    include_default_rule : bool
        Whether to append a default (catch-all) rule to the rule set.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 200,
        n_iterations: int = 50,
        n_repetitions: int = 2,
        tournament_size: int = 4,
        crossover_prob: float = 0.6,
        mutation_prob: float = 0.6,
        prob_generalize: float = 0.10,
        prob_specialize: float = 0.10,
        prob_include: float = 0.5,
        prob_one: float = 0.75,
        coverage_break: float = 0.01,
        coverage_ratio: float = 0.90,
        mdl_initial_ratio: float = 0.25,
        mdl_activate_iter: int = 10,
        mdl_relax_factor: float = 0.90,
        max_rules: int = 15,
        max_consecutive_fails: int = 3,
        n_strata: int = 1,
        default_class_policy: str = "major",
        low_cardinality_threshold: int = 10,
        include_default_rule: bool = True,
        random_state: int | None = None,
    ):
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.n_repetitions = n_repetitions
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.prob_generalize = prob_generalize
        self.prob_specialize = prob_specialize
        self.prob_include = prob_include
        self.prob_one = prob_one
        self.coverage_break = coverage_break
        self.coverage_ratio = coverage_ratio
        self.mdl_initial_ratio = mdl_initial_ratio
        self.mdl_activate_iter = mdl_activate_iter
        self.mdl_relax_factor = mdl_relax_factor
        self.max_rules = max_rules
        self.max_consecutive_fails = max_consecutive_fails
        self.n_strata = n_strata
        self.default_class_policy = default_class_policy
        self.low_cardinality_threshold = low_cardinality_threshold
        self.include_default_rule = include_default_rule
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        n_features = X.shape[1]
        self.feature_names_in_ = [f"x{i}" for i in range(n_features)]

        rng = np.random.default_rng(self.random_state)

        # Build feature info
        feature_info = self._build_feature_info(X)

        # Default class
        default_class: int | None = None
        if self.default_class_policy == "major":
            counts = np.bincount(y.astype(int))
            default_class = int(np.argmax(counts))
        elif self.default_class_policy == "disabled":
            default_class = None
        # "auto": default_class stays None during GA, determined later

        # Per-class coverage breakpoint
        total = len(y)
        class_counts = {int(c): int(np.sum(y == c)) for c in self.classes_}

        # Iterative rule learning
        rules: list[_Individual] = []
        remaining_mask = np.ones(len(y), dtype=bool)
        fail_count = 0

        for _rule_idx in range(self.max_rules):
            if np.sum(remaining_mask) == 0:
                break

            X_rem = X[remaining_mask]
            y_rem = y[remaining_mask]

            # Per-class coverage break
            cb = self.coverage_break
            # Scale for class imbalance as BioHEL does
            n_remaining = len(y_rem)

            best_rule: _Individual | None = None
            for _rep in range(self.n_repetitions):
                rule = _run_ga(
                    rng, X_rem, y_rem, feature_info, self.classes_,
                    default_class,
                    population_size=self.population_size,
                    n_iterations=self.n_iterations,
                    tournament_size=self.tournament_size,
                    crossover_prob=self.crossover_prob,
                    mutation_prob=self.mutation_prob,
                    prob_generalize=self.prob_generalize,
                    prob_specialize=self.prob_specialize,
                    prob_include=self.prob_include,
                    prob_one=self.prob_one,
                    coverage_break=cb,
                    coverage_ratio=self.coverage_ratio,
                    mdl_initial_ratio=self.mdl_initial_ratio,
                    mdl_activate_iter=self.mdl_activate_iter,
                    mdl_relax_factor=self.mdl_relax_factor,
                    n_strata=self.n_strata,
                )
                if best_rule is None or rule.fitness < best_rule.fitness:
                    best_rule = rule

            if best_rule is None:
                break

            # Check majority condition
            if _is_majority_rule(best_rule, X_rem, y_rem, cb):
                rules.append(best_rule)
                fail_count = 0
                # Remove matched instances of the rule's class (vectorised)
                idx_remaining = np.where(remaining_mask)[0]
                sub_match = _matches_mask(best_rule, X[idx_remaining])
                class_match = y[idx_remaining] == best_rule.class_value
                to_remove = idx_remaining[sub_match & class_match]
                remaining_mask[to_remove] = False
            else:
                fail_count += 1
                if fail_count >= self.max_consecutive_fails:
                    break

        # Determine default class from remaining instances
        remaining_y = y[remaining_mask]
        if len(remaining_y) > 0:
            counts = np.bincount(remaining_y.astype(int), minlength=len(self.classes_))
            default_class_final = int(np.argmax(counts))
        else:
            # All covered — default to majority of full data
            counts = np.bincount(y.astype(int), minlength=len(self.classes_))
            default_class_final = int(np.argmax(counts))

        # Build ScoredRuleSet
        self.rules_ = rules
        self.default_class_ = default_class_final
        self.ruleset_ = self._build_ruleset(rules, default_class_final)
        self.ruleset_.validate()
        return self

    # ------------------------------------------------------------- predict
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return predict_from_ruleset(self.ruleset_, X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return predict_proba_from_ruleset(self.ruleset_, X)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self)
        return self.ruleset_

    # ------------------------------------------------------------ internal
    def _build_feature_info(self, X: np.ndarray) -> list[dict]:
        """Analyse features: detect numeric vs categorical."""
        info: list[dict] = []
        for fi in range(X.shape[1]):
            col = X[:, fi]
            unique_vals = np.unique(col)
            n_unique = len(unique_vals)
            # Categorical heuristic: integer-valued and low cardinality
            is_cat = (
                n_unique <= self.low_cardinality_threshold
                and np.all(col == np.round(col))
            )
            if is_cat:
                info.append({
                    "numeric": False,
                    "values": set(int(v) for v in unique_vals),
                    "min": float(col.min()),
                    "max": float(col.max()),
                })
            else:
                info.append({
                    "numeric": True,
                    "values": set(),
                    "min": float(col.min()),
                    "max": float(col.max()),
                })
        return info

    def _build_ruleset(
        self,
        rules: list[_Individual],
        default_class: int,
    ) -> ScoredRuleSet:
        """Convert internal rules to ScoredRuleSet."""
        n_classes = len(self.classes_)
        scored_rules: list[Rule] = []

        for ridx, ind in enumerate(rules):
            atoms: list[Atom] = []
            for p in ind.predicates:
                fname = self.feature_names_in_[p.feature_idx]
                if p.is_numeric:
                    atoms.append(Atom(
                        feature=fname,
                        op="between",
                        value=[float(p.lo), float(p.hi)],
                    ))
                else:
                    if len(p.allowed) == 1:
                        atoms.append(Atom(
                            feature=fname,
                            op="==",
                            value=int(next(iter(p.allowed))),
                        ))
                    else:
                        atoms.append(Atom(
                            feature=fname,
                            op="in",
                            value=[int(v) for v in sorted(p.allowed)],
                        ))

            # Scores: high for the predicted class
            scores = [0.0] * n_classes
            class_idx = list(self.classes_).index(ind.class_value)
            scores[class_idx] = 1.0
            scored_rules.append(Rule(
                atoms=atoms,
                scores=scores,
                rule_id=f"R{ridx}",
            ))

        # Default rule
        if self.include_default_rule:
            default_scores = [0.0] * n_classes
            default_idx = list(self.classes_).index(default_class)
            default_scores[default_idx] = 1.0
            scored_rules.append(Rule(
                atoms=[],
                scores=default_scores,
                rule_id="default",
                metadata={"default_rule": True},
            ))

        return ScoredRuleSet(
            class_labels=[int(c) for c in self.classes_],
            rules=scored_rules,
            feature_names=list(self.feature_names_in_),
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={
                "estimator": "RuleLCSClassifier",
                "algorithm": "BioHEL-inspired IRL",
                "n_rules": len(rules),
                "default_class": int(default_class),
            },
        )
