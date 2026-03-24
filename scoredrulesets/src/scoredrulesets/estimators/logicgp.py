"""
logicGP – Python-Implementierung (logicGP-FLCW Variante)
=========================================================
Basiert auf dem Paper:
  Nunkesser (GECCO'25): "logicGP -- A Framework for Literal Based
  Classification with a Focus on Software Architecture and Open Source
  Implementation"
  DOI: 10.1145/3712255.3734300

und dem C#-Referenz-Repository:
  https://github.com/Italbytz/nuget-adapters-algorithms-ea

Implementiert wird die logicGP-FLCW-Macro/Micro-Variante fuer
Mehrklassen-Klassifikation mit kategorischen oder diskretisierten
kontinuierlichen Features.

Vorhersagemodell
----------------
  M_hat = sum_i  w_i  * prod_j ( 1 if l_{i,j}(x) else 0 )
  Y_hat = M_hat  falls irgendein Monom gefeuert hat, sonst w_0
  G_hat = argmax_c Y_hat[c]

Gewichte (FLCW)
---------------
  w_i[c] = P(Klasse=c | Monom i feuert)   (= relative Klassenhaeufigkeit
            unter den Trainingsinstanzen, fuer die Monom i wahr ist)
  w_0[c] = relative Klassenhaeufigkeit unter den Instanzen, fuer die
            KEIN Monom feuert

GP-Algorithmus
--------------
  Initialpopulation: ein Individuum pro Literal
  Pro Generation:
    - 2 Eltern -> Crossover -> 1 Kind
    - 5 Mutationstypen, je 1 Individuum -> 5 neue Individuen
    - Reproduktion: alle aktuellen Individuen
    - Selektion: Pareto-Dominanz-Selektion (behalte nicht-dominierte)
  Fitness: pro Klasse Recall + Modellgroesse (Anzahl distinkte Literale)
  Terminierung: max_generations oder Stagnation
  Finale Modellauswahl: bestes Modell je Groesse, dann Auswahlstrategie
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Union

import numpy as np
from sklearn.metrics import f1_score as _f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
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
    Gewichtetes Polynom: Liste von Monomen + Default-Gewichte w_0.
    Vorhersage:
      M_hat = sum der Monom-Gewichte fuer gefeuerte Monome
      Y_hat = M_hat falls irgendein Monom feuerte, sonst w_0
      G_hat = argmax(Y_hat)
    """
    monomials: list[_Monomial]
    default_weights: np.ndarray  # w_0, shape (n_classes,)

    @property
    def size(self) -> int:
        """Anzahl distinkte Literale im Polynom."""
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
    Multi-objective Fitness: per-Klasse Recall + Modellgroesse.
    Objective[c] = Recall_c = TP_c / max(N_c, 1)
    Size = Anzahl distinkte Literale (minimize)
    """
    objectives: np.ndarray  # shape (n_classes,), recall per class
    size: int

    def dominates(self, other: "_Fitness") -> bool:
        """True iff self dominiert other (mindestens gleich gut, mind. 1 besser)."""
        if self.size > other.size:
            # size muss <= sein fuer Dominanz
            return False
        # per-Klasse Recall muss >= sein
        if np.any(self.objectives < other.objectives - 1e-12):
            return False
        # mind. eine Komponente strikt besser
        return (self.size < other.size) or np.any(self.objectives > other.objectives + 1e-12)

    @property
    def consolidated(self) -> float:
        """Zusammengefasstes Fitness-Mass fuer finale Sortierung."""
        return float(np.mean(self.objectives))


@dataclass
class _FitnessRLCW:
    """
    3-Objective Fitness fuer RLCW-Trainer (Restricted Literals Computed Weights).

    Objective 1 (max_recall):   max per-class recall
    Objective 2 (mean_other):   mean of remaining per-class recalls (ohne argmax-Klasse)
    Objective 3 (size):         Modellgroesse (minimize)

    Domination: A dominiert B nur wenn
      - A.best_class == B.best_class  (class-bound Dominanz)
      - A.size <= B.size
      - A.max_recall >= B.max_recall
      - A.mean_other >= B.mean_other
      - mindestens eine Komponente strikt besser
    """
    max_recall: float    # obj1: max(per_class_recalls)
    mean_other: float    # obj2: mean der uebrigen per-class recalls
    size: int            # obj3: Modellgroesse (minimize)
    best_class: int      # argmax der per-class recalls (fuer class-bound Vergleich)

    def dominates(self, other: "_FitnessRLCW") -> bool:
        """True iff self dominiert other (class-bound Pareto-Dominanz)."""
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
        """Zusammengefasstes Fitness-Mass fuer finale Sortierung."""
        return (self.max_recall + self.mean_other) / 2.0


# ---------------------------------------------------------------------------
# Gewichtsberechnung (FLCW)
# ---------------------------------------------------------------------------

def _compute_weights(
    poly: _Polynomial,
    X_disc: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
) -> None:
    """
    Berechnet Gewichte in-place fuer FLCW-Variante.
    w_i = Klassenverteilung unter Instanzen, fuer die Monom i feuert.
    w_0 = Klassenverteilung unter Instanzen, fuer die KEIN Monom feuert.
    """
    n = X_disc.shape[0]
    any_fired = np.zeros(n, dtype=bool)

    for mon in poly.monomials:
        mask = mon.fires_mask(X_disc)
        any_fired |= mask
        counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
        total = counts.sum()
        if total > 0:
            mon.weights = counts / total
        else:
            mon.weights = np.ones(n_classes, dtype=float) / n_classes

    # Default-Gewichte: wo kein Monom gefeuert hat
    no_fire_mask = ~any_fired
    if no_fire_mask.any():
        counts0 = np.bincount(y_idx[no_fire_mask], minlength=n_classes).astype(float)
        total0 = counts0.sum()
        poly.default_weights = counts0 / total0 if total0 > 0 else np.ones(n_classes) / n_classes
    else:
        # Fallback: gleichmaessige Verteilung
        poly.default_weights = np.ones(n_classes, dtype=float) / n_classes


# ---------------------------------------------------------------------------
# Fitness-Berechnung
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
    Berechnet RLCW-Fitness mit 3 Zielen:
    1. max per-class recall
    2. mean der uebrigen per-class recalls
    3. Modellgroesse (minimize)
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
    """Behaelt nur nicht-dominierte Individuen."""
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
    """Behaelt nur nicht-dominierte Individuen (class-bound RLCW-Dominanz)."""
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
    Trimmt eine Population auf n_keep Individuen durch Turnier-Selektion.
    Wird verwendet, wenn die Pareto-Front die Populationsgroesse uebersteigt.
    """
    if len(individuals) <= n_keep:
        return individuals

    selected = []
    # Arbeite mit Indizes, um numpy-Array-Vergleiche bei pool.remove() zu vermeiden
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
# Mutations- und Crossover-Operatoren
# ---------------------------------------------------------------------------

def _crossover(
    poly_a: _Polynomial,
    poly_b: _Polynomial,
    rng: np.random.Generator,
) -> _Polynomial:
    """
    Crossover: Nehme poly_b als Basis, ersetze ein zufaelliges Monom durch
    ein zufaelliges Monom aus poly_a.
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
    """Fuegt ein zufaelliges Literal in ein zufaelliges Monom ein."""
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
    """Loescht ein zufaelliges Literal aus einem zufaelligen Monom."""
    result = poly.clone()
    # Nur Monome mit mehr als einem Literal kommen in Frage
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
    """Ersetzt ein zufaelliges Literal durch ein neues zufaelliges Literal."""
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
    """Fuegt ein neues Monom (einzelnes Literal) hinzu."""
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
    """Loescht ein zufaelliges Monom."""
    if len(poly.monomials) <= 1:
        return poly
    result = poly.clone()
    idx = int(rng.integers(0, len(result.monomials)))
    result.monomials.pop(idx)
    return result


# ---------------------------------------------------------------------------
# Literal-Generierung (Suchraum)
# ---------------------------------------------------------------------------

def _generate_literals(X_disc: np.ndarray) -> list[_SetLiteral]:
    """
    Generiert alle nicht-trivialen Teilmengen-Literale (SetLiterals) fuer
    jedes Feature in X_disc. Entspricht dem FLCW-Suchraum (Full Literals).
    """
    n_features = X_disc.shape[1]
    literals: list[_SetLiteral] = []
    for feat_idx in range(n_features):
        col = X_disc[:, feat_idx]
        cats = sorted(set(col.tolist()))
        k = len(cats)
        if k < 2:
            continue  # Kein Literal moeglich fuer konstante Features
        all_cats = tuple(cats)
        power_set_count = 1 << k
        for bitmask in range(1, power_set_count - 1):  # alle nicht-trivialen Teilmengen
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
# Finale Modellauswahl (aus paper: Algorithm 1)
# ---------------------------------------------------------------------------

def _final_model_selection(
    candidates: list[tuple[_Polynomial, _Fitness, float]],  # (poly, fit, val_accuracy)
    min_improvement: float = 0.01,
) -> _Polynomial:
    """
    Implementiert die Auswahlstrategie aus dem logicGP-Paper:
    1. Behalte bestes Modell je Groesse (Konsolidierung)
    2. Verwerfe Modelle, die weniger als 1% Verbesserung gegenueber kleineren bieten
    3. Waehle groesstes Modell mit maximaler Genauigkeit
    """
    if not candidates:
        raise ValueError("Keine Kandidaten vorhanden.")

    # Schritt 1: Bestes Modell je Groesse
    best_per_size: dict[int, tuple[_Polynomial, _Fitness, float]] = {}
    for poly, fit, acc in candidates:
        s = fit.size
        if s not in best_per_size or acc > best_per_size[s][2]:
            best_per_size[s] = (poly, fit, acc)

    sorted_by_size = sorted(best_per_size.values(), key=lambda x: x[1].size)

    # Schritt 2: Verwerfe Modelle ohne ausreichende Verbesserung ggb. kleineren
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

    # Schritt 3: Waehle Modell mit maximaler Genauigkeit (bei Gleichstand: groesstes)
    best = max(filtered, key=lambda x: (x[2], x[1].size))
    return best[0]


# ---------------------------------------------------------------------------
# Diskretisierung kontinuierlicher Features
# ---------------------------------------------------------------------------

def _discretize_features(
    X: np.ndarray,
    n_bins: int = 5,
    fitted_binners: list | None = None,
    cat_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Diskretisiert kontinuierliche Features in Bin-Indizes.
    Kategorische Features (nicht-numerisch oder <= n_bins unique Werte) bleiben unveraendert.
    Gibt (X_disc, fitted_binners, cat_mask) zurueck.
    cat_mask[i] = True -> Feature i wurde als kategorisch behandelt (kein Binner).
    """
    n_samples, n_features = X.shape
    X_disc = np.empty((n_samples, n_features), dtype=object)

    is_fit = fitted_binners is not None

    if not is_fit:
        fitted_binners = []
        cat_masks_list = []
    else:
        cat_masks_list = None  # wird nicht benoetigt

    for f in range(n_features):
        col = X[:, f]
        arr = np.asarray(col, dtype=object)

        is_numeric = False
        float_col = None
        try:
            float_col = arr.astype(float)
            is_numeric = True
        except (ValueError, TypeError):
            pass

        if is_numeric and float_col is not None:
            unique_vals = np.unique(float_col)
            n_unique = len(unique_vals)
            is_cat = n_unique <= n_bins
        else:
            is_cat = True
            float_col = None
            n_unique = len(np.unique(arr))

        if not is_fit:
            cat_masks_list.append(is_cat)

        if is_cat or not is_numeric:
            X_disc[:, f] = arr
            if not is_fit:
                fitted_binners.append(None)
        else:
            actual_bins = min(n_bins, n_unique)
            if not is_fit:
                binner = KBinsDiscretizer(
                    n_bins=actual_bins,
                    encode="ordinal",
                    strategy="quantile",
                    subsample=None,
                )
                binner.fit(float_col.reshape(-1, 1))
                fitted_binners.append(binner)
            else:
                binner = fitted_binners[f]

            if binner is not None:
                binned = binner.transform(float_col.reshape(-1, 1)).ravel().astype(int)
                X_disc[:, f] = binned
            else:
                X_disc[:, f] = arr

    if not is_fit:
        cat_masks_arr = np.array(cat_masks_list, dtype=bool)
    else:
        cat_masks_arr = cat_masks

    return X_disc, fitted_binners, cat_masks_arr


# ---------------------------------------------------------------------------
# Hauptklasse: LogicGPClassifier
# ---------------------------------------------------------------------------

class LogicGPClassifier(BaseRuleSetEstimator):
    """
    logicGP Python-Implementierung (FLCW und RLCW Varianten).

    Unterstuetzt zwei Trainer-Familien:

    **FLCW** (Full Literals Computed Weights, Original-Variante):
      - ``trainer="flcw_macro"`` oder ``"flcw_micro"``
      - Alle nicht-trivialen Teilmengenliterale als Suchraum
      - n_classes Pareto-Ziele (per-class Recall + Groesse)

    **RLCW** (Restricted Literals Computed Weights, effizientere Variante):
      - ``trainer="rlcw_macro"`` oder ``"rlcw_micro"``
      - Eingeschraenkter Suchraum via ``min_max_weight``
      - 3 Pareto-Ziele (max-Recall, mean-other-Recall, Groesse)
      - Class-bound Pareto-Dominanz (foerdert Klassenvielfalt)
      - Optionale Populationsgroessen-Begrenzung mit Turnier-Selektion

    Parameters
    ----------
    trainer : str
        Trainer-Variante: ``"flcw_macro"`` (Standard), ``"flcw_micro"``,
        ``"rlcw_macro"``, ``"rlcw_micro"``.
    max_generations : int
        Maximale Anzahl von GP-Generationen.
    stagnation_generations : int
        Abbruch nach dieser Anzahl von Generationen ohne Verbesserung.
    n_bins : int
        Anzahl Bins fuer Diskretisierung kontinuierlicher Features.
    min_max_weight : float
        Literale herausfiltern, deren maximales Klassengewicht <= diesem Wert.
        0.0 = kein Filter. Empfohlen fuer RLCW: 0.1–0.3.
    min_improvement_pct : float
        Mindestverbesserung in Prozent fuer Modellauswahl (Standard: 0.01 = 1 %).
    population_size : int or None
        Maximale Populationsgroesse nach Selektion. Falls ``None``, wird die
        vollstaendige Pareto-Front behalten (Standard: FLCW-Verhalten).
        Bei RLCW empfohlen: 30–100.
    n_adaptations_per_gen : int
        Anzahl neuer Individuen pro Generation. Standardmaessig 6
        (1 Crossover + 5 Mutationen). Fuer RLCW kann ein hoehere Wert
        (z. B. 12–24) die Konvergenz verbessern.
    tournament_size : int
        Turniergrösse fuer Turnier-Selektion (nur aktiv wenn
        ``population_size`` gesetzt und Pareto-Front > ``population_size``).
    max_model_size : int or None
        Maximale Modellgroesse (Anzahl Literale). Falls gesetzt, werden
        Individuen mit groesserer Modellgroesse ignoriert (Suchphase des
        Zwei-Phasen-Modellauswahl-Algorithmus aus logicGP-RLCW).
    validation_fraction : float
        Anteil der Trainingsdaten, der als Validierungsmenge fuer die
        finale Modellauswahl reserviert wird.  Die Auswahl nutzt dann die
        tatsaechliche Macro-F1 auf dem Val-Set statt auf den Trainingsdaten.
        ``0`` (Standard) deaktiviert den Split – die Modellauswahl erfolgt
        dann anhand der Training-Macro-F1, die deutlich zuverlaessiger ist
        als ``consolidated`` (mean recall) und keinen Datenverlust verursacht.
        Werte > 0 sind nur fuer grosse Datensaetze empfohlen (n >= 200).
    max_fit_seconds : float or None
        Maximale Laufzeit fuer den GP-Loop in Sekunden. Falls gesetzt,
        wird die Evolution nach Ablauf der Zeit sauber abgebrochen und
        das bis dahin beste Modell zurueckgegeben. ``None`` (Standard)
        deaktiviert das Zeitlimit. Empfohlen im Benchmark-Kontext:
        ein Wert unter dem aeusseren Timeout (z. B. 240 bei 300s-Timeout).
    random_state : int or None
    """

    def __init__(
        self,
        trainer: str = "flcw_macro",
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
        random_state: int | None = None,
    ):
        self.trainer = trainer
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

        # Diskretisierung
        X_disc, self._binners_, self._cat_masks_ = _discretize_features(
            X_valid, n_bins=self.n_bins
        )

        # ----- Validation-Split fuer bessere finale Modellauswahl -----
        use_val = (
            self.validation_fraction > 0
            and X_disc.shape[0] >= 30          # zu wenig Samples -> kein Split
            and len(np.unique(y_idx)) >= 2     # mindestens 2 Klassen
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
                # Stratifizierung fehlgeschlagen (zu wenig Samples pro Klasse)
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
                "Keine Literale generierbar. Bitte Features oder n_bins pruefen."
            )

        # Initialpopulation: ein Individuum pro Literal + Seeds
        population = self._init_population(
            all_literals, n_classes, X_disc_train, y_idx_train
        )

        # GP-Schleife (auf Train-Split)
        best_poly = self._run_gp(
            population, all_literals, X_disc_train, y_idx_train, n_classes,
            X_val=X_disc_val, y_val=y_idx_val,
        )

        # Gewichte auf GESAMTEN Daten neu berechnen
        if use_val:
            _compute_weights(best_poly, X_disc, y_idx, n_classes)

        # ScoredRuleSet erstellen
        self.ruleset_ = self._poly_to_ruleset(best_poly, n_classes)
        self.ruleset_.validate()
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr: np.ndarray = np.asarray(check_array(X, dtype=None))
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X hat {X_arr.shape[1]} Features, erwartet {self.n_features_in_}."
            )
        X_disc, _, _ = _discretize_features(
            X_arr, fitted_binners=self._binners_, cat_masks=self._cat_masks_
        )
        return predict_from_ruleset(self.ruleset_, X_disc)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr: np.ndarray = np.asarray(check_array(X, dtype=None))
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X hat {X_arr.shape[1]} Features, erwartet {self.n_features_in_}."
            )
        X_disc, _, _ = _discretize_features(
            X_arr, fitted_binners=self._binners_, cat_masks=self._cat_masks_
        )
        return predict_proba_from_ruleset(self.ruleset_, X_disc)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

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
        raise ValueError(
            f"model_selection must be a string or callable, got {type(sel)}."
        )

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
        """Filtert Literale heraus, deren maximales Klassengewicht zu gering ist."""
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
        """Erstellt Initialpopulation: ein Individuum pro Literal + Seeds.

        Zusaetzlich zu den Standard-Einzel-Literal-Individuen werden
        klassen-diskriminative Seed-Individuen erzeugt:
        1. Fuer jede Klasse: beste 2-Literal-Konjunktion (verschiedene Features)
        2. Multi-Monom-Individuen: je ein Monom pro Klasse (fuer Mehrklassen)
        Dies beschleunigt die Konvergenz besonders bei Mehrklassen-Problemen.
        """
        population = []
        # Standard: ein Individuum pro Literal
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

        # Klassen-diskriminatives Seeding (nur wenn X_disc/y_idx vorhanden)
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
        """Erzeugt klassen-diskriminative Seed-Individuen.

        Fuer jede Klasse wird das Literal mit der besten Kombination aus
        Purity (Anteil der Klasse unter den feuernden Samples) und Coverage
        (Anteil der korrekt abgedeckten Klasseninstanzen) ermittelt.
        Daraus werden Multi-Literal-Monome und Multi-Monom-Polynome erzeugt.
        """
        seeds: list[_Polynomial] = []
        n_samples = X_disc.shape[0]
        unif = np.ones(n_classes, dtype=float) / n_classes

        # Evaluiere Diskriminationskraft jedes Literals pro Klasse
        lit_fire_masks: list[np.ndarray] = []
        for lit in all_literals:
            col = X_disc[:, lit.feature_idx]
            cats = np.array(sorted(lit.category_set))
            mask = np.isin(col, cats)
            lit_fire_masks.append(mask)

        # Finde bestes Literal pro Klasse (nach purity * coverage)
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

        # Sortiere und behalte Top-5 pro Klasse
        top_k = 5
        class_top_lits: list[list[int]] = []
        for c in range(n_classes):
            best_per_class[c].sort(key=lambda x: x[1], reverse=True)
            class_top_lits.append([idx for idx, _ in best_per_class[c][:top_k]])

        # Seed 1: Fuer jede Klasse ein 2-Literal-Monom (verschiedene Features)
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
                        break  # ein Seed pro Klasse reicht

        # Seed 2: Multi-Monom-Polynom – ein Monom pro Klasse
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
    ) -> _Polynomial:
        """
        Fuehrt den GP-Hauptloop aus und gibt das beste Polynom zurueck.
        Unterstuetzt FLCW- und RLCW-Varianten basierend auf ``self.trainer``.

        Die finale Modellauswahl bewertet Kandidaten anhand der
        tatsaechlichen F1 (statt ``fit.consolidated`` = mean recall).
        Macro- oder Micro-Averaging wird aus dem ``trainer``-Suffix
        abgeleitet (``*_macro`` → ``average="macro"``,
        ``*_micro`` → ``average="micro"``).
        Wenn ``X_val``/``y_val`` gegeben: F1 auf Validation-Set.
        Sonst: F1 auf Trainingsdaten (kein Datenverlust, zuverlaessiger
        fuer kleine Datensaetze).
        """
        use_rlcw = self.trainer.lower().startswith("rlcw")
        f1_average = "micro" if self.trainer.lower().endswith("_micro") else "macro"
        has_val = X_val is not None and y_val is not None

        # Konfigurierbare Strategien aufloesen
        resolved_fe = self._resolve_fitness_evaluator()
        if resolved_fe is not None:
            evaluate_fn = resolved_fe
            use_rlcw = (evaluate_fn is _evaluate_fitness_rlcw)
        else:
            evaluate_fn = _evaluate_fitness_rlcw if use_rlcw else _evaluate_fitness
        pareto_fn = _pareto_front_rlcw if use_rlcw else _pareto_front
        select_model_fn = self._resolve_model_selector()

        # ------------------------------------------------------------------
        # Gewichte berechnen und Initialpopulation evaluieren
        # ------------------------------------------------------------------
        for poly in population:
            _compute_weights(poly, X_disc, y_idx, n_classes)

        evaluated: list = [
            (poly, evaluate_fn(poly, X_disc, y_idx, n_classes))
            for poly in population
        ]
        evaluated = pareto_fn(evaluated)

        # Populationsgroesse begrenzen (RLCW mit population_size)
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
        # Elitismus: Verfolge das beste Individuum nach Macro-F1, um
        # Regression zu verhindern.  Dies beschleunigt die Konvergenz
        # besonders bei Mehrklassen-Problemen.
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

        # Mutationsoperatoren als Liste fuer zyklisches Durchlaufen
        _MUT_OPS = [
            lambda p: _mut_insert_literal(p, all_literals, self._rng_),
            lambda p: _mut_delete_literal(p, self._rng_),
            lambda p: _mut_replace_literal(p, all_literals, self._rng_),
            lambda p: _mut_insert_monomial(p, all_literals, n_classes, self._rng_),
            lambda p: _mut_delete_monomial(p, self._rng_),
        ]
        n_adapt = max(1, self.n_adaptations_per_gen)

        # Zeitbudget fuer fruehzeitigen Abbruch (max_fit_seconds)
        _gp_start_time = time.monotonic()
        _time_budget = self.max_fit_seconds  # None = kein Limit

        for _gen in range(self.max_generations):
            # Zeitbudget pruefen
            if _time_budget is not None:
                elapsed = time.monotonic() - _gp_start_time
                if elapsed >= _time_budget:
                    break

            # ------------------------------------------------------------------
            # Neue Individuen generieren (n_adaptations_per_gen Stueck)
            # ------------------------------------------------------------------
            new_polys: list[_Polynomial] = []

            for i in range(n_adapt):
                if i == 0 and len(evaluated) >= 2:
                    # Crossover: erstes Kind ist Crossover-Kind
                    p_a, p_b = self._select_two_parents(evaluated)
                    new_polys.append(_crossover(p_a, p_b, self._rng_))
                else:
                    # Mutation: zyklisch durch alle 5 Operator-Typen
                    parent = self._select_parent(evaluated)
                    new_polys.append(_MUT_OPS[(i - 1) % 5](parent))

            # ------------------------------------------------------------------
            # Fitness fuer neue Individuen
            # ------------------------------------------------------------------
            new_evaluated: list = []
            for poly in new_polys:
                if not poly.monomials:
                    continue
                # Groessenbeschraenkung (Suchphase RLCW)
                if self.max_model_size is not None and poly.size > self.max_model_size:
                    continue
                _compute_weights(poly, X_disc, y_idx, n_classes)
                fit = evaluate_fn(poly, X_disc, y_idx, n_classes)
                new_evaluated.append((poly, fit))

            # ------------------------------------------------------------------
            # Pareto-Selektion + optionale Populationsgroessen-Begrenzung
            # ------------------------------------------------------------------
            combined = evaluated + new_evaluated
            evaluated = pareto_fn(combined)

            if self.population_size is not None and len(evaluated) > self.population_size:
                evaluated = _tournament_trim(
                    evaluated, self.population_size, self.tournament_size, self._rng_
                )

            # ------------------------------------------------------------------
            # Elitismus: Elite-Individuum in Population sicherstellen
            # ------------------------------------------------------------------
            if elite_poly is not None and elite_fit is not None:
                # Prüfe ob Elite noch in der Population ist
                elite_ids = {id(p) for p, _ in evaluated}
                if id(elite_poly) not in elite_ids:
                    evaluated.append((elite_poly, elite_fit))

            # Elite-Update: prüfe ob ein neues bestes Individuum existiert
            for poly, fit in new_evaluated:
                preds = poly.predict_classes(eval_X_elite)
                f1 = float(_f1_score(eval_y_elite, preds, average=f1_average, labels=labels))
                if f1 > elite_f1:
                    elite_f1 = f1
                    elite_poly = poly.clone()
                    elite_fit = fit

            # ------------------------------------------------------------------
            # Stagnations-Tracking (immer auf consolidated, da guenstig)
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
        # Finale Modellauswahl
        # ------------------------------------------------------------------
        # Bewerte Kandidaten anhand der tatsaechlichen Macro-F1 statt
        # ``consolidated`` (mean recall), da F1 besser mit der realen
        # Vorhersagequalitaet korreliert.
        # Bei aktivem Validation-Split: F1 auf Val-Set (Generalisierung).
        # Ohne Split: F1 auf Trainingsdaten (kein Datenverlust).
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

        return select_model_fn(f1_candidates, self.min_improvement_pct)

    def _select_two_parents(self, evaluated: list) -> tuple[_Polynomial, _Polynomial]:
        """Waehlt zwei Eltern per Domination-Turnier."""
        idx_a, idx_b = self._rng_.choice(len(evaluated), size=2, replace=False)
        poly_a, fit_a = evaluated[idx_a]
        poly_b, fit_b = evaluated[idx_b]
        if fit_a.dominates(fit_b):
            return poly_a, poly_b
        if fit_b.dominates(fit_a):
            return poly_b, poly_a
        return poly_a, poly_b

    def _select_parent(self, evaluated: list) -> _Polynomial:
        """Zufaellige Elternauswahl aus der aktuellen Population."""
        idx = int(self._rng_.integers(0, len(evaluated)))
        return evaluated[idx][0]

    def _poly_to_ruleset(self, poly: _Polynomial, n_classes: int) -> ScoredRuleSet:
        """Konvertiert ein _Polynomial in ein ScoredRuleSet (Gewichte bereits gesetzt)."""
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

        return ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=feature_names,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            rules=rules,
            metadata={
                "source": "logicgp",
                "trainer": self.trainer,
                "n_monomials": len(poly.monomials),
                "model_size": poly.size,
                "max_generations": self.max_generations,
                "n_bins": self.n_bins,
                "population_size": self.population_size,
                "n_adaptations_per_gen": self.n_adaptations_per_gen,
                "max_model_size": self.max_model_size,
                "literal_generator": self._strategy_name(self.literal_generator),
                "model_selection": self._strategy_name(self.model_selection),
            },
        )

    def _monomial_to_atoms(self, mon: _Monomial) -> list[Atom]:
        """Konvertiert Monomial-Literale in Atom-Liste fuer ScoredRuleSet."""
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











