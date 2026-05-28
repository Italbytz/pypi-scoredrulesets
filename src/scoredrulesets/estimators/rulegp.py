"""
ruleGP - logicGP-RLCW style evolution on native atom/rule/rule-set structures
=============================================================================

This estimator is intentionally independent from logicGP internals.
It mirrors the RLCW workflow (class-bound Pareto dominance + 3 objectives)
while using atom-based rules and native numeric/categorical handling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from sklearn.metrics import f1_score as _f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .atom_space import NativeAtomSpaceStrategy
from .atom_space import build_native_feature_specs
from .base import BaseRuleSetEstimator


AtomPreselectionStrategy = Literal[
    "none",
    "logicgp_singleton",
    "logicgp_binned_sets",
]

ObjectiveMode = Literal["recall", "f1"]


@dataclass(frozen=True)
class _AtomGene2:
    feature_idx: int
    op: str
    value: object


@dataclass
class _Rule2:
    atoms: list[_AtomGene2]
    weights: np.ndarray

    def clone(self) -> "_Rule2":
        return _Rule2(atoms=list(self.atoms), weights=self.weights.copy())


@dataclass
class _RuleSet2:
    rules: list[_Rule2]
    default_weights: np.ndarray

    @staticmethod
    def _atom_key(atom: _AtomGene2) -> tuple[int, str, object]:
        if atom.op in ("between", "in"):
            return (atom.feature_idx, atom.op, tuple(atom.value))
        return (atom.feature_idx, atom.op, atom.value)

    @property
    def size(self) -> int:
        # Match logicGP's complexity notion: count unique predicates.
        seen: set[tuple[int, str, object]] = set()
        for rule in self.rules:
            for atom in rule.atoms:
                seen.add(self._atom_key(atom))
        return len(seen)

    def clone(self) -> "_RuleSet2":
        return _RuleSet2(
            rules=[r.clone() for r in self.rules],
            default_weights=self.default_weights.copy(),
        )

    def _atom_mask(self, atom: _AtomGene2, X: np.ndarray) -> np.ndarray:
        col = X[:, atom.feature_idx]
        if atom.op == "<=":
            return np.asarray(col, dtype=float) <= float(atom.value)
        if atom.op == "<":
            return np.asarray(col, dtype=float) < float(atom.value)
        if atom.op == ">=":
            return np.asarray(col, dtype=float) >= float(atom.value)
        if atom.op == ">":
            return np.asarray(col, dtype=float) > float(atom.value)
        if atom.op == "between":
            lo, hi = atom.value
            c = np.asarray(col, dtype=float)
            return (c >= float(lo)) & (c <= float(hi))
        if atom.op == "==":
            try:
                return (np.asarray(col, dtype=float) == float(atom.value)).astype(bool)
            except (TypeError, ValueError):
                return (np.asarray(col, dtype=object) == atom.value).astype(bool)
        if atom.op == "in":
            return np.isin(np.asarray(col, dtype=object), list(atom.value)).astype(bool)
        return np.ones(X.shape[0], dtype=bool)

    def _rule_mask(self, rule: _Rule2, X: np.ndarray) -> np.ndarray:
        if not rule.atoms:
            return np.zeros(X.shape[0], dtype=bool)
        mask = np.ones(X.shape[0], dtype=bool)
        for atom in rule.atoms:
            mask &= self._atom_mask(atom, X)
        return mask

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_classes = self.default_weights.shape[0]
        scores = np.zeros((n_samples, n_classes), dtype=float)
        any_fired = np.zeros(n_samples, dtype=bool)
        for rule in self.rules:
            mask = self._rule_mask(rule, X)
            if mask.any():
                scores[mask] += rule.weights
                any_fired |= mask
        scores[~any_fired] = self.default_weights
        return scores

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)


@dataclass(frozen=True)
class _FitnessRLCW2:
    max_recall: float
    mean_other_recall: float
    size: int
    best_class: int

    @property
    def consolidated(self) -> float:
        return (self.max_recall + self.mean_other_recall) / 2.0

    def dominates(self, other: "_FitnessRLCW2") -> bool:
        if self.best_class != other.best_class:
            return False
        ge = (
            self.max_recall >= other.max_recall
            and self.mean_other_recall >= other.mean_other_recall
            and self.size <= other.size
        )
        gt = (
            self.max_recall > other.max_recall
            or self.mean_other_recall > other.mean_other_recall
            or self.size < other.size
        )
        return ge and gt


def _build_feature_specs(
    X: np.ndarray,
    max_thresholds: int | None = None,
    low_cardinality_threshold: int = 10,
    atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
) -> list[dict[str, Any]]:
    return build_native_feature_specs(
        X,
        max_thresholds=max_thresholds,
        low_cardinality_threshold=low_cardinality_threshold,
        strategy=atom_space_strategy,
    )


def _evaluate_fitness_rlcw2(
    rs: _RuleSet2,
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    objective_mode: ObjectiveMode,
) -> _FitnessRLCW2:
    pred = rs.predict_classes(X)

    if objective_mode == "recall":
        conf = np.zeros((n_classes, n_classes), dtype=float)
        for yt, yp in zip(y_idx, pred):
            conf[yt, yp] += 1.0
        per_class_metric = np.divide(
            np.diag(conf),
            conf.sum(axis=1),
            out=np.zeros(n_classes, dtype=float),
            where=conf.sum(axis=1) > 0,
        )
    elif objective_mode == "f1":
        per_class_metric = np.asarray(
            _f1_score(
                y_idx,
                pred,
                average=None,
                labels=list(range(n_classes)),
                zero_division=0,
            ),
            dtype=float,
        )
    else:
        raise ValueError("objective_mode must be 'recall' or 'f1'.")

    best = int(np.argmax(per_class_metric))
    other = np.delete(per_class_metric, best)
    return _FitnessRLCW2(
        max_recall=float(per_class_metric[best]),
        mean_other_recall=float(np.mean(other)) if other.size else 0.0,
        size=rs.size,
        best_class=best,
    )


def _pareto_front_rlcw2(
    individuals: list[tuple[_RuleSet2, _FitnessRLCW2]],
) -> list[tuple[_RuleSet2, _FitnessRLCW2]]:
    front: list[tuple[_RuleSet2, _FitnessRLCW2]] = []
    for cand_rs, cand_fit in individuals:
        dominated = False
        kept: list[tuple[_RuleSet2, _FitnessRLCW2]] = []
        for cur_rs, cur_fit in front:
            if cur_fit.dominates(cand_fit):
                dominated = True
                break
            if not cand_fit.dominates(cur_fit):
                kept.append((cur_rs, cur_fit))
        if not dominated:
            kept.append((cand_rs, cand_fit))
            front = kept
    return front


def _tournament_trim(
    individuals: list[tuple[_RuleSet2, _FitnessRLCW2]],
    n_keep: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> list[tuple[_RuleSet2, _FitnessRLCW2]]:
    if len(individuals) <= n_keep:
        return individuals

    selected: list[tuple[_RuleSet2, _FitnessRLCW2]] = []
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


def _compute_weights2(
    rs: _RuleSet2,
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
) -> None:
    class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)

    n_samples = X.shape[0]
    any_fired = np.zeros(n_samples, dtype=bool)

    for rule in rs.rules:
        mask = rs._rule_mask(rule, X)
        any_fired |= mask
        if not mask.any():
            rule.weights = np.ones(n_classes, dtype=float) / n_classes
            continue
        counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
        balanced = counts / class_counts
        total = float(balanced.sum())
        if total > 0:
            rule.weights = balanced / total
        else:
            rule.weights = np.ones(n_classes, dtype=float) / n_classes

    no_fire = ~any_fired
    if no_fire.any():
        counts = np.bincount(y_idx[no_fire], minlength=n_classes).astype(float)
        balanced = counts / class_counts
        total = float(balanced.sum())
        if total > 0:
            rs.default_weights = balanced / total
        else:
            rs.default_weights = np.ones(n_classes, dtype=float) / n_classes
    else:
        rs.default_weights = np.ones(n_classes, dtype=float) / n_classes


class RuleGPClassifier(BaseRuleSetEstimator):
    """RLCW-style GP with atoms/rules/rule sets and native numeric handling."""

    def __init__(
        self,
        f1_averaging: str = "macro",
        objective_mode: ObjectiveMode = "recall",
        max_generations: int = 500,
        stagnation_generations: int = 80,
        early_stopping_metric: str = "consolidated",
        min_max_weight: float = 0.0,
        min_improvement_pct: float = 0.01,
        population_size: int | None = 120,
        n_adaptations_per_gen: int = 20,
        tournament_size: int = 5,
        max_model_size: int | None = None,
        validation_fraction: float = 0.0,
        model_selection: str = "paper",
        zero_mcr_followup_ratio: float = 0.0,
        max_fit_seconds: float | None = None,
        max_rules: int | None = None,
        max_atoms_per_rule: int | None = None,
        min_samples_leaf: int = 1,
        max_thresholds_per_feature: int | None = None,
        atom_space_strategy: NativeAtomSpaceStrategy = "hybrid",
        atom_preselection_strategy: AtomPreselectionStrategy = "none",
        feature_names: list[str] | None = None,
        random_state: int | None = None,
    ):
        self.f1_averaging = f1_averaging
        self.max_generations = max_generations
        self.stagnation_generations = stagnation_generations
        self.early_stopping_metric = early_stopping_metric
        self.min_max_weight = min_max_weight
        self.min_improvement_pct = min_improvement_pct
        self.population_size = population_size
        self.n_adaptations_per_gen = n_adaptations_per_gen
        self.tournament_size = tournament_size
        self.max_model_size = max_model_size
        self.validation_fraction = validation_fraction
        if model_selection not in ("paper", "best_f1", "shortest_zero_train_mcr"):
            raise ValueError(
                "model_selection must be 'paper', 'best_f1', or 'shortest_zero_train_mcr'."
            )
        self.model_selection = model_selection
        self.zero_mcr_followup_ratio = float(zero_mcr_followup_ratio)
        if self.zero_mcr_followup_ratio < 0.0:
            raise ValueError("zero_mcr_followup_ratio must be >= 0.0.")
        self.max_fit_seconds = max_fit_seconds
        self.max_rules = max_rules
        self.max_atoms_per_rule = max_atoms_per_rule
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds_per_feature = max_thresholds_per_feature
        self.atom_space_strategy = atom_space_strategy
        if atom_preselection_strategy not in (
            "none",
            "logicgp_singleton",
            "logicgp_binned_sets",
        ):
            raise ValueError(
                "atom_preselection_strategy must be 'none', "
                "'logicgp_singleton', or 'logicgp_binned_sets'."
            )
        self.atom_preselection_strategy = atom_preselection_strategy
        self.feature_names = feature_names
        self.random_state = random_state
        if objective_mode not in ("recall", "f1"):
            raise ValueError("objective_mode must be 'recall' or 'f1'.")
        self.objective_mode = objective_mode

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
        if self.feature_names is not None:
            if len(self.feature_names) != self.n_features_in_:
                raise ValueError(
                    f"feature_names has length {len(self.feature_names)}; expected {self.n_features_in_}."
                )
            self.feature_names_in_ = np.asarray([str(name) for name in self.feature_names], dtype=object)
        else:
            self.feature_names_in_ = np.asarray(
                [f"f{i}" for i in range(self.n_features_in_)], dtype=object
            )
        self.classes_ = unique_labels(y_valid)
        n_classes = len(self.classes_)

        self._rng_ = np.random.default_rng(self.random_state)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y_valid], dtype=int)

        X_train, y_train = X_valid, y_idx
        X_val = None
        y_val = None
        if (
            self.validation_fraction > 0
            and X_valid.shape[0] >= 30
            and len(np.unique(y_idx)) >= 2
        ):
            try:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=self.validation_fraction,
                    random_state=(self.random_state if self.random_state is not None else 0),
                )
                train_idx, val_idx = next(sss.split(X_valid, y_idx))
                X_train = X_valid[train_idx]
                y_train = y_idx[train_idx]
                X_val = X_valid[val_idx]
                y_val = y_idx[val_idx]
            except ValueError:
                pass

        specs = _build_feature_specs(
            X_train,
            self.max_thresholds_per_feature,
            atom_space_strategy=self.atom_space_strategy,
        )
        atom_pool = self._build_atom_pool(specs, X_train, y_train, n_classes)
        all_atoms = [a for atoms in atom_pool.values() for a in atoms]
        if not all_atoms:
            all_atoms = self._fallback_atoms(specs)
        if not all_atoms:
            raise ValueError("No atoms can be generated from the given data.")

        population = self._init_population(
            all_atoms=all_atoms,
            specs=specs,
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes,
        )
        best_rs = self._run_gp(
            population=population,
            all_atoms=all_atoms,
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes,
            X_val=X_val,
            y_val=y_val,
        )

        if X_val is not None and y_val is not None:
            _compute_weights2(best_rs, X_valid, y_idx, n_classes)

        self.ruleset_ = _to_ruleset_rulegp(self, best_rs, n_classes)
        self.ruleset_.validate()
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr: np.ndarray = np.asarray(check_array(X, dtype=None))
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}."
            )
        return predict_from_ruleset(self.ruleset_, X_arr)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr: np.ndarray = np.asarray(check_array(X, dtype=None))
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}."
            )
        return predict_proba_from_ruleset(self.ruleset_, X_arr)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

    def _build_atom_pool(
        self,
        specs: list[dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
    ) -> dict[int, list[_AtomGene2]]:
        class_counts = np.bincount(y, minlength=n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)

        pool: dict[int, list[_AtomGene2]] = {}
        for spec in specs:
            fi = int(spec["idx"])
            candidates: list[_AtomGene2] = []

            if spec["kind"] in ("num", "both"):
                for thr in spec.get("thresholds", []):
                    candidates.append(_AtomGene2(fi, "<=", float(thr)))
                    candidates.append(_AtomGene2(fi, ">", float(thr)))
                for lo, hi in spec.get("intervals", []):
                    candidates.append(_AtomGene2(fi, "between", [float(lo), float(hi)]))

            if spec["kind"] in ("cat", "both"):
                cats = list(spec.get("categories", []))
                for cat in cats:
                    candidates.append(_AtomGene2(fi, "==", cat))
                # logicGP-like subset space: all non-trivial subsets for moderate cardinalities.
                if len(cats) >= 3:
                    k = len(cats)
                    if k <= 12:
                        for r in range(2, k):
                            for subset in combinations(cats, r):
                                candidates.append(_AtomGene2(fi, "in", list(subset)))
                    else:
                        candidates.append(_AtomGene2(fi, "in", sorted(cats[:2])))
                        candidates.append(_AtomGene2(fi, "in", sorted(cats[-2:])))
                        mid = k // 2
                        candidates.append(_AtomGene2(fi, "in", sorted(cats[mid - 1:mid + 1])))

            if self.atom_preselection_strategy == "logicgp_singleton":
                candidates = [a for a in candidates if a.op == "=="]
            elif self.atom_preselection_strategy == "logicgp_binned_sets":
                # Keep logicGP-like set atoms and add contiguous interval atoms for
                # ordered bin categories (e.g. bin indices 0..k-1).
                # For purely continuous features that have no categories, fall back
                # to the full numeric-threshold candidates so the model stays usable.
                binned = [a for a in candidates if a.op in ("==", "in")]

                cats = list(spec.get("categories", []))
                if cats:
                    numeric_cats: list[float] = []
                    for c in cats:
                        try:
                            numeric_cats.append(float(c))
                        except (TypeError, ValueError):
                            numeric_cats = []
                            break

                    if numeric_cats:
                        ordered = sorted(set(numeric_cats))
                        if len(ordered) >= 3:
                            for i in range(len(ordered) - 1):
                                for j in range(i + 1, len(ordered)):
                                    binned.append(
                                        _AtomGene2(
                                            fi,
                                            "between",
                                            [float(ordered[i]), float(ordered[j])],
                                        )
                                    )

                # If no categorical/binned atoms exist for this feature (e.g. purely
                # numeric), fall back to the original threshold candidates.
                candidates = binned if binned else candidates

            scored: list[tuple[float, _AtomGene2]] = []
            for atom in candidates:
                mask = self._atom_mask(atom, X)
                support = int(mask.sum())
                if support < self.min_samples_leaf:
                    continue
                counts = np.bincount(y[mask], minlength=n_classes).astype(float)
                balanced = counts / class_counts
                total = float(balanced.sum())
                if total <= 0:
                    continue
                max_weight = float((balanced / total).max())
                if self.min_max_weight > 0 and max_weight <= self.min_max_weight:
                    continue
                scored.append((max_weight, atom))

            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                unique: dict[tuple[int, str, str], _AtomGene2] = {}
                for _, atom in scored:
                    unique[(atom.feature_idx, atom.op, str(atom.value))] = atom
                pool[fi] = list(unique.values())

        return pool

    def _fallback_atoms(self, specs: list[dict[str, Any]]) -> list[_AtomGene2]:
        atoms: list[_AtomGene2] = []
        for spec in specs:
            fi = int(spec["idx"])
            if self.atom_preselection_strategy == "logicgp_singleton":
                cats = list(spec.get("categories", []))
                if cats:
                    atoms.append(_AtomGene2(fi, "==", cats[0]))
                continue
            if self.atom_preselection_strategy == "logicgp_binned_sets":
                cats = list(spec.get("categories", []))
                if cats:
                    atoms.append(_AtomGene2(fi, "==", cats[0]))
                    if len(cats) >= 2:
                        try:
                            nums = sorted(set(float(c) for c in cats))
                            atoms.append(_AtomGene2(fi, "between", [nums[0], nums[-1]]))
                        except (TypeError, ValueError):
                            pass
                elif spec["kind"] in ("num", "both") and spec.get("thresholds"):
                    # No categories: fall back to a numeric threshold atom.
                    atoms.append(_AtomGene2(fi, "<=", float(spec["thresholds"][0])))
                continue
            if spec["kind"] in ("num", "both") and spec.get("thresholds"):
                thr = spec["thresholds"][0]
                atoms.append(_AtomGene2(fi, "<=", float(thr)))
            elif spec.get("categories"):
                atoms.append(_AtomGene2(fi, "==", spec["categories"][0]))
        return atoms

    def _init_population(
        self,
        all_atoms: list[_AtomGene2],
        specs: list[dict[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: int,
    ) -> list[_RuleSet2]:
        unif = np.ones(n_classes, dtype=float) / n_classes
        pop: list[_RuleSet2] = []
        for atom in all_atoms:
            pop.append(
                _RuleSet2(
                    rules=[_Rule2(atoms=[atom], weights=unif.copy())],
                    default_weights=unif.copy(),
                )
            )

        # logicGP-like class-discriminative seeds to avoid over-bias towards tiny models.
        pop.extend(self._seed_class_discriminative(all_atoms, X_train, y_train, n_classes))
        return pop

    def _seed_class_discriminative(
        self,
        all_atoms: list[_AtomGene2],
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
    ) -> list[_RuleSet2]:
        seeds: list[_RuleSet2] = []
        unif = np.ones(n_classes, dtype=float) / n_classes

        atom_masks: list[np.ndarray] = [self._atom_mask(atom, X) for atom in all_atoms]
        best_per_class: list[list[tuple[int, float]]] = [[] for _ in range(n_classes)]

        for ai, mask in enumerate(atom_masks):
            n_fire = int(mask.sum())
            if n_fire < self.min_samples_leaf:
                continue
            for c in range(n_classes):
                class_mask = y_idx == c
                n_class = int(class_mask.sum())
                if n_class == 0:
                    continue
                tp = int((mask & class_mask).sum())
                purity = tp / n_fire
                coverage = tp / n_class
                best_per_class[c].append((ai, purity * coverage))

        top_k = 5
        class_top_atoms: list[list[int]] = []
        for c in range(n_classes):
            best_per_class[c].sort(key=lambda x: x[1], reverse=True)
            class_top_atoms.append([ai for ai, _ in best_per_class[c][:top_k]])

        for c in range(n_classes):
            top = class_top_atoms[c]
            if len(top) < 2:
                continue
            for i in range(min(3, len(top))):
                for j in range(i + 1, min(4, len(top))):
                    atom_i = all_atoms[top[i]]
                    atom_j = all_atoms[top[j]]
                    if atom_i.feature_idx != atom_j.feature_idx:
                        seeds.append(
                            _RuleSet2(
                                rules=[_Rule2(atoms=[atom_i, atom_j], weights=unif.copy())],
                                default_weights=unif.copy(),
                            )
                        )
                        break

        if n_classes >= 2 and all(len(top) > 0 for top in class_top_atoms):
            multi_rules = [
                _Rule2(atoms=[all_atoms[class_top_atoms[c][0]]], weights=unif.copy())
                for c in range(n_classes)
            ]
            seeds.append(_RuleSet2(rules=multi_rules, default_weights=unif.copy()))

            if all(len(top) >= 2 for top in class_top_atoms):
                multi_rules_2 = [
                    _Rule2(atoms=[all_atoms[class_top_atoms[c][1]]], weights=unif.copy())
                    for c in range(n_classes)
                ]
                seeds.append(_RuleSet2(rules=multi_rules_2, default_weights=unif.copy()))

        return seeds

    def _atom_mask(self, atom: _AtomGene2, X: np.ndarray) -> np.ndarray:
        col = X[:, atom.feature_idx]
        if atom.op == "<=":
            return np.asarray(col, dtype=float) <= float(atom.value)
        if atom.op == ">":
            return np.asarray(col, dtype=float) > float(atom.value)
        if atom.op == "between":
            lo, hi = atom.value
            c = np.asarray(col, dtype=float)
            return (c >= float(lo)) & (c <= float(hi))
        if atom.op == "==":
            try:
                return (np.asarray(col, dtype=float) == float(atom.value)).astype(bool)
            except (TypeError, ValueError):
                return (np.asarray(col, dtype=object) == atom.value).astype(bool)
        if atom.op == "in":
            return np.isin(np.asarray(col, dtype=object), list(atom.value)).astype(bool)
        return np.ones(X.shape[0], dtype=bool)

    def _select_two_parents(
        self, evaluated: list[tuple[_RuleSet2, _FitnessRLCW2]]
    ) -> tuple[_RuleSet2, _RuleSet2]:
        idx_a, idx_b = self._rng_.choice(len(evaluated), size=2, replace=False)
        rs_a, fit_a = evaluated[idx_a]
        rs_b, fit_b = evaluated[idx_b]
        if fit_a.dominates(fit_b):
            return rs_a, rs_b
        if fit_b.dominates(fit_a):
            return rs_b, rs_a
        return rs_a, rs_b

    def _select_parent(self, evaluated: list[tuple[_RuleSet2, _FitnessRLCW2]]) -> _RuleSet2:
        idx = int(self._rng_.integers(0, len(evaluated)))
        return evaluated[idx][0]

    def _crossover(self, a: _RuleSet2, b: _RuleSet2) -> _RuleSet2:
        offspring = b.clone()
        if not a.rules or not offspring.rules:
            return offspring
        src = a.rules[int(self._rng_.integers(0, len(a.rules)))].clone()
        tgt_idx = int(self._rng_.integers(0, len(offspring.rules)))
        offspring.rules[tgt_idx] = src
        return offspring

    def _mut_insert_atom(self, rs: _RuleSet2, all_atoms: list[_AtomGene2]) -> _RuleSet2:
        if not all_atoms or not rs.rules:
            return rs
        out = rs.clone()
        ri = int(self._rng_.integers(0, len(out.rules)))
        if self.max_atoms_per_rule is not None and len(out.rules[ri].atoms) >= self.max_atoms_per_rule:
            return out
        atom = all_atoms[int(self._rng_.integers(0, len(all_atoms)))]
        out.rules[ri].atoms.append(atom)
        return out

    def _mut_delete_atom(self, rs: _RuleSet2) -> _RuleSet2:
        out = rs.clone()
        candidates = [i for i, r in enumerate(out.rules) if len(r.atoms) > 1]
        if not candidates:
            return out
        ri = candidates[int(self._rng_.integers(0, len(candidates)))]
        ai = int(self._rng_.integers(0, len(out.rules[ri].atoms)))
        out.rules[ri].atoms.pop(ai)
        return out

    def _mut_replace_atom(self, rs: _RuleSet2, all_atoms: list[_AtomGene2]) -> _RuleSet2:
        if not all_atoms or not rs.rules:
            return rs
        out = rs.clone()
        ri = int(self._rng_.integers(0, len(out.rules)))
        if not out.rules[ri].atoms:
            return out
        ai = int(self._rng_.integers(0, len(out.rules[ri].atoms)))
        out.rules[ri].atoms[ai] = all_atoms[int(self._rng_.integers(0, len(all_atoms)))]
        return out

    def _mut_insert_rule(self, rs: _RuleSet2, all_atoms: list[_AtomGene2], n_classes: int) -> _RuleSet2:
        if not all_atoms:
            return rs
        out = rs.clone()
        if self.max_rules is not None and len(out.rules) >= self.max_rules:
            return out
        atom = all_atoms[int(self._rng_.integers(0, len(all_atoms)))]
        out.rules.append(
            _Rule2(
                atoms=[atom],
                weights=np.ones(n_classes, dtype=float) / n_classes,
            )
        )
        return out

    def _mut_delete_rule(self, rs: _RuleSet2) -> _RuleSet2:
        if len(rs.rules) <= 1:
            return rs
        out = rs.clone()
        ri = int(self._rng_.integers(0, len(out.rules)))
        out.rules.pop(ri)
        return out

    def _run_gp(
        self,
        population: list[_RuleSet2],
        all_atoms: list[_AtomGene2],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: int,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> _RuleSet2:
        if self.f1_averaging.lower().strip() not in ("micro", "macro"):
            raise ValueError("f1_averaging must be 'micro' or 'macro'.")
        if self.early_stopping_metric not in ("f1", "consolidated"):
            raise ValueError("early_stopping_metric must be 'f1' or 'consolidated'.")
        if self.objective_mode not in ("recall", "f1"):
            raise ValueError("objective_mode must be 'recall' or 'f1'.")

        has_val = X_val is not None and y_val is not None
        eval_X = X_val if has_val else X_train
        eval_y = y_val if has_val else y_train
        labels = list(range(n_classes))

        for rs in population:
            _compute_weights2(rs, X_train, y_train, n_classes)

        evaluated = [
            (
                rs,
                _evaluate_fitness_rlcw2(
                    rs,
                    X_train,
                    y_train,
                    n_classes,
                    self.objective_mode,
                ),
            )
            for rs in population
        ]
        evaluated = _pareto_front_rlcw2(evaluated)

        if self.population_size is not None and len(evaluated) > self.population_size:
            evaluated = _tournament_trim(evaluated, self.population_size, self.tournament_size, self._rng_)

        if self.early_stopping_metric == "f1":
            best_signal = max(
                float(_f1_score(eval_y, rs.predict_classes(eval_X), average=self.f1_averaging, labels=labels))
                for rs, _ in evaluated
            )
        else:
            best_signal = max(fit.consolidated for _, fit in evaluated)
        stagnation = 0
        all_candidates: list[tuple[_RuleSet2, _FitnessRLCW2, float, float]] = []

        for rs, fit in evaluated:
            pred = rs.predict_classes(eval_X)
            f1 = float(_f1_score(eval_y, pred, average=self.f1_averaging, labels=labels))
            train_preds = rs.predict_classes(X_train)
            train_mcr = float(np.mean(train_preds != y_train))
            all_candidates.append((rs, fit, f1, train_mcr))

        elite_rs = max(all_candidates, key=lambda x: x[2])[0].clone()
        elite_fit = _evaluate_fitness_rlcw2(
            elite_rs,
            X_train,
            y_train,
            n_classes,
            self.objective_mode,
        )
        elite_f1 = max(c[2] for c in all_candidates)

        mut_ops = [
            lambda p: self._mut_insert_atom(p, all_atoms),
            self._mut_delete_atom,
            lambda p: self._mut_replace_atom(p, all_atoms),
            lambda p: self._mut_insert_rule(p, all_atoms, n_classes),
            self._mut_delete_rule,
        ]

        n_adapt = max(1, self.n_adaptations_per_gen)
        t0 = time.monotonic()
        zero_mcr_target_generation: int | None = None

        for generation_idx in range(self.max_generations):
            if self.max_fit_seconds is not None and (time.monotonic() - t0) >= self.max_fit_seconds:
                break

            new_rs: list[_RuleSet2] = []
            for i in range(n_adapt):
                if i == 0 and len(evaluated) >= 2:
                    p_a, p_b = self._select_two_parents(evaluated)
                    child = self._crossover(p_a, p_b)
                else:
                    parent = self._select_parent(evaluated)
                    child = mut_ops[(i - 1) % len(mut_ops)](parent)

                if self.max_model_size is not None and child.size > self.max_model_size:
                    continue
                if self.max_rules is not None and len(child.rules) > self.max_rules:
                    child.rules = child.rules[: self.max_rules]
                for rule in child.rules:
                    if self.max_atoms_per_rule is not None and len(rule.atoms) > self.max_atoms_per_rule:
                        rule.atoms = rule.atoms[: self.max_atoms_per_rule]
                if not child.rules:
                    continue
                new_rs.append(child)

            new_eval: list[tuple[_RuleSet2, _FitnessRLCW2]] = []
            for rs in new_rs:
                _compute_weights2(rs, X_train, y_train, n_classes)
                fit = _evaluate_fitness_rlcw2(
                    rs,
                    X_train,
                    y_train,
                    n_classes,
                    self.objective_mode,
                )
                new_eval.append((rs, fit))

            combined = evaluated + new_eval
            evaluated = _pareto_front_rlcw2(combined)
            if self.population_size is not None and len(evaluated) > self.population_size:
                evaluated = _tournament_trim(evaluated, self.population_size, self.tournament_size, self._rng_)

            if elite_rs is not None:
                ids = {id(rs) for rs, _ in evaluated}
                if id(elite_rs) not in ids:
                    evaluated.append((elite_rs, elite_fit))

            for rs, fit in new_eval:
                pred = rs.predict_classes(eval_X)
                f1 = float(_f1_score(eval_y, pred, average=self.f1_averaging, labels=labels))
                train_preds = rs.predict_classes(X_train)
                train_mcr = float(np.mean(train_preds != y_train))
                all_candidates.append((rs, fit, f1, train_mcr))
                if f1 > elite_f1:
                    elite_f1 = f1
                    elite_rs = rs.clone()
                    elite_fit = fit

            if (
                self.model_selection == "shortest_zero_train_mcr"
                and self.zero_mcr_followup_ratio > 0.0
                and zero_mcr_target_generation is None
            ):
                has_zero_mcr = any(float(train_mcr) <= 1e-12 for _rs, _fit, _f1, train_mcr in all_candidates)
                if has_zero_mcr:
                    generations_until_zero = generation_idx + 1
                    extra_generations = int(np.ceil(generations_until_zero * self.zero_mcr_followup_ratio))
                    zero_mcr_target_generation = min(
                        self.max_generations,
                        generations_until_zero + extra_generations,
                    )

            if self.early_stopping_metric == "f1":
                current_signal = max(
                    float(_f1_score(eval_y, rs.predict_classes(eval_X), average=self.f1_averaging, labels=labels))
                    for rs, _ in evaluated
                )
            else:
                current_signal = max(fit.consolidated for _, fit in evaluated)

            if current_signal > best_signal + 1e-10:
                best_signal = current_signal
                stagnation = 0
            else:
                stagnation += 1

            # If a perfect-fit candidate appears, optionally keep exploring for
            # a fixed follow-up budget to find a shorter zero-MCR model.
            if zero_mcr_target_generation is not None:
                if (generation_idx + 1) >= zero_mcr_target_generation:
                    break
                continue

            if stagnation >= self.stagnation_generations:
                break

        if self.model_selection == "best_f1":
            best = _select_model_best_f1(all_candidates)
        elif self.model_selection == "shortest_zero_train_mcr":
            best = _select_model_shortest_zero_train_mcr(
                all_candidates,
                min_improvement=self.min_improvement_pct,
            )
        else:
            best = _select_model_paper(
                [(rs, fit, f1) for rs, fit, f1, _train_mcr in all_candidates]
            )
        return best


def _select_model_best_f1(
    candidates: list[tuple[_RuleSet2, _FitnessRLCW2, float, float]],
) -> _RuleSet2:
    if not candidates:
        raise ValueError("No rule-set candidates available.")
    best = max(candidates, key=lambda x: (x[2], -x[1].size))
    return best[0]


def _select_model_shortest_zero_train_mcr(
    candidates: list[tuple[_RuleSet2, _FitnessRLCW2, float, float]],
    min_improvement: float = 0.01,
) -> _RuleSet2:
    # Select the shortest candidate with zero training MCR if available.
    if not candidates:
        raise ValueError("No rule-set candidates available.")

    zero_mcr = [c for c in candidates if float(c[3]) <= 1e-12]
    if zero_mcr:
        best = min(zero_mcr, key=lambda x: (x[1].size, -x[2], -x[1].consolidated))
        return best[0]

    reduced = [(rs, fit, f1) for rs, fit, f1, _train_mcr in candidates]
    return _select_model_paper(reduced, min_improvement=min_improvement)


def _select_model_paper(
    candidates: list[tuple[_RuleSet2, _FitnessRLCW2, float]],
    min_improvement: float = 0.01,
) -> _RuleSet2:
    if not candidates:
        raise ValueError("No rule-set candidates available.")

    best_per_size: dict[int, tuple[_RuleSet2, _FitnessRLCW2, float]] = {}
    for rs, fit, f1 in candidates:
        s = fit.size
        if s not in best_per_size or f1 > best_per_size[s][2]:
            best_per_size[s] = (rs, fit, f1)

    sorted_by_size = sorted(best_per_size.values(), key=lambda x: x[1].size)
    filtered: list[tuple[_RuleSet2, _FitnessRLCW2, float]] = []
    for rs, fit, f1 in sorted_by_size:
        smaller_max = max((cf1 for _, cfit, cf1 in filtered if cfit.size < fit.size), default=0.0)
        if not filtered or f1 >= smaller_max * (1.0 + min_improvement):
            filtered.append((rs, fit, f1))

    if not filtered:
        filtered = sorted_by_size

    return max(filtered, key=lambda x: (x[2], x[1].size))[0]

def _to_ruleset_rulegp(classifier: RuleGPClassifier, rs: _RuleSet2, n_classes: int) -> ScoredRuleSet:
    def _serialize_atom_value(atom: _AtomGene2) -> object:
        if atom.op == "between":
            return [float(atom.value[0]), float(atom.value[1])]
        if atom.op == "in":
            return list(atom.value)
        if isinstance(atom.value, (int, float, np.integer, np.floating)):
            return float(atom.value)
        return atom.value

    rules: list[Rule] = [
        Rule(
            atoms=[],
            scores=rs.default_weights.tolist(),
            rule_id="rulegp_default",
            metadata={"source": "rulegp", "kind": "default"},
        )
    ]

    for i, rule in enumerate(rs.rules):
        atoms = [
            Atom(
                feature=(
                    str(classifier.feature_names_in_[a.feature_idx])
                    if a.feature_idx < len(classifier.feature_names_in_)
                    else f"f{a.feature_idx}"
                ),
                op=a.op,
                value=_serialize_atom_value(a),
            )
            for a in rule.atoms
        ]
        rules.append(
            Rule(
                atoms=atoms,
                scores=rule.weights.tolist(),
                rule_id=f"rulegp_rule_{i}",
                metadata={"source": "rulegp", "kind": "rule"},
            )
        )

    return ScoredRuleSet(
        class_labels=classifier.classes_.tolist(),
        feature_names=classifier.feature_names_in_.tolist(),
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        rules=rules,
        metadata={
            "source": "rulegp",
            "f1_averaging": classifier.f1_averaging,
            "objective_mode": classifier.objective_mode,
            "max_generations": classifier.max_generations,
            "population_size": classifier.population_size,
            "n_adaptations_per_gen": classifier.n_adaptations_per_gen,
            "early_stopping_metric": classifier.early_stopping_metric,
            "model_selection": classifier.model_selection,
            "zero_mcr_followup_ratio": classifier.zero_mcr_followup_ratio,
            "max_rules": classifier.max_rules,
            "max_atoms_per_rule": classifier.max_atoms_per_rule,
            "max_model_size": classifier.max_model_size,
            "min_max_weight": classifier.min_max_weight,
            "atom_space_strategy": classifier.atom_space_strategy,
            "atom_preselection_strategy": classifier.atom_preselection_strategy,
        },
    )
