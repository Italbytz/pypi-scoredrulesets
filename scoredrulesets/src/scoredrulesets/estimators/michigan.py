from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator


@dataclass
class _MichiganRule:
    atoms: list[Atom]
    class_idx: int
    fitness: float
    numerosity: int = 1
    experience: int = 0
    match_count: int = 0


class MichiganRuleSetClassifier(BaseRuleSetEstimator):
    """Lightweight Michigan-style LCS baseline with online rule updates."""

    def __init__(
        self,
        aggregation: str = "argmax_sum",
        temperature: float = 1.0,
        include_default_rule: bool = True,
        enable_categorical_rules: bool = True,
        population_size: int = 60,
        epochs: int = 8,
        max_atoms_per_rule: int = 2,
        min_samples_leaf: int = 3,
        learning_rate: float = 0.08,
        mutation_rate: float = 0.08,
        covering_probability: float = 0.12,
        min_rule_fitness: float = 0.02,
        max_final_rules: int | None = None,
        random_state: int | None = None,
    ):
        self.aggregation = aggregation
        self.temperature = temperature
        self.include_default_rule = include_default_rule
        self.enable_categorical_rules = enable_categorical_rules
        self.population_size = population_size
        self.epochs = epochs
        self.max_atoms_per_rule = max_atoms_per_rule
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.covering_probability = covering_probability
        self.min_rule_fitness = min_rule_fitness
        self.max_final_rules = max_final_rules
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = np.asarray([f"f{i}" for i in range(self.n_features_in_)], dtype=object)
        self.classes_ = unique_labels(y_valid)
        self._rng_ = np.random.default_rng(self.random_state)

        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[v] for v in y_valid], dtype=int)
        n_classes = len(self.classes_)

        self._feature_specs_ = self._build_feature_specs(X_valid)
        self._population_: list[_MichiganRule] = []

        # Initialize with one covering rule per class for a stable warm start.
        for class_idx in range(n_classes):
            class_rows = np.where(y_idx == class_idx)[0]
            if class_rows.size == 0:
                continue
            seed_idx = int(self._rng_.choice(class_rows))
            self._population_.append(
                self._covering_rule(X_valid[seed_idx], class_idx)
            )

        for _ in range(max(1, int(self.epochs))):
            order = self._rng_.permutation(X_valid.shape[0])
            for row_idx in order:
                x_row = X_valid[row_idx]
                y_class = int(y_idx[row_idx])

                matched = [rule for rule in self._population_ if self._rule_matches(rule, x_row)]
                need_cover = not matched or (self._rng_.random() < float(self.covering_probability))
                if need_cover:
                    self._population_.append(self._covering_rule(x_row, y_class))
                    matched = [rule for rule in self._population_ if self._rule_matches(rule, x_row)]

                for rule in matched:
                    rule.experience += 1
                    rule.match_count += 1
                    target = 1.0 if rule.class_idx == y_class else 0.0
                    rule.fitness = (1.0 - self.learning_rate) * rule.fitness + self.learning_rate * target

                    if self._rng_.random() < float(self.mutation_rate):
                        self._mutate_rule(rule)

                self._trim_population()

        prior_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        default_scores = self._distribution_to_scores(prior_counts)

        kept_rules = [
            rule
            for rule in self._population_
            if rule.fitness >= float(self.min_rule_fitness) and rule.match_count > 0
        ]
        kept_rules.sort(
            key=lambda r: (r.fitness * np.log1p(r.match_count), r.match_count, r.experience),
            reverse=True,
        )

        max_final = int(self.max_final_rules) if self.max_final_rules is not None else int(self.population_size)
        max_final = max(1, max_final)

        final_rules: list[Rule] = []
        for idx, rule in enumerate(kept_rules[:max_final]):
            score_strength = float(max(rule.fitness, 1e-6) * np.log1p(rule.match_count + rule.numerosity))
            scores = [0.0] * n_classes
            scores[rule.class_idx] = score_strength
            final_rules.append(
                Rule(
                    atoms=list(rule.atoms),
                    scores=scores,
                    rule_id=f"michigan_rule_{idx}",
                    metadata={
                        "source": "michigan_lcs",
                        "fitness": float(rule.fitness),
                        "experience": int(rule.experience),
                        "match_count": int(rule.match_count),
                        "numerosity": int(rule.numerosity),
                    },
                )
            )

        if self.include_default_rule:
            final_rules.append(
                Rule(
                    atoms=[],
                    scores=default_scores,
                    rule_id="michigan_default_prior",
                    metadata={"source": "michigan_lcs", "kind": "class_prior"},
                )
            )

        self.ruleset_ = ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=self.feature_names_in_.tolist(),
            aggregation=AggregationSpec(type=self.aggregation, temperature=self.temperature),
            rules=final_rules,
            metadata={
                "source": "michigan_lcs",
                "model_type": "michigan_online_rule_population",
                "population_size_final": int(len(final_rules)),
                "selected_rule_count": int(len([r for r in final_rules if r.rule_id != "michigan_default_prior"])),
                "selected_atom_count": int(sum(len(r.atoms) for r in final_rules if r.rule_id != "michigan_default_prior")),
                "epochs": int(self.epochs),
                "covering_probability": float(self.covering_probability),
                "mutation_rate": float(self.mutation_rate),
                "learning_rate": float(self.learning_rate),
                "max_final_rules": None if self.max_final_rules is None else int(self.max_final_rules),
            },
        )
        self.ruleset_.validate()
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = np.asarray(check_array(X, dtype=None))
        return predict_from_ruleset(self.ruleset_, X_valid)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = np.asarray(check_array(X, dtype=None))
        return predict_proba_from_ruleset(self.ruleset_, X_valid)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_

    def _build_feature_specs(self, X: np.ndarray) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for feature_idx in range(X.shape[1]):
            col = X[:, feature_idx]
            arr = np.asarray(col)
            if np.issubdtype(arr.dtype, np.number):
                values = np.unique(arr.astype(float))
                if values.size <= 1:
                    continue
                if values.size <= 20:
                    thresholds = ((values[:-1] + values[1:]) / 2.0).tolist()
                else:
                    q = np.unique(np.quantile(values, [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]))
                    thresholds = q.astype(float).tolist()
                specs.append(
                    {
                        "idx": feature_idx,
                        "kind": "num",
                        "thresholds": thresholds,
                    }
                )
            else:
                if not self.enable_categorical_rules:
                    continue
                categories = np.unique(np.asarray(col, dtype=object)).tolist()
                if len(categories) <= 1:
                    continue
                specs.append(
                    {
                        "idx": feature_idx,
                        "kind": "cat",
                        "categories": categories,
                    }
                )
        return specs

    def _covering_rule(self, row: np.ndarray, class_idx: int) -> _MichiganRule:
        atoms: list[Atom] = []
        if not self._feature_specs_:
            return _MichiganRule(atoms=[], class_idx=class_idx, fitness=0.55)

        n_atoms = int(self._rng_.integers(1, max(2, int(self.max_atoms_per_rule) + 1)))
        chosen_specs = self._rng_.choice(
            self._feature_specs_,
            size=min(n_atoms, len(self._feature_specs_)),
            replace=False,
        )

        for spec in np.atleast_1d(chosen_specs):
            spec = dict(spec)
            feature_idx = int(spec["idx"])
            feature_name = str(self.feature_names_in_[feature_idx])
            value = row[feature_idx]

            if spec["kind"] == "num":
                thresholds = list(spec.get("thresholds", []))
                if not thresholds:
                    continue
                threshold = float(self._nearest_threshold(float(value), thresholds))
                op = "<=" if float(value) <= threshold else ">"
                atoms.append(Atom(feature=feature_name, op=op, value=threshold))
            else:
                atoms.append(Atom(feature=feature_name, op="==", value=value))

        return _MichiganRule(atoms=atoms, class_idx=class_idx, fitness=0.6)

    def _mutate_rule(self, rule: _MichiganRule) -> None:
        if not self._feature_specs_:
            return

        if not rule.atoms or self._rng_.random() < 0.35:
            feature_spec = dict(self._feature_specs_[int(self._rng_.integers(0, len(self._feature_specs_)))])
            feature_idx = int(feature_spec["idx"])
            feature_name = str(self.feature_names_in_[feature_idx])
            thresholds = list(feature_spec.get("thresholds", []))
            if feature_spec["kind"] == "num" and thresholds:
                threshold = float(thresholds[int(self._rng_.integers(0, len(thresholds)))])
                op = "<=" if self._rng_.random() < 0.5 else ">"
                atom = Atom(feature=feature_name, op=op, value=threshold)
            else:
                categories = list(feature_spec.get("categories", []))
                if not categories:
                    return
                value = categories[int(self._rng_.integers(0, len(categories)))]
                atom = Atom(feature=feature_name, op="==", value=value)
            if len(rule.atoms) < max(1, int(self.max_atoms_per_rule)):
                rule.atoms.append(atom)
            else:
                replace_idx = int(self._rng_.integers(0, len(rule.atoms)))
                rule.atoms[replace_idx] = atom
            return

        replace_idx = int(self._rng_.integers(0, len(rule.atoms)))
        old_atom = rule.atoms[replace_idx]
        feature_name = str(old_atom.feature)
        feature_idx = int(feature_name[1:]) if feature_name.startswith("f") else None
        if feature_idx is None:
            return
        feature_spec = next(
            (spec for spec in self._feature_specs_ if int(spec.get("idx", -1)) == feature_idx),
            None,
        )
        if feature_spec is None:
            return

        feature_spec = dict(feature_spec)
        thresholds = list(feature_spec.get("thresholds", []))
        if feature_spec["kind"] == "num" and thresholds:
            threshold = float(thresholds[int(self._rng_.integers(0, len(thresholds)))])
            op = "<=" if self._rng_.random() < 0.5 else ">"
            rule.atoms[replace_idx] = Atom(feature=feature_name, op=op, value=threshold)
        else:
            categories = list(feature_spec.get("categories", []))
            if categories:
                value = categories[int(self._rng_.integers(0, len(categories)))]
                rule.atoms[replace_idx] = Atom(feature=feature_name, op="==", value=value)

    def _trim_population(self) -> None:
        max_pop = max(4, int(self.population_size))
        if len(self._population_) <= max_pop:
            return
        self._population_.sort(
            key=lambda r: (r.fitness, r.match_count, r.experience),
            reverse=True,
        )
        del self._population_[max_pop:]

    def _rule_matches(self, rule: _MichiganRule, row: np.ndarray) -> bool:
        for atom in rule.atoms:
            idx = int(str(atom.feature)[1:])
            value = row[idx]
            if atom.op == "<=":
                if float(value) > float(atom.value):
                    return False
            elif atom.op == ">":
                if float(value) <= float(atom.value):
                    return False
            elif atom.op == "==":
                if value != atom.value:
                    return False
            else:
                return False
        return True

    def _distribution_to_scores(self, counts: np.ndarray) -> list[float]:
        probs = counts / max(float(np.sum(counts)), 1.0)
        if self.aggregation == "softmax_sum":
            return np.log(np.maximum(probs, 1e-12)).tolist()
        return probs.tolist()

    @staticmethod
    def _nearest_threshold(value: float, thresholds: list[float]) -> float:
        arr = np.asarray(thresholds, dtype=float)
        return float(arr[np.argmin(np.abs(arr - value))])


