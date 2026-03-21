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
class _AtomGene:
    feature_idx: int
    op: str
    value: object


@dataclass
class _RuleGene:
    atoms: list[_AtomGene]


class GeneticScoredRuleSetClassifier(BaseRuleSetEstimator):
    """Einfacher genetischer Learner fuer Scored Rule Sets."""

    def __init__(
        self,
        aggregation: str = "softmax_sum",
        temperature: float = 1.0,
        score_mode: str = "auto",
        include_default_rule: bool = True,
        population_size: int = 40,
        generations: int = 20,
        max_rules: int = 6,
        max_atoms_per_rule: int = 3,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        tournament_size: int = 3,
        min_samples_leaf: int = 3,
        complexity_penalty: float = 0.01,
        random_state: int | None = None,
    ):
        self.aggregation = aggregation
        self.temperature = temperature
        self.score_mode = score_mode
        self.include_default_rule = include_default_rule
        self.population_size = population_size
        self.generations = generations
        self.max_rules = max_rules
        self.max_atoms_per_rule = max_atoms_per_rule
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.min_samples_leaf = min_samples_leaf
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self._resolved_score_mode_ = self._resolve_score_mode()
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = np.asarray([f"f{i}" for i in range(self.n_features_in_)], dtype=object)
        self.classes_ = unique_labels(y_valid)

        self._rng_ = np.random.default_rng(self.random_state)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[val] for val in y_valid], dtype=int)
        n_classes = len(self.classes_)

        feature_specs = self._build_feature_specs(X_valid)
        population = [
            self._random_rule_gene(feature_specs)
            for _ in range(max(2, int(self.population_size)))
        ]

        hall_of_fame: list[tuple[float, _RuleGene]] = []
        for _ in range(max(1, int(self.generations))):
            scored = [
                (self._fitness(ind, X_valid, y_idx, n_classes), ind)
                for ind in population
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            hall_of_fame.extend(scored[: max(1, int(self.max_rules))])

            next_population: list[_RuleGene] = [scored[0][1]]
            while len(next_population) < len(population):
                parent_a = self._tournament_select(scored)
                parent_b = self._tournament_select(scored)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child, feature_specs)
                next_population.append(child)
            population = next_population

        hall_of_fame.sort(key=lambda item: item[0], reverse=True)
        best_genes: list[_RuleGene] = []
        seen = set()
        for _, gene in hall_of_fame:
            key = self._gene_key(gene)
            if key in seen:
                continue
            seen.add(key)
            best_genes.append(gene)
            if len(best_genes) >= max(0, int(self.max_rules)):
                break

        rules = self._genes_to_rules(best_genes, X_valid, y_idx, n_classes)
        if self.include_default_rule:
            prior_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
            rules.append(
                Rule(
                    atoms=[],
                    scores=self._distribution_to_scores(prior_counts),
                    rule_id="gp_default_prior",
                    metadata={"source": "gp_native", "kind": "class_prior"},
                )
            )

        self.ruleset_ = ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=self.feature_names_in_.tolist(),
            aggregation=AggregationSpec(type=self.aggregation, temperature=self.temperature),
            rules=rules,
            metadata={
                "source": "gp_native",
                "model_type": "genetic_programming",
                "population_size": int(self.population_size),
                "generations": int(self.generations),
                "score_mode": self._resolved_score_mode_,
            },
        )
        self.ruleset_.validate()
        return self

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = check_array(X, dtype=None)
        if X_valid.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_valid.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )
        return predict_from_ruleset(self.ruleset_, X_valid)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_valid = check_array(X, dtype=None)
        if X_valid.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_valid.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )
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
                thresholds = ((values[:-1] + values[1:]) / 2.0).tolist() if values.size >= 2 else []
                specs.append({"idx": feature_idx, "kind": "num", "thresholds": thresholds})
            else:
                categories = np.unique(np.asarray(col, dtype=object)).tolist()
                specs.append({"idx": feature_idx, "kind": "cat", "categories": categories})
        return specs

    def _random_rule_gene(self, specs: list[dict[str, object]]) -> _RuleGene:
        n_atoms = int(self._rng_.integers(1, max(2, int(self.max_atoms_per_rule) + 1)))
        atoms: list[_AtomGene] = []
        for _ in range(n_atoms):
            atoms.append(self._random_atom(specs))
        return _RuleGene(atoms=atoms)

    def _random_atom(self, specs: list[dict[str, object]]) -> _AtomGene:
        spec = specs[int(self._rng_.integers(0, len(specs)))]
        feature_idx = int(spec["idx"])
        if spec["kind"] == "num" and spec["thresholds"]:
            threshold = spec["thresholds"][int(self._rng_.integers(0, len(spec["thresholds"]))) ]
            op = "<=" if self._rng_.random() < 0.5 else ">"
            return _AtomGene(feature_idx=feature_idx, op=op, value=float(threshold))

        categories = spec.get("categories", [])
        if categories:
            value = categories[int(self._rng_.integers(0, len(categories)))]
            return _AtomGene(feature_idx=feature_idx, op="==", value=value)

        # Fallback fuer konstante numerische Features.
        return _AtomGene(feature_idx=feature_idx, op=">", value=0.0)

    def _rule_mask(self, gene: _RuleGene, X: np.ndarray) -> np.ndarray:
        mask = np.ones(X.shape[0], dtype=bool)
        for atom in gene.atoms:
            col = X[:, atom.feature_idx]
            if atom.op == "<=":
                mask &= np.asarray(col, dtype=float) <= float(atom.value)
            elif atom.op == ">":
                mask &= np.asarray(col, dtype=float) > float(atom.value)
            elif atom.op == "==":
                mask &= np.asarray(col, dtype=object) == atom.value
        return mask

    def _fitness(self, gene: _RuleGene, X: np.ndarray, y_idx: np.ndarray, n_classes: int) -> float:
        mask = self._rule_mask(gene, X)
        support = int(mask.sum())
        if support < self.min_samples_leaf:
            return -1e9

        counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
        purity = float(np.max(counts) / max(np.sum(counts), 1.0))
        coverage = float(support / X.shape[0])
        complexity = self.complexity_penalty * len(gene.atoms)
        return purity * (1.0 + coverage) - complexity

    def _tournament_select(self, scored: list[tuple[float, _RuleGene]]) -> _RuleGene:
        k = min(max(1, int(self.tournament_size)), len(scored))
        idx = self._rng_.choice(len(scored), size=k, replace=False)
        winner = max((scored[int(i)] for i in idx), key=lambda item: item[0])
        return winner[1]

    def _crossover(self, a: _RuleGene, b: _RuleGene) -> _RuleGene:
        if self._rng_.random() >= self.crossover_rate or not a.atoms or not b.atoms:
            return _RuleGene(atoms=list(a.atoms))

        cut_a = int(self._rng_.integers(1, len(a.atoms) + 1))
        cut_b = int(self._rng_.integers(1, len(b.atoms) + 1))
        atoms = list(a.atoms[:cut_a]) + list(b.atoms[cut_b - 1 :])
        atoms = atoms[: max(1, int(self.max_atoms_per_rule))]
        return _RuleGene(atoms=atoms)

    def _mutate(self, gene: _RuleGene, specs: list[dict[str, object]]) -> _RuleGene:
        atoms = list(gene.atoms)
        if self._rng_.random() < self.mutation_rate:
            if atoms and self._rng_.random() < 0.33:
                del atoms[int(self._rng_.integers(0, len(atoms)))]
            elif len(atoms) < max(1, int(self.max_atoms_per_rule)) and self._rng_.random() < 0.66:
                atoms.append(self._random_atom(specs))
            elif atoms:
                atoms[int(self._rng_.integers(0, len(atoms)))] = self._random_atom(specs)
        if not atoms:
            atoms = [self._random_atom(specs)]
        return _RuleGene(atoms=atoms)

    def _genes_to_rules(
        self,
        genes: list[_RuleGene],
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
    ) -> list[Rule]:
        rules: list[Rule] = []
        for idx, gene in enumerate(genes):
            mask = self._rule_mask(gene, X)
            if int(mask.sum()) < self.min_samples_leaf:
                continue
            counts = np.bincount(y_idx[mask], minlength=n_classes).astype(float)
            atoms = [
                Atom(
                    feature=str(self.feature_names_in_[a.feature_idx]),
                    op=a.op,
                    value=float(a.value) if a.op in {"<=", ">"} else a.value,
                )
                for a in gene.atoms
            ]
            rules.append(
                Rule(
                    atoms=atoms,
                    scores=self._distribution_to_scores(counts),
                    rule_id=f"gp_rule_{idx}",
                    metadata={"source": "gp_native", "support": int(mask.sum())},
                )
            )
        return rules

    def _distribution_to_scores(self, counts: np.ndarray) -> list[float]:
        probs = counts / max(float(np.sum(counts)), 1.0)
        if self._resolved_score_mode_ == "log_proba":
            return np.log(np.maximum(probs, 1e-12)).tolist()
        return probs.tolist()

    def _resolve_score_mode(self) -> str:
        if self.score_mode == "auto":
            return "log_proba" if self.aggregation == "softmax_sum" else "proba"
        if self.score_mode in {"log_proba", "proba"}:
            return self.score_mode
        raise ValueError(
            "Invalid score_mode. Expected one of: 'auto', 'log_proba', 'proba'."
        )

    @staticmethod
    def _gene_key(gene: _RuleGene) -> tuple[tuple[int, str, str], ...]:
        as_tuples = [(a.feature_idx, a.op, str(a.value)) for a in gene.atoms]
        return tuple(sorted(as_tuples))


