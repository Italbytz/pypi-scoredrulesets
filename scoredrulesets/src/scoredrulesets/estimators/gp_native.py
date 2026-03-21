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
        selection_mode: str = "fitness",
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
        validation_fraction: float = 0.2,
        early_stopping_rounds: int = 5,
        random_state: int | None = None,
    ):
        self.aggregation = aggregation
        self.temperature = temperature
        self.score_mode = score_mode
        self.selection_mode = selection_mode
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
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    def fit(self, X, y):
        X_valid, y_valid = check_X_y(X, y, dtype=None)
        self._resolved_score_mode_ = self._resolve_score_mode()
        self._resolved_selection_mode_ = self._resolve_selection_mode()
        self.n_features_in_ = X_valid.shape[1]
        self.feature_names_in_ = np.asarray([f"f{i}" for i in range(self.n_features_in_)], dtype=object)
        self.classes_ = unique_labels(y_valid)

        self._rng_ = np.random.default_rng(self.random_state)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[val] for val in y_valid], dtype=int)
        n_classes = len(self.classes_)
        train_idx, val_idx = self._train_val_indices(y_idx)
        self._used_validation_ = val_idx is not None

        feature_specs = self._build_feature_specs(X_valid)
        population = [
            self._random_rule_gene(feature_specs)
            for _ in range(max(2, int(self.population_size)))
        ]

        hall_of_fame: list[tuple[int, float, int, _RuleGene]] = []
        max_generations = max(1, int(self.generations))
        best_generation_score = -np.inf
        no_improvement_rounds = 0
        generations_ran = 0
        for generation_idx in range(max_generations):
            scored = self._score_population(
                population,
                X_valid,
                y_idx,
                n_classes,
                train_idx,
                val_idx,
            )
            ranked = self._rank_scored_population(scored)
            generations_ran = generation_idx + 1
            hall_of_fame.extend(ranked[: max(1, int(self.max_rules))])

            top_score = float(ranked[0][1])
            if top_score > best_generation_score + 1e-12:
                best_generation_score = top_score
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
                if no_improvement_rounds >= max(1, int(self.early_stopping_rounds)):
                    break

            next_population: list[_RuleGene] = [ranked[0][3]]
            while len(next_population) < len(population):
                parent_a = self._tournament_select(ranked)
                parent_b = self._tournament_select(ranked)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child, feature_specs)
                next_population.append(child)
            population = next_population

        hall_of_fame.sort(key=lambda item: (item[0], -item[1], item[2]))
        best_genes: list[_RuleGene] = []
        seen = set()
        for _, _, _, gene in hall_of_fame:
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
                "generations_ran": int(generations_ran),
                "early_stopped": bool(generations_ran < max_generations),
                "used_validation": bool(self._used_validation_),
                "validation_fraction": float(self.validation_fraction),
                "score_mode": self._resolved_score_mode_,
                "selection_mode": self._resolved_selection_mode_,
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
                intervals = []
                if values.size >= 3:
                    q_points = np.unique(np.quantile(values, [0.15, 0.35, 0.5, 0.65, 0.85]))
                    for i in range(len(q_points) - 1):
                        low = float(q_points[i])
                        high = float(q_points[i + 1])
                        if low < high:
                            intervals.append((low, high))
                specs.append(
                    {
                        "idx": feature_idx,
                        "kind": "num",
                        "thresholds": thresholds,
                        "intervals": intervals,
                    }
                )
            else:
                categories = np.unique(np.asarray(col, dtype=object)).tolist()
                groups = []
                if len(categories) >= 3:
                    for i in range(len(categories)):
                        for j in range(i + 1, len(categories)):
                            groups.append([categories[i], categories[j]])
                specs.append(
                    {
                        "idx": feature_idx,
                        "kind": "cat",
                        "categories": categories,
                        "groups": groups,
                    }
                )
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
        if spec["kind"] == "num":
            thresholds = spec.get("thresholds", [])
            intervals = spec.get("intervals", [])
            ops = []
            if thresholds:
                ops.extend(["<=", ">"])
            if intervals:
                ops.append("between")

            if ops:
                op = ops[int(self._rng_.integers(0, len(ops)))]
                if op in {"<=", ">"}:
                    threshold = thresholds[int(self._rng_.integers(0, len(thresholds)))]
                    return _AtomGene(feature_idx=feature_idx, op=op, value=float(threshold))
                interval = intervals[int(self._rng_.integers(0, len(intervals)))]
                return _AtomGene(feature_idx=feature_idx, op="between", value=[float(interval[0]), float(interval[1])])

        categories = spec.get("categories", [])
        groups = spec.get("groups", [])
        if categories:
            if groups and self._rng_.random() < 0.5:
                value = groups[int(self._rng_.integers(0, len(groups)))]
                return _AtomGene(feature_idx=feature_idx, op="in", value=list(value))
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
            elif atom.op == "between":
                low, high = atom.value
                num_col = np.asarray(col, dtype=float)
                mask &= (num_col >= float(low)) & (num_col <= float(high))
            elif atom.op == "==":
                mask &= np.asarray(col, dtype=object) == atom.value
            elif atom.op == "in":
                mask &= np.isin(np.asarray(col, dtype=object), list(atom.value))
        return mask

    def _fitness(
        self,
        gene: _RuleGene,
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> float:
        train_score = self._subset_quality(gene, X, y_idx, n_classes, train_idx)
        if train_score <= -1e8:
            return train_score

        if val_idx is None:
            return train_score

        val_score = self._subset_quality(gene, X, y_idx, n_classes, val_idx)
        if val_score <= -1e8:
            return val_score

        # Leichte Strafung, wenn der Trainingsscore den Val-Score deutlich uebersteigt.
        overfit_penalty = max(0.0, train_score - val_score) * 0.2
        return val_score - overfit_penalty

    def _subset_quality(
        self,
        gene: _RuleGene,
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        subset_idx: np.ndarray,
    ) -> float:
        if subset_idx.size == 0:
            return -1e9

        mask = self._rule_mask(gene, X[subset_idx])
        support = int(mask.sum())
        if support < self.min_samples_leaf:
            return -1e9

        counts = np.bincount(y_idx[subset_idx][mask], minlength=n_classes).astype(float)
        purity = float(np.max(counts) / max(np.sum(counts), 1.0))
        coverage = float(support / subset_idx.size)
        complexity = self.complexity_penalty * len(gene.atoms)
        return purity * (1.0 + coverage) - complexity

    def _train_val_indices(self, y_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        n_samples = y_idx.shape[0]
        if n_samples < 4 or self.validation_fraction <= 0.0:
            return np.arange(n_samples), None

        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for class_idx in np.unique(y_idx):
            cls_indices = np.where(y_idx == class_idx)[0]
            if cls_indices.size <= 1:
                train_parts.append(cls_indices)
                continue

            shuffled = cls_indices[self._rng_.permutation(cls_indices.size)]
            n_val = int(round(cls_indices.size * float(self.validation_fraction)))
            n_val = max(1, min(cls_indices.size - 1, n_val))
            val_parts.append(shuffled[:n_val])
            train_parts.append(shuffled[n_val:])

        train_idx = np.concatenate(train_parts) if train_parts else np.arange(n_samples)
        if not val_parts:
            return train_idx, None

        val_idx = np.concatenate(val_parts)
        if val_idx.size == 0:
            return train_idx, None
        return train_idx, val_idx

    def _tournament_select(self, scored: list[tuple[int, float, int, _RuleGene]]) -> _RuleGene:
        k = min(max(1, int(self.tournament_size)), len(scored))
        idx = self._rng_.choice(len(scored), size=k, replace=False)
        winner = min((scored[int(i)] for i in idx), key=lambda item: (item[0], -item[1], item[2]))
        return winner[3]

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
                    value=self._atom_value_to_schema(a),
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

    def _resolve_selection_mode(self) -> str:
        if self.selection_mode in {"fitness", "pareto"}:
            return self.selection_mode
        raise ValueError(
            "Invalid selection_mode. Expected one of: 'fitness', 'pareto'."
        )

    def _score_population(
        self,
        population: list[_RuleGene],
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None,
    ) -> list[tuple[float, int, _RuleGene]]:
        scored = []
        for ind in population:
            quality = self._fitness(ind, X, y_idx, n_classes, train_idx, val_idx)
            complexity = len(ind.atoms)
            scored.append((float(quality), int(complexity), ind))
        return scored

    def _rank_scored_population(
        self,
        scored: list[tuple[float, int, _RuleGene]],
    ) -> list[tuple[int, float, int, _RuleGene]]:
        if self._resolved_selection_mode_ == "fitness":
            ordered = sorted(scored, key=lambda item: item[0], reverse=True)
            return [(0, quality, complexity, gene) for quality, complexity, gene in ordered]

        objectives = [(quality, complexity) for quality, complexity, _ in scored]
        ranks = self._pareto_front_ranks(objectives)
        ranked = [
            (ranks[idx], scored[idx][0], scored[idx][1], scored[idx][2])
            for idx in range(len(scored))
        ]
        ranked.sort(key=lambda item: (item[0], -item[1], item[2]))
        return ranked

    @staticmethod
    def _pareto_front_ranks(objectives: list[tuple[float, int]]) -> list[int]:
        # Maximiert quality, minimiert complexity.
        n = len(objectives)
        dominates = [set() for _ in range(n)]
        dominated_count = [0 for _ in range(n)]
        ranks = [0 for _ in range(n)]

        def _dominates(a: tuple[float, int], b: tuple[float, int]) -> bool:
            q_a, c_a = a
            q_b, c_b = b
            not_worse = (q_a >= q_b) and (c_a <= c_b)
            strictly_better = (q_a > q_b) or (c_a < c_b)
            return not_worse and strictly_better

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if _dominates(objectives[i], objectives[j]):
                    dominates[i].add(j)
                elif _dominates(objectives[j], objectives[i]):
                    dominated_count[i] += 1

        current_front = [i for i in range(n) if dominated_count[i] == 0]
        front_rank = 0
        while current_front:
            next_front: list[int] = []
            for idx in current_front:
                ranks[idx] = front_rank
                for j in dominates[idx]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            front_rank += 1
            current_front = next_front
        return ranks

    @staticmethod
    def _gene_key(gene: _RuleGene) -> tuple[tuple[int, str, str], ...]:
        as_tuples = []
        for atom in gene.atoms:
            if atom.op == "between":
                low, high = atom.value
                value_repr = f"{float(low):.12g}:{float(high):.12g}"
            elif atom.op == "in":
                value_repr = "|".join(sorted(str(v) for v in atom.value))
            else:
                value_repr = str(atom.value)
            as_tuples.append((atom.feature_idx, atom.op, value_repr))
        return tuple(sorted(as_tuples))

    @staticmethod
    def _atom_value_to_schema(atom: _AtomGene):
        if atom.op in {"<=", ">"}:
            return float(atom.value)
        if atom.op == "between":
            low, high = atom.value
            return [float(low), float(high)]
        if atom.op == "in":
            return list(atom.value)
        return atom.value


