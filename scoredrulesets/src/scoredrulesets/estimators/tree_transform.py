from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet


@dataclass
class TreeTransformParams:
    depth_decay_lambda: float = 2.0
    include_default_rule: bool = False
    default_rule_strength: float = 0.0
    aggressive_prune: bool = False


def estimator_to_scored_ruleset(
    estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
    params: TreeTransformParams,
) -> ScoredRuleSet:
    tree_estimator = _unwrap_tree_estimator(estimator)
    if not hasattr(tree_estimator, "tree_"):
        raise TypeError("Estimator kann nicht in tree_ aufgeloest werden")

    rules: list[Rule] = []
    tree = tree_estimator.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    # Bei sklearn ist value je Blatt die Klassenverteilung.
    leaf_values = tree.value.squeeze(axis=1) if tree.value.ndim == 3 else tree.value

    def visit(node_id: int, path_atoms: list[Atom], depth: int) -> None:
        is_leaf = children_left[node_id] == children_right[node_id]
        if is_leaf:
            class_idx = int(np.argmax(leaf_values[node_id]))
            weight = float(params.depth_decay_lambda ** (-max(depth, 1)))
            scores = [0.0 for _ in class_labels]
            scores[class_idx] = weight
            rules.append(
                Rule(
                    atoms=list(path_atoms),
                    scores=scores,
                    rule_id=f"leaf_{node_id}",
                    metadata={"depth": depth, "source": "tree_path"},
                )
            )
            return

        split_feature_idx = int(feature[node_id])
        split_name = feature_names[split_feature_idx]
        split_threshold = float(threshold[node_id])

        visit(
            children_left[node_id],
            path_atoms
            + [Atom(feature=split_name, op="<=", value=split_threshold)],
            depth + 1,
        )
        visit(
            children_right[node_id],
            path_atoms + [Atom(feature=split_name, op=">", value=split_threshold)],
            depth + 1,
        )

    visit(0, [], 0)

    if params.include_default_rule:
        default_scores = [float(params.default_rule_strength) for _ in class_labels]
        rules.append(Rule(atoms=[], scores=default_scores, rule_id="default"))

    if params.aggressive_prune:
        rules = _deduplicate_atoms(rules)

    ruleset = ScoredRuleSet(
        class_labels=class_labels,
        feature_names=feature_names,
        rules=rules,
        aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
        metadata={"transform": "tree_to_scored_ruleset", "depth_decay_lambda": params.depth_decay_lambda},
    )
    ruleset.validate()
    return ruleset


def _unwrap_tree_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "tree_"):
        return estimator

    for attr in ("estimator_", "model_", "best_estimator_", "tree_estimator_"):
        inner = getattr(estimator, attr, None)
        if inner is not None and inner is not estimator:
            if hasattr(inner, "tree_"):
                return inner

    return estimator


def _deduplicate_atoms(rules: list[Rule]) -> list[Rule]:
    compact_rules: list[Rule] = []
    for rule in rules:
        seen: set[tuple[str, str, str]] = set()
        deduped = []
        for atom in rule.atoms:
            key = (str(atom.feature), atom.op, str(atom.value))
            if key not in seen:
                seen.add(key)
                deduped.append(atom)
        compact_rules.append(
            Rule(
                atoms=deduped,
                scores=rule.scores,
                rule_id=rule.rule_id,
                metadata=rule.metadata,
            )
        )
    return compact_rules

