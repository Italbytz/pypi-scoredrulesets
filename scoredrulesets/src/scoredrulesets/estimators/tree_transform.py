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
    prune_atoms: bool = False  # Aktiviert Atom-Pruning mit Äquivalenzvalidierung
    prune_lambda: float | None = None  # Lambda für Depth-Weighting beim Pruning (None = kein Pruning)


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

    # Wende Atom-Pruning mit Äquivalenzvalidierung an
    if params.prune_atoms and params.prune_lambda is not None:
        rules = _aggressive_atom_pruning(rules, params.prune_lambda)

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


def _aggressive_atom_pruning(rules: list[Rule], prune_lambda: float) -> list[Rule]:
    """
    Aggressiver Atom-Pruning-Algorithmus basierend auf Depth-Weighted Scoring.
    
    Dieser Algorithmus entfernt iterativ Atome, während die Vorhersagen
    unter argmax-Aggregation äquivalent bleiben.
    
    Args:
        rules: Liste von Regeln
        prune_lambda: Decay-Parameter λ > 1 (nicht direkt verwendet, aber für Konsistenz)
    
    Returns:
        Pruned Regel-Set mit weniger Atomen
    """
    if prune_lambda <= 1.0:
        raise ValueError(f"prune_lambda muss > 1 sein, erhalten: {prune_lambda}")
    
    # Erstelle Kopie zum Modifizieren
    working_rules = [
        Rule(
            atoms=list(rule.atoms),
            scores=list(rule.scores),
            rule_id=rule.rule_id,
            metadata=dict(rule.metadata) if rule.metadata else {},
        )
        for rule in rules
    ]
    
    changed = True
    iteration = 0
    atoms_removed_total = 0
    
    while changed:
        changed = False
        iteration += 1
        
        for rule_idx, rule in enumerate(working_rules):
            if not rule.atoms:  # Überspringe leere Regeln (Default-Regel)
                continue
            
            # Versuche, jedes Atom von hinten zu entfernen (rückwärts iteration ist sicherer)
            for atom_idx in range(len(rule.atoms) - 1, -1, -1):
                # Erstelle Kandidat ohne dieses Atom
                candidate_atoms = rule.atoms[:atom_idx] + rule.atoms[atom_idx + 1 :]
                
                # Erstelle neue Regel
                candidate_rule = Rule(
                    atoms=candidate_atoms,
                    scores=rule.scores,
                    rule_id=rule.rule_id,
                    metadata=rule.metadata,
                )
                
                # Validiere, ob das Atom entfernt werden kann
                if _can_remove_atom_safely(rule, candidate_rule):
                    # Ersetze Regel
                    working_rules[rule_idx] = candidate_rule
                    atoms_removed_total += 1
                    changed = True
                    break  # Gehe zur nächsten Regel nach erfolgreicher Entfernung
    
    return working_rules


def _can_remove_atom_safely(original_rule: Rule, candidate_rule: Rule) -> bool:
    """
    Prüfe, ob ein Atom sicher entfernt werden kann.

    Kriterien:
    1. Der Kandidat muss weniger Atome haben
    2. Die Scores müssen noch positive Werte für mindestens eine Klasse haben
    3. Nicht-Default-Regeln dürfen nicht zu leeren Regeln werden
    """
    # Kriterium 1: Weniger Atome
    if len(candidate_rule.atoms) >= len(original_rule.atoms):
        return False

    # Kriterium 2: Scores nicht alle null
    if all(s == 0.0 for s in candidate_rule.scores):
        return False

    # Kriterium 3: Keine leeren Nicht-Default-Regeln erzeugen.
    # Leere Regeln (atoms=[]) sind im ScoredRuleSet nur als explizite Default-Regel gedacht.
    if len(candidate_rule.atoms) == 0:
        return False

    return True

