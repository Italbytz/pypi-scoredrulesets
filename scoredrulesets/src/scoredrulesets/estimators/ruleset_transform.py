"""
Transformationen für RuleKit und ExSTraCS zu Scored Rule Sets
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet


def rulekit_to_scored_ruleset(
    estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
) -> ScoredRuleSet:
    """
    Transformiere RuleKit Rule-Liste zu Scored Rule Set.
    
    RuleKit produziert eine Regel-Liste mit Voting-Schema.
    Jede Regel hat:
    - Conditions (Atome)
    - Class label (Zielklasse)
    
    Transformation:
    - Jede Regel mit Zielklasse c bekommt Score 1.0 für Klasse c
    - Atome werden aus den Conditions extrahiert (intervals → Range atoms)
    """
    try:
        # Versuche auf RuleKit-interne Struktur zuzugreifen
        if not hasattr(estimator, "rules"):
            raise TypeError("RuleKit Estimator hat keine 'rules' Attribut")
        
        rules: list[Rule] = []
        n_classes = len(class_labels)
        
        # Extrahiere Regeln aus RuleKit
        for rule_idx, rule in enumerate(estimator.rules):
            atoms: list[Atom] = []
            
            # Extrahiere Conditions aus der Regel
            if hasattr(rule, "conditions"):
                for condition in rule.conditions:
                    atom = _condition_to_atom(condition, feature_names)
                    if atom is not None:
                        atoms.append(atom)
            elif hasattr(rule, "premise"):
                # Alternative Struktur
                for condition in rule.premise:
                    atom = _condition_to_atom(condition, feature_names)
                    if atom is not None:
                        atoms.append(atom)
            
            # Extrahiere Zielklasse
            class_idx = 0
            if hasattr(rule, "conclusion"):
                class_label = rule.conclusion
                try:
                    class_idx = class_labels.index(class_label)
                except (ValueError, IndexError):
                    class_idx = int(class_label) if isinstance(class_label, (int, np.integer)) else 0
            elif hasattr(rule, "target"):
                class_label = rule.target
                try:
                    class_idx = class_labels.index(class_label)
                except (ValueError, IndexError):
                    class_idx = int(class_label) if isinstance(class_label, (int, np.integer)) else 0
            
            # Erstelle Score-Vektor (1.0 für Zielklasse, 0 sonst)
            scores = [0.0] * n_classes
            scores[class_idx] = 1.0
            
            rules.append(
                Rule(
                    atoms=atoms,
                    scores=scores,
                    rule_id=f"rulekit_{rule_idx}",
                    metadata={"source": "rulekit", "class_label": class_labels[class_idx]},
                )
            )
        
        # Füge Default-Regel hinzu (falls nicht vorhanden)
        if not any(len(r.atoms) == 0 for r in rules):
            # Berechne Häufigkeiten für Default-Regel
            default_scores = [1.0 / n_classes] * n_classes
            rules.insert(
                0,
                Rule(
                    atoms=[],
                    scores=default_scores,
                    rule_id="default",
                    metadata={"source": "rulekit_default"},
                ),
            )
        
        ruleset = ScoredRuleSet(
            class_labels=class_labels,
            feature_names=feature_names,
            rules=rules,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={"transform": "rulekit_to_scored_ruleset", "n_rules": len(rules)},
        )
        ruleset.validate()
        return ruleset
        
    except Exception as e:
        raise TypeError(f"Konnte RuleKit zu Scored Rule Set nicht transformieren: {e}") from e


def exstracs_to_scored_ruleset(
    estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
) -> ScoredRuleSet:
    """
    Transformiere ExSTraCS Population zu Scored Rule Set.
    
    ExSTraCS produziert eine Population von Regeln mit:
    - Conditions (intervals)
    - Class label (Zielklasse)
    - Fitness und Numerosity
    
    Transformation:
    - Jede Regel bekommt Score = fitness × numerosity für ihre Klasse
    - Intervals werden zu Range-Atoms
    """
    try:
        # Versuche auf ExSTraCS-interne Struktur zuzugreifen
        if not hasattr(estimator, "pop") and not hasattr(estimator, "population"):
            raise TypeError("ExSTraCS Estimator hat keine 'pop' oder 'population' Attribut")
        
        population = estimator.pop if hasattr(estimator, "pop") else estimator.population
        rules: list[Rule] = []
        n_classes = len(class_labels)
        
        # Verarbeite jede Regel in der Population
        for rule_idx, rule in enumerate(population):
            atoms: list[Atom] = []
            
            # Extrahiere Conditions (Intervals)
            if hasattr(rule, "condition"):
                # skExSTraCS Format
                for feat_idx, interval in enumerate(rule.condition):
                    if feat_idx < len(feature_names):
                        feature_name = feature_names[feat_idx]
                        # Überspringe "don't care" Zustände (None oder wild)
                        if interval is not None and interval != "#":
                            atom = _interval_to_atom(interval, feature_name)
                            if atom is not None:
                                atoms.append(atom)
            elif hasattr(rule, "alleles"):
                # Alternative Format
                for feat_idx, allele in enumerate(rule.alleles):
                    if feat_idx < len(feature_names):
                        feature_name = feature_names[feat_idx]
                        if allele is not None and allele != "#":
                            atom = _interval_to_atom(allele, feature_name)
                            if atom is not None:
                                atoms.append(atom)
            
            # Extrahiere Zielklasse und Fitness
            class_idx = 0
            fitness = 1.0
            numerosity = 1
            
            if hasattr(rule, "phenotype"):
                class_label = rule.phenotype
                try:
                    class_idx = class_labels.index(class_label)
                except (ValueError, IndexError):
                    class_idx = int(class_label) if isinstance(class_label, (int, np.integer)) else 0
            
            if hasattr(rule, "fitness"):
                fitness = float(rule.fitness)
            if hasattr(rule, "numerosity"):
                numerosity = float(rule.numerosity)
            
            # Score = fitness × numerosity
            score_value = fitness * numerosity
            
            # Erstelle Score-Vektor
            scores = [0.0] * n_classes
            scores[class_idx] = score_value
            
            rules.append(
                Rule(
                    atoms=atoms,
                    scores=scores,
                    rule_id=f"exstracs_{rule_idx}",
                    metadata={
                        "source": "exstracs",
                        "fitness": fitness,
                        "numerosity": numerosity,
                        "class_label": class_labels[class_idx],
                    },
                )
            )
        
        # Füge Default-Regel hinzu (falls nicht vorhanden)
        if not any(len(r.atoms) == 0 for r in rules):
            # Berechne Häufigkeiten für Default-Regel
            default_scores = [1.0 / n_classes] * n_classes
            rules.insert(
                0,
                Rule(
                    atoms=[],
                    scores=default_scores,
                    rule_id="default",
                    metadata={"source": "exstracs_default"},
                ),
            )
        
        ruleset = ScoredRuleSet(
            class_labels=class_labels,
            feature_names=feature_names,
            rules=rules,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={"transform": "exstracs_to_scored_ruleset", "n_rules": len(rules)},
        )
        ruleset.validate()
        return ruleset
        
    except Exception as e:
        raise TypeError(f"Konnte ExSTraCS zu Scored Rule Set nicht transformieren: {e}") from e


def _condition_to_atom(condition: Any, feature_names: list[str]) -> Atom | None:
    """
    Transformiere RuleKit Condition zu Atom.
    
    RuleKit Conditions können verschiedene Formate haben.
    """
    try:
        # Versuche Standard-Attribute zu extrahieren
        if hasattr(condition, "attribute") and hasattr(condition, "value"):
            feature_name = condition.attribute
            if isinstance(feature_name, int) and feature_name < len(feature_names):
                feature_name = feature_names[feature_name]
            
            if hasattr(condition, "operator"):
                op = condition.operator
                # Normalisiere Operator
                if op in ("=", "==", "equals"):
                    return Atom(feature=str(feature_name), op="==", value=float(condition.value))
                elif op in ("<", "less"):
                    return Atom(feature=str(feature_name), op="<", value=float(condition.value))
                elif op in (">", "greater"):
                    return Atom(feature=str(feature_name), op=">", value=float(condition.value))
                elif op in ("<=", "less_or_equal"):
                    return Atom(feature=str(feature_name), op="<=", value=float(condition.value))
                elif op in (">=", "greater_or_equal"):
                    return Atom(feature=str(feature_name), op=">=", value=float(condition.value))
        
        return None
    except Exception:
        return None


def _interval_to_atom(interval: Any, feature_name: str) -> Atom | None:
    """
    Transformiere ExSTraCS Interval zu Atom.
    
    ExSTraCS nutzt Intervals [lower, upper] für kontinuierliche Features.
    """
    try:
        # Interval als Tupel (lower, upper)
        if isinstance(interval, (tuple, list)) and len(interval) == 2:
            lower, upper = interval
            # Nutze Range-Atom (zwischen lower und upper)
            # Wir konstruieren ein "in range" Atom
            midpoint = (float(lower) + float(upper)) / 2.0
            return Atom(
                feature=str(feature_name),
                op="in",
                value={"lower": float(lower), "upper": float(upper)},
            )
        
        return None
    except Exception:
        return None

