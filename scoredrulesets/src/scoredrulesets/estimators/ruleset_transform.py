"""
Transformationen für RuleKit und ExSTraCS zu Scored Rule Sets
"""

from __future__ import annotations

import math
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
        rule_source = _extract_rulekit_rules(estimator)
        if rule_source is None:
            raise TypeError("RuleKit Estimator hat keine bekannte Regelquelle ('rules' oder 'model.rules')")
        
        rules: list[Rule] = []
        n_classes = len(class_labels)
        
        # Extrahiere Regeln aus RuleKit
        for rule_idx, rule in enumerate(rule_source):
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
            elif hasattr(rule, "_java_object") and hasattr(rule._java_object, "getPremise"):
                premise = rule._java_object.getPremise()
                subconditions = premise.getSubconditions() if premise is not None else None
                if subconditions is not None:
                    for idx in range(subconditions.size()):
                        atom = _condition_to_atom(subconditions.get(idx), feature_names)
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
            elif hasattr(rule, "decision_class"):
                class_idx = _resolve_class_index(rule.decision_class, class_labels)
            
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
        if hasattr(population, "popSet"):
            iterable_population = population.popSet
        else:
            iterable_population = population
        rules: list[Rule] = []
        n_classes = len(class_labels)

        # Debug: Sammle alle Zielklassen (phenotype) aus der Population
        all_phenotypes = set()

        # Verarbeite jede Regel in der Population
        for rule_idx, rule in enumerate(iterable_population):
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
                all_phenotypes.add(class_label)
                try:
                    class_idx = class_labels.index(class_label)
                except (ValueError, IndexError):
                    print(f"[WARN] ExSTraCS-Regel {rule_idx}: phenotype '{class_label}' nicht in class_labels {class_labels}. Mapping auf 0.")
                    class_idx = 0
            else:
                print(f"[WARN] ExSTraCS-Regel {rule_idx}: Keine phenotype gefunden, mapping auf 0.")

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
                        "class_label": class_labels[class_idx] if class_idx < len(class_labels) else class_idx,
                        "phenotype": getattr(rule, "phenotype", None),
                    },
                )
            )

        # Debug-Ausgabe: Alle gefundenen Zielklassen
        print(f"[DEBUG] ExSTraCS: class_labels={class_labels}, phenotypes in population={sorted(all_phenotypes)}")

        # Füge Default-Regel hinzu (falls nicht vorhanden, und nur einmal!)
        default_rule_count = sum(1 for r in rules if len(r.atoms) == 0)
        if default_rule_count == 0:
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
        elif default_rule_count > 1:
            # Entferne alle bis auf eine Default-Regel
            new_rules = [r for r in rules if len(r.atoms) > 0]
            # Füge die erste gefundene Default-Regel wieder ein
            for r in rules:
                if len(r.atoms) == 0:
                    new_rules.insert(0, r)
                    break
            rules = new_rules

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


def rulefit_to_scored_ruleset(
    estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
) -> ScoredRuleSet:
    """
    Transformiere RuleFit-Regeln zu Scored Rule Set.
    Jede Regel wird als Atom-Liste mit Score übernommen.
    """
    try:
        rules = []
        n_classes = len(class_labels)
        # RuleFit speichert Regeln in estimator.rule_ und estimator.coef_
        rule_attrs = getattr(estimator, "rule_", None)
        coefs = getattr(estimator, "coef_", None)
        if rule_attrs is None or coefs is None:
            raise TypeError("RuleFit Estimator hat keine 'rule_' oder 'coef_' Attribute")
        for idx, (rule, coef) in enumerate(zip(rule_attrs, coefs)):
            if coef == 0 or rule is None:
                continue
            # Regel kann entweder eine DecisionRule oder ein LinearTerm sein
            atoms = []
            if hasattr(rule, "feature_index") and rule.feature_index is not None:
                # LinearTerm: Einzelnes Feature
                fname = feature_names[rule.feature_index]
                atoms.append(Atom(feature=fname, op="linear", value=None))
            elif hasattr(rule, "rule") and rule.rule is not None:
                # DecisionRule: Bedingungen als String
                # Wir parsen die Bedingung grob (z.B. "feature <= 1.5 and feature2 > 0.2")
                conds = str(rule.rule).split(" and ")
                for cond in conds:
                    for op in ["<=", ">=", "<", ">", "=="]:
                        if op in cond:
                            fname, val = cond.split(op)
                            fname = fname.strip()
                            val = val.strip()
                            try:
                                val = float(val)
                            except Exception:
                                pass
                            atoms.append(Atom(feature=fname, op=op, value=val))
                            break
            # Score-Vektor: Für Klassifikation: Score auf alle Klassen, für Regression: auf alle
            scores = [0.0] * n_classes
            # RuleFit ist meist binär, Score auf positive Klasse
            if n_classes == 2:
                scores[1] = float(coef)
            else:
                # Multi-Klasse: Score auf alle (hier: nur auf Index 0)
                scores[0] = float(coef)
            rules.append(
                Rule(
                    atoms=atoms,
                    scores=scores,
                    rule_id=f"rulefit_{idx}",
                    metadata={"source": "rulefit"},
                )
            )
        # Default-Regel
        if not any(len(r.atoms) == 0 for r in rules):
            default_scores = [1.0 / n_classes] * n_classes
            rules.insert(
                0,
                Rule(
                    atoms=[],
                    scores=default_scores,
                    rule_id="default",
                    metadata={"source": "rulefit_default"},
                ),
            )
        ruleset = ScoredRuleSet(
            class_labels=class_labels,
            feature_names=feature_names,
            rules=rules,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={"transform": "rulefit_to_scored_ruleset", "n_rules": len(rules)},
        )
        ruleset.validate()
        return ruleset
    except Exception as e:
        raise TypeError(f"Konnte RuleFit zu Scored Rule Set nicht transformieren: {e}") from e


def _condition_to_atom(condition: Any, feature_names: list[str]) -> Atom | None:
    """
    Transformiere RuleKit Condition zu Atom.
    
    RuleKit Conditions können verschiedene Formate haben.
    """
    try:
        # RuleKit-Java-Objektpfad: ElementaryCondition mit ValueSet
        if hasattr(condition, "getAttribute") and hasattr(condition, "getValueSet"):
            feature_name = str(condition.getAttribute())
            if feature_name.startswith("att") and feature_name[3:].isdigit():
                idx = int(feature_name[3:]) - 1
                if 0 <= idx < len(feature_names):
                    feature_name = feature_names[idx]

            value_set = condition.getValueSet()
            if value_set is None:
                return None

            if hasattr(value_set, "getLeft") and hasattr(value_set, "getRight"):
                left = float(value_set.getLeft())
                right = float(value_set.getRight())
                left_sign = str(value_set.getLeftSign()) if hasattr(value_set, "getLeftSign") else ">="
                right_sign = str(value_set.getRightSign()) if hasattr(value_set, "getRightSign") else "<="

                left_inf = not math.isfinite(left) or left <= -1e300
                right_inf = not math.isfinite(right) or right >= 1e300

                if left_inf and not right_inf:
                    return Atom(feature=str(feature_name), op=right_sign, value=right)
                if right_inf and not left_inf:
                    return Atom(feature=str(feature_name), op=left_sign, value=left)
                if not left_inf and not right_inf:
                    # Bei endlichen Intervallen nutzen wir ein zwischen-Atom.
                    return Atom(feature=str(feature_name), op="between", value=[left, right])
                return None

            if hasattr(value_set, "getValue"):
                return Atom(feature=str(feature_name), op="==", value=float(value_set.getValue()))
            if hasattr(value_set, "getValueAsString"):
                value = value_set.getValueAsString()
                try:
                    value = float(value)
                except Exception:
                    value = str(value)
                return Atom(feature=str(feature_name), op="==", value=value)

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


def _extract_rulekit_rules(estimator: Any):
    if hasattr(estimator, "rules") and estimator.rules is not None:
        return estimator.rules

    model = getattr(estimator, "model", None)
    if model is not None and hasattr(model, "rules") and model.rules is not None:
        return model.rules

    return None


def _resolve_class_index(class_label: Any, class_labels: list[Any]) -> int:
    for candidate in (class_label, str(class_label)):
        try:
            return class_labels.index(candidate)
        except (ValueError, IndexError):
            continue

    try:
        idx = int(class_label)
        if 0 <= idx < len(class_labels):
            return idx
    except Exception:
        pass
    return 0


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

