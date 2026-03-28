"""
Transformations from RuleKit and ExSTraCS to scored rule sets.
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
    y_train: np.ndarray | None = None,
) -> ScoredRuleSet:
    """
    Transform a RuleKit rule list into a scored rule set.
    
    RuleKit produces a rule list with voting semantics.
    Each rule has:
    - conditions (atoms)
    - class label (target class)
    
    Transformation:
    - each rule with target class c gets score 1.0 for class c
    - atoms are extracted from conditions (intervals -> range atoms)
    """
    try:
        rule_source = _extract_rulekit_rules(estimator)
        if rule_source is None:
            raise TypeError("RuleKit estimator has no known rule source ('rules' or 'model.rules')")
        
        rules: list[Rule] = []
        n_classes = len(class_labels)
        
        # Extract rules from RuleKit.
        for rule_idx, rule in enumerate(rule_source):
            atoms: list[Atom] = []
            
            # Extract conditions from the rule.
            if hasattr(rule, "conditions"):
                for condition in rule.conditions:
                    atoms.extend(_condition_to_atoms(condition, feature_names))
            elif hasattr(rule, "premise"):
                # Alternative Struktur
                for condition in rule.premise:
                    atoms.extend(_condition_to_atoms(condition, feature_names))
            elif hasattr(rule, "_java_object") and hasattr(rule._java_object, "getPremise"):
                premise = rule._java_object.getPremise()
                subconditions = premise.getSubconditions() if premise is not None else None
                if subconditions is not None:
                    for idx in range(subconditions.size()):
                        atoms.extend(_condition_to_atoms(subconditions.get(idx), feature_names))
            
            # Extract target class.
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
            
            # Use RuleKit-like weighting instead of pure one-hot scores.
            rule_strength = _extract_rulekit_rule_strength(rule)
            scores = [0.0] * n_classes
            scores[class_idx] = rule_strength
            
            rules.append(
                Rule(
                    atoms=atoms,
                    scores=scores,
                    rule_id=f"rulekit_{rule_idx}",
                    metadata={
                        "source": "rulekit",
                        "class_label": class_labels[class_idx],
                        "rule_strength": rule_strength,
                    },
                )
            )
        
        # Add default rule if missing.
        if not any(len(r.atoms) == 0 for r in rules):
            default_scores = _rulekit_default_scores(
                estimator=estimator,
                class_labels=class_labels,
                y_train=y_train,
            )
            rules.insert(
                0,
                Rule(
                    atoms=[],
                    scores=default_scores,
                    rule_id="default",
                    metadata={"source": "rulekit_default", "prior_based": True},
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
        raise TypeError(f"Could not transform RuleKit to scored rule set: {e}") from e


def exstracs_to_scored_ruleset(
    estimator: Any,
    class_labels: list[Any],
    feature_names: list[str],
) -> ScoredRuleSet:
    """
        Transform an ExSTraCS population into a scored rule set.
    
        ExSTraCS produces a population of rules with:
        - specifiedAttList: list of specified feature indices (sparse representation)
        - condition: list of conditions (parallel to specifiedAttList)
        - phenotype: target class
    - fitness und numerosity
    
    Transformation:
        - each rule gets score = fitness x numerosity x classPredictionWeight
        - continuous intervals [lower, upper] are converted to two atoms: feature > lower AND feature < upper
            (ExSTraCS uses strict inequalities)
        - discrete values are converted to == atoms
    """
    try:
        # Try to access ExSTraCS internal structures.
        if not hasattr(estimator, "pop") and not hasattr(estimator, "population"):
            raise TypeError("ExSTraCS estimator has no 'pop' or 'population' attribute")

        population = estimator.pop if hasattr(estimator, "pop") else estimator.population
        if hasattr(population, "popSet"):
            iterable_population = population.popSet
        else:
            iterable_population = population
        rules: list[Rule] = []
        n_classes = len(class_labels)

        # Extract classPredictionWeights from the estimator (as in native ExSTraCS).
        # classPredictionWeights[c] = 1 - (count_c / total), weighting minority classes higher.
        class_prediction_weights: dict[Any, float] = {}
        try:
            if hasattr(estimator, "env") and hasattr(estimator.env, "formatData"):
                cpw = getattr(estimator.env.formatData, "classPredictionWeights", None)
                if cpw is not None:
                    class_prediction_weights = dict(cpw)
        except Exception:
            pass

        # Extract attributeInfoType to distinguish discrete vs continuous features.
        attribute_info_type: list[bool] = []
        try:
            if hasattr(estimator, "env") and hasattr(estimator.env, "formatData"):
                ait = getattr(estimator.env.formatData, "attributeInfoType", None)
                if ait is not None:
                    attribute_info_type = list(ait)
        except Exception:
            pass

        # Debug: collect all phenotypes from the population.
        all_phenotypes = set()

        # Process each rule in the population.
        for rule_idx, rule in enumerate(iterable_population):
            atoms: list[Atom] = []

            # Extract conditions with correct feature mapping via specifiedAttList.
            specified_att_list = getattr(rule, "specifiedAttList", None)
            conditions = getattr(rule, "condition", None)

            if specified_att_list is not None and conditions is not None:
                # skExSTraCS format: condition[i] belongs to feature specifiedAttList[i].
                for cond_idx, interval in enumerate(conditions):
                    if cond_idx < len(specified_att_list):
                        feat_idx = specified_att_list[cond_idx]
                        if feat_idx < len(feature_names):
                            feature_name = feature_names[feat_idx]
                        else:
                            feature_name = f"f{feat_idx}"

                        if interval is None or interval == "#":
                            continue

                        # Determine whether feature is continuous or discrete.
                        is_continuous = True
                        if attribute_info_type and feat_idx < len(attribute_info_type):
                            is_continuous = attribute_info_type[feat_idx]

                        new_atoms = _exstracs_condition_to_atoms(
                            interval, feature_name, is_continuous
                        )
                        atoms.extend(new_atoms)
            elif conditions is not None:
                # Fallback: no specifiedAttList available (unexpected format).
                # Iterate over conditions and use index as feature index.
                for feat_idx, interval in enumerate(conditions):
                    if feat_idx < len(feature_names):
                        feature_name = feature_names[feat_idx]
                        if interval is not None and interval != "#":
                            new_atoms = _exstracs_condition_to_atoms(
                                interval, feature_name, True
                            )
                            atoms.extend(new_atoms)

            # Extract target class and fitness.
            class_idx = 0
            fitness = 1.0
            numerosity = 1

            if hasattr(rule, "phenotype"):
                class_label = rule.phenotype
                all_phenotypes.add(class_label)
                try:
                    class_idx = class_labels.index(class_label)
                except (ValueError, IndexError):
                    # Try float comparison (ExSTraCS often stores phenotype as float).
                    found = False
                    for ci, cl in enumerate(class_labels):
                        try:
                            if float(cl) == float(class_label):
                                class_idx = ci
                                found = True
                                break
                        except (ValueError, TypeError):
                            continue
                    if not found:
                        import warnings
                        warnings.warn(
                            f"ExSTraCS rule {rule_idx}: phenotype '{class_label}' "
                            f"not in class_labels {class_labels}. Mapping to 0."
                        )
                        class_idx = 0
            else:
                import warnings
                warnings.warn(f"ExSTraCS rule {rule_idx}: no phenotype found, mapping to 0.")

            if hasattr(rule, "fitness"):
                fitness = float(rule.fitness)
            if hasattr(rule, "numerosity"):
                numerosity = float(rule.numerosity)

            # Score = fitness x numerosity x classPredictionWeight (as in ExSTraCS prediction)
            score_value = fitness * numerosity

            # Apply classPredictionWeight if available.
            phenotype_key = getattr(rule, "phenotype", None)
            if phenotype_key is not None and phenotype_key in class_prediction_weights:
                score_value *= class_prediction_weights[phenotype_key]

            # Build score vector.
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
                        "phenotype": phenotype_key,
                    },
                )
            )

        # Diagnostics: warn if too many rules have no atoms.
        n_no_atoms = sum(1 for r in rules if len(r.atoms) == 0)
        if n_no_atoms > len(rules) * 0.5:
            import warnings
            warnings.warn(
                f"ExSTraCS: {n_no_atoms}/{len(rules)} rules have no atoms! "
                f"Conditions may not be extracted correctly."
            )

        # Default-rule handling: treat defaults like any other rule and combine scores when needed.
        default_rules = [r for r in rules if len(r.atoms) == 0]
        if len(default_rules) == 0:
            # No default rule found: add one with neutral scores.
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
        elif len(default_rules) > 1:
            # Multiple defaults: sum scores and merge into one rule.
            summed_scores = [0.0] * n_classes
            for r in default_rules:
                summed_scores = [a + b for a, b in zip(summed_scores, r.scores)]
            # Remove all default rules.
            rules = [r for r in rules if len(r.atoms) > 0]
            # Insert merged default rule.
            rules.insert(
                0,
                Rule(
                    atoms=[],
                    scores=summed_scores,
                    rule_id="default",
                    metadata={"source": "exstracs_default_combined", "n_combined": len(default_rules)},
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
        raise TypeError(f"Could not transform ExSTraCS to scored rule set: {e}") from e



def _condition_to_atoms(condition: Any, feature_names: list[str]) -> list[Atom]:
    """
    Transform a RuleKit condition into atom(s).
    
    RuleKit conditions can appear in different formats.
    """
    atoms: list[Atom] = []
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
                return []

            if hasattr(value_set, "getLeft") and hasattr(value_set, "getRight"):
                left = float(value_set.getLeft())
                right = float(value_set.getRight())
                left_sign = str(value_set.getLeftSign()) if hasattr(value_set, "getLeftSign") else ">="
                right_sign = str(value_set.getRightSign()) if hasattr(value_set, "getRightSign") else "<="

                left_inf = not math.isfinite(left) or left <= -1e300
                right_inf = not math.isfinite(right) or right >= 1e300

                if left_inf and not right_inf:
                    op = _normalize_rulekit_bound_op(right_sign, is_left=False)
                    return [Atom(feature=str(feature_name), op=op, value=right)]
                if right_inf and not left_inf:
                    op = _normalize_rulekit_bound_op(left_sign, is_left=True)
                    return [Atom(feature=str(feature_name), op=op, value=left)]
                if not left_inf and not right_inf:
                    left_op = _normalize_rulekit_bound_op(left_sign, is_left=True)
                    right_op = _normalize_rulekit_bound_op(right_sign, is_left=False)
                    return [
                        Atom(feature=str(feature_name), op=left_op, value=left),
                        Atom(feature=str(feature_name), op=right_op, value=right),
                    ]
                return []

            if hasattr(value_set, "getValue"):
                return [Atom(feature=str(feature_name), op="==", value=float(value_set.getValue()))]
            if hasattr(value_set, "getValueAsString"):
                value = value_set.getValueAsString()
                try:
                    value = float(value)
                except Exception:
                    value = str(value)
                return [Atom(feature=str(feature_name), op="==", value=value)]

        # Try extracting from common attribute names.
        if hasattr(condition, "attribute") and hasattr(condition, "value"):
            feature_name = condition.attribute
            if isinstance(feature_name, int) and feature_name < len(feature_names):
                feature_name = feature_names[feature_name]
            
            if hasattr(condition, "operator"):
                op = condition.operator
                # Normalize operator.
                if op in ("=", "==", "equals"):
                    return [Atom(feature=str(feature_name), op="==", value=float(condition.value))]
                elif op in ("<", "less"):
                    return [Atom(feature=str(feature_name), op="<", value=float(condition.value))]
                elif op in (">", "greater"):
                    return [Atom(feature=str(feature_name), op=">", value=float(condition.value))]
                elif op in ("<=", "less_or_equal"):
                    return [Atom(feature=str(feature_name), op="<=", value=float(condition.value))]
                elif op in (">=", "greater_or_equal"):
                    return [Atom(feature=str(feature_name), op=">=", value=float(condition.value))]
        
        return atoms
    except Exception:
        return []


def _condition_to_atom(condition: Any, feature_names: list[str]) -> Atom | None:
    """Legacy helper: return first extracted atom if available."""
    atoms = _condition_to_atoms(condition, feature_names)
    return atoms[0] if atoms else None


def _normalize_rulekit_bound_op(sign: str, *, is_left: bool) -> str:
    token = str(sign).strip().lower()
    if token in ("[", "<=", "left_closed", "closed", "inclusive"):
        return ">=" if is_left else "<="
    if token in ("(", "<", "left_open", "open", "exclusive"):
        return ">" if is_left else "<"
    if token in (">", ">="):
        return token
    if token in ("<", "<="):
        return token
    return ">=" if is_left else "<="


def _extract_rulekit_rule_strength(rule: Any) -> float:
    """Estimate rule strength from available RuleKit rule statistics."""
    weight = _safe_positive_float(getattr(rule, "weight", None))
    if weight is None and hasattr(rule, "getWeight"):
        weight = _safe_positive_float(_try_call_no_args(rule.getWeight))

    confidence = _safe_positive_float(getattr(rule, "confidence", None))
    if confidence is None and hasattr(rule, "getConfidence"):
        confidence = _safe_positive_float(_try_call_no_args(rule.getConfidence))

    support = _safe_positive_float(getattr(rule, "support", None))
    if support is None and hasattr(rule, "getSupport"):
        support = _safe_positive_float(_try_call_no_args(rule.getSupport))

    if weight is not None:
        return max(weight, 1e-6)
    if confidence is not None and support is not None:
        return max(confidence * support, 1e-6)
    if confidence is not None:
        return max(confidence, 1e-6)
    if support is not None:
        return max(support, 1e-6)
    return 1.0


def _rulekit_default_scores(
    *,
    estimator: Any,
    class_labels: list[Any],
    y_train: np.ndarray | None,
) -> list[float]:
    """Build default scores using class priors to mimic RuleKit fallback behaviour."""
    counts = np.zeros(len(class_labels), dtype=float)
    labels_array = np.asarray(class_labels)

    if y_train is not None:
        y_arr = np.asarray(y_train).reshape(-1)
        for idx, label in enumerate(labels_array):
            counts[idx] = float(np.sum(y_arr == label))

    if float(np.sum(counts)) <= 0.0:
        class_hist = getattr(estimator, "class_counts_", None)
        if class_hist is not None and isinstance(class_hist, dict):
            for idx, label in enumerate(labels_array):
                if label in class_hist:
                    counts[idx] = float(class_hist[label])

    total = float(np.sum(counts))
    if total <= 0.0:
        return [1.0 / max(len(class_labels), 1)] * len(class_labels)

    priors = counts / total
    return priors.tolist()


def _safe_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _try_call_no_args(fn: Any) -> Any:
    try:
        return fn()
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


def _exstracs_condition_to_atoms(
    condition: Any, feature_name: str, is_continuous: bool
) -> list[Atom]:
    """
        Transform an ExSTraCS condition into atom(s).
    
        For continuous features:
            ExSTraCS uses intervals [lower, upper] with strict inequality:
        lower < value < upper
            -> two atoms: feature > lower AND feature < upper
    
        For discrete features:
            ExSTraCS uses equality: value == condition
            -> one atom: feature == value
    """
    atoms: list[Atom] = []
    try:
        # Check whether this is an interval (list, tuple, or 2-element numpy array).
        is_array_like = isinstance(condition, (tuple, list))
        if not is_array_like:
            try:
                # Support numpy-like arrays.
                if hasattr(condition, "__len__") and hasattr(condition, "__getitem__"):
                    is_array_like = True
            except Exception:
                pass

        if is_continuous and is_array_like and len(condition) == 2:
            lower = float(condition[0])
            upper = float(condition[1])
            # Ensure lower <= upper.
            if lower > upper:
                lower, upper = upper, lower
            # ExSTraCS uses strict inequality: lower < value < upper.
            atoms.append(Atom(feature=str(feature_name), op=">", value=lower))
            atoms.append(Atom(feature=str(feature_name), op="<", value=upper))
        elif not is_continuous:
            # Discrete feature: equality.
            atoms.append(Atom(feature=str(feature_name), op="==", value=condition))
        elif is_array_like and len(condition) == 2:
            # Fallback: treat as interval.
            lower = float(condition[0])
            upper = float(condition[1])
            if lower > upper:
                lower, upper = upper, lower
            atoms.append(Atom(feature=str(feature_name), op=">", value=lower))
            atoms.append(Atom(feature=str(feature_name), op="<", value=upper))
        else:
            # Scalar value: treat as equality.
            atoms.append(Atom(feature=str(feature_name), op="==", value=condition))

    except Exception:
        pass
    return atoms


def _interval_to_atom(interval: Any, feature_name: str) -> Atom | None:
    """
    Transform ExSTraCS interval to atom (legacy helper).
    
    Prefer _exstracs_condition_to_atoms for correct strict inequalities.
    """
    atoms = _exstracs_condition_to_atoms(interval, feature_name, is_continuous=True)
    return atoms[0] if atoms else None

