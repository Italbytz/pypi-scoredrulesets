"""RuleFit extraction and curation for RuleGP warmstart and atom selection."""

from __future__ import annotations

import re
import numpy as np
from typing import Any


def _parse_feature_idx(feat_str: str, feature_names: list[str]) -> int:
    """Resolve a feature string from imodels Rule into an integer index."""
    if feat_str in feature_names:
        return feature_names.index(feat_str)
    m = re.match(r"^X_?(\d+)$", feat_str)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot resolve feature '{feat_str}' to an index.")


def extract_rulefit_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_rules: int = 30,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 3,
    max_rules_seed: int = 15,
    jaccard_max_sim: float = 0.8,
    random_state: int | None = None,
) -> tuple[list[tuple[int, str, float]], list[list[tuple[int, str, float]]]]:
    """
    Fits a RuleFitClassifier and extracts curated split atoms and rule seed conjunctions.

    Returns
    -------
    curated_atoms : list of (feature_idx, op, threshold)
    curated_rule_conjunctions : list of rules, where each rule is a list of (feature_idx, op, threshold)
    """
    try:
        from imodels import RuleFitClassifier
    except ImportError as exc:
        raise ImportError(
            "imodels is required for warmstart_strategy='rulefit'. "
            "Please install it via 'pip install imodels'."
        ) from exc

    n_samples, n_features = X_train.shape
    rf = RuleFitClassifier(max_rules=max_rules, random_state=random_state)
    rf.fit(X_train, y_train, feature_names=feature_names)

    raw_rules = getattr(rf, "rules_", [])
    if not raw_rules:
        return [], []

    # 1. Parse active rules and split terms
    active_rules_parsed = []
    splits_per_feature: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_features)}

    for r in raw_rules:
        coeff = float(r.args[0]) if hasattr(r, "args") and len(r.args) > 0 else 1.0
        abs_coeff = abs(coeff)
        if abs_coeff < 1e-6:
            continue

        terms = getattr(r, "terms", [])
        parsed_atoms = []
        for t in terms:
            f_str, op, val_str = t[0], t[1], t[2]
            try:
                fi = _parse_feature_idx(f_str, feature_names)
            except ValueError:
                continue
            thr = float(val_str)
            parsed_atoms.append((fi, op, thr))
            splits_per_feature[fi].append((thr, abs_coeff))

        if parsed_atoms:
            active_rules_parsed.append({
                "atoms": parsed_atoms,
                "abs_coeff": abs_coeff,
            })

    # 2. Sample-based clustering of split thresholds per feature
    curated_atoms: list[tuple[int, str, float]] = []
    for fi, thr_weights in splits_per_feature.items():
        if not thr_weights:
            continue
        thr_weights.sort(key=lambda tw: tw[0])
        col_vals = np.sort(X_train[:, fi])

        clusters: list[list[tuple[float, float]]] = []
        for thr, weight in thr_weights:
            if not clusters:
                clusters.append([(thr, weight)])
            else:
                last_thr = clusters[-1][-1][0]
                n_between = int(np.sum((col_vals > min(last_thr, thr)) & (col_vals < max(last_thr, thr))))
                if n_between < min_samples_split:
                    clusters[-1].append((thr, weight))
                else:
                    clusters.append([(thr, weight)])

        rep_thresholds = []
        for cl in clusters:
            best_thr = max(cl, key=lambda tw: tw[1])[0]
            cum_weight = sum(tw[1] for tw in cl)
            rep_thresholds.append((best_thr, cum_weight))

        rep_thresholds.sort(key=lambda tw: tw[1], reverse=True)
        for rep_thr, _ in rep_thresholds[:max_thresholds_per_feature]:
            curated_atoms.append((fi, "<=", float(rep_thr)))
            curated_atoms.append((fi, ">", float(rep_thr)))

    # Linear term extraction (Decision 7: add median-splits for features with active linear terms)
    if hasattr(rf, "_estimator") and hasattr(rf._estimator, "coef_"):
        coef = np.asarray(rf._estimator.coef_).ravel()
        if len(coef) >= n_features:
            linear_coefs = coef[:n_features]
            top_linear_indices = np.argsort(np.abs(linear_coefs))[::-1]
            for fi in top_linear_indices[:min(5, n_features)]:
                if abs(linear_coefs[fi]) > 1e-4 and fi not in splits_per_feature:
                    median_val = float(np.median(X_train[:, fi]))
                    curated_atoms.append((fi, "<=", median_val))
                    curated_atoms.append((fi, ">", median_val))

    # Deduplicate atoms
    unique_atoms: dict[tuple[int, str, float], tuple[int, str, float]] = {}
    for fi, op, thr in curated_atoms:
        key = (fi, op, round(thr, 6))
        if key not in unique_atoms:
            unique_atoms[key] = (fi, op, thr)
    curated_atom_list = list(unique_atoms.values())

    # 3. Seed rules generation with Jaccard coverage filter
    active_rules_parsed.sort(key=lambda r: r["abs_coeff"], reverse=True)
    selected_rules: list[list[tuple[int, str, float]]] = []
    selected_masks: list[np.ndarray] = []

    for r_dict in active_rules_parsed:
        # Calculate mask for this rule
        rule_atoms = r_dict["atoms"]
        mask = np.ones(n_samples, dtype=bool)
        for fi, op, thr in rule_atoms:
            col = X_train[:, fi]
            if op == "<=":
                mask &= (col <= thr)
            elif op == "<":
                mask &= (col < thr)
            elif op == ">=":
                mask &= (col >= thr)
            elif op == ">":
                mask &= (col > thr)

        if not mask.any():
            continue

        # Jaccard similarity check against existing seeds
        is_redundant = False
        for prev_mask in selected_masks:
            intersection = np.sum(mask & prev_mask)
            union = np.sum(mask | prev_mask)
            if union > 0 and (intersection / union) > jaccard_max_sim:
                is_redundant = True
                break

        if not is_redundant:
            selected_masks.append(mask)
            selected_rules.append(rule_atoms)
            if len(selected_rules) >= max_rules_seed:
                break

    return curated_atom_list, selected_rules
