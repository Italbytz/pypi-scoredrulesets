"""General warmstart extraction and curation strategies for RuleGP and RuleNSGA-II."""

from __future__ import annotations

import re
import numpy as np
from typing import Any


def _parse_feature_idx(feat_str: str, feature_names: list[str]) -> int:
    """Resolve a feature string into an integer index."""
    if feat_str in feature_names:
        return feature_names.index(feat_str)
    m = re.match(r"^X_?(\d+)$", str(feat_str))
    if m:
        return int(m.group(1))
    m2 = re.match(r"^f_(\d+)$", str(feat_str))
    if m2:
        return int(m2.group(1))
    raise ValueError(f"Cannot resolve feature '{feat_str}' to an index.")


def _espresso_expand_rule(
    rule_atoms: list[tuple[int, str, float]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    min_purity_loss: float = 0.02,
) -> list[tuple[int, str, float]]:
    """Prunes non-essential literals from a rule conjunction while preserving class precision."""
    if len(rule_atoms) <= 1:
        return rule_atoms

    def _eval(atoms: list[tuple[int, str, float]]) -> tuple[float, int, int]:
        m = np.ones(X_train.shape[0], dtype=bool)
        for fi, op, thr in atoms:
            col = X_train[:, fi]
            if op == "<=":
                m &= (col <= thr)
            elif op == "<":
                m &= (col < thr)
            elif op == ">=":
                m &= (col >= thr)
            elif op == ">":
                m &= (col > thr)
        if not m.any():
            return 0.0, 0, 0
        counts = np.bincount(y_train[m])
        dom_class = int(np.argmax(counts))
        purity = float(counts[dom_class] / m.sum())
        return purity, int(m.sum()), dom_class

    orig_purity, orig_cov, orig_class = _eval(rule_atoms)
    current_atoms = list(rule_atoms)
    changed = True
    while changed and len(current_atoms) > 1:
        changed = False
        for idx in range(len(current_atoms)):
            candidate = current_atoms[:idx] + current_atoms[idx + 1:]
            purity, cov, dom_class = _eval(candidate)
            if dom_class == orig_class and purity >= orig_purity - min_purity_loss and cov >= orig_cov:
                current_atoms = candidate
                orig_purity = purity
                orig_cov = cov
                changed = True
                break
    return current_atoms


def _cluster_thresholds(
    splits_per_feature: dict[int, list[tuple[float, float]]],
    X_train: np.ndarray,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 3,
) -> list[tuple[int, str, float]]:
    """Sample-based clustering of split thresholds per feature."""
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

    # Deduplicate atoms
    unique_atoms: dict[tuple[int, str, float], tuple[int, str, float]] = {}
    for fi, op, thr in curated_atoms:
        key = (fi, op, round(thr, 6))
        if key not in unique_atoms:
            unique_atoms[key] = (fi, op, thr)
    return list(unique_atoms.values())


def _filter_rules_jaccard(
    candidate_rules: list[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_rules_seed: int = 15,
    jaccard_max_sim: float = 0.8,
    espresso_expand_seeds: bool = False,
) -> list[list[tuple[int, str, float]]]:
    """Applies Jaccard coverage filter and optional Espresso expansion to candidate rules."""
    n_samples = X_train.shape[0]
    candidate_rules.sort(key=lambda r: r.get("importance", 1.0), reverse=True)

    selected_rules: list[list[tuple[int, str, float]]] = []
    selected_masks: list[np.ndarray] = []

    for r_dict in candidate_rules:
        rule_atoms = r_dict["atoms"]
        if not rule_atoms:
            continue
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

        is_redundant = False
        for prev_mask in selected_masks:
            intersection = int(np.sum(mask & prev_mask))
            union = int(np.sum(mask | prev_mask))
            if union > 0 and (intersection / union) > jaccard_max_sim:
                is_redundant = True
                break

        if not is_redundant:
            rule_to_add = (
                _espresso_expand_rule(rule_atoms, X_train, y_train)
                if espresso_expand_seeds
                else rule_atoms
            )
            selected_masks.append(mask)
            selected_rules.append(rule_to_add)
            if len(selected_rules) >= max_rules_seed:
                break

    return selected_rules


def _expand_ladder(
    selected_rules: list[list[tuple[int, str, float]]],
    max_rungs: int = 60,
) -> list[list[tuple[int, str, float]]]:
    """Expands each seed path into a size ladder of its prefixes.

    Tree paths are ordered root-to-leaf, so every proper prefix of a path is
    itself a valid internal-node rule.  Emitting all prefixes (deduplicated
    across seeds, importance order preserved) makes the (F1, size) front
    spannable by construction: within one ladder, larger sizes are strictly
    more specific rules of the same lineage.
    """
    rungs: list[list[tuple[int, str, float]]] = []
    seen: set[tuple] = set()
    for rule in selected_rules:
        for k in range(len(rule), 0, -1):
            key = tuple(rule[:k])
            if key not in seen:
                seen.add(key)
                rungs.append(list(rule[:k]))
                if len(rungs) >= max_rungs:
                    return rungs
    return rungs


def extract_rulefit_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_rules: int = 30,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 3,
    max_rules_seed: int = 15,
    jaccard_max_sim: float = 0.8,
    espresso_expand_seeds: bool = False,
    random_state: int | None = None,
    return_estimator: bool = False,
) -> tuple[Any, ...]:
    """Fits RuleFit and extracts curated split atoms and rule seed conjunctions."""
    try:
        from imodels import RuleFitClassifier
    except ImportError as exc:
        raise ImportError("imodels is required for RuleFit warmstart.") from exc

    n_samples, n_features = X_train.shape
    rf = RuleFitClassifier(max_rules=max_rules, random_state=random_state)
    rf.fit(X_train, y_train, feature_names=feature_names)

    raw_rules = getattr(rf, "rules_", [])
    if not raw_rules:
        return ([], [], rf) if return_estimator else ([], [])

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
                "importance": abs_coeff,
            })

    curated_atoms = _cluster_thresholds(
        splits_per_feature, X_train, min_samples_split, max_thresholds_per_feature
    )

    # Linear term extraction
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

    # Re-deduplicate
    unique = {}
    for fi, op, thr in curated_atoms:
        k = (fi, op, round(thr, 6))
        if k not in unique:
            unique[k] = (fi, op, thr)
    curated_atom_list = list(unique.values())

    selected_rules = _filter_rules_jaccard(
        active_rules_parsed, X_train, y_train, max_rules_seed, jaccard_max_sim, espresso_expand_seeds
    )

    if return_estimator:
        return curated_atom_list, selected_rules, rf
    return curated_atom_list, selected_rules


def extract_extratrees_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_rules: int = 30,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 3,
    max_rules_seed: int = 15,
    jaccard_max_sim: float = 0.8,
    espresso_expand_seeds: bool = False,
    random_state: int | None = None,
    return_estimator: bool = False,
    scouting_max_depth: int = 3,
) -> tuple[Any, ...]:
    """Fits ExtraTrees and extracts curated split atoms and rule seed conjunctions."""
    from sklearn.ensemble import ExtraTreesClassifier

    n_samples, n_features = X_train.shape
    et = ExtraTreesClassifier(
        n_estimators=max_rules,
        max_depth=scouting_max_depth,
        max_features="sqrt",
        random_state=random_state,
    )
    et.fit(X_train, y_train)

    splits_per_feature: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_features)}
    candidate_rules = []

    for tree_est in et.estimators_:
        t = tree_est.tree_

        def _traverse(node_id: int, path: list[tuple[int, str, float]]):
            left = t.children_left[node_id]
            right = t.children_right[node_id]
            if left == -1 and right == -1:
                counts = t.value[node_id][0]
                total = float(np.sum(counts))
                purity = float(np.max(counts) / total) if total > 0 else 0.5
                n_node_samples = float(t.n_node_samples[node_id])
                imp = n_node_samples * purity
                if path:
                    candidate_rules.append({"atoms": path, "importance": imp})
                return

            fi = int(t.feature[node_id])
            thr = float(t.threshold[node_id])
            # Weight split by samples passing through it
            splits_per_feature[fi].append((thr, float(t.n_node_samples[node_id])))
            if left != -1:
                _traverse(left, path + [(fi, "<=", thr)])
            if right != -1:
                _traverse(right, path + [(fi, ">", thr)])

        _traverse(0, [])

    curated_atoms = _cluster_thresholds(
        splits_per_feature, X_train, min_samples_split, max_thresholds_per_feature
    )
    selected_rules = _filter_rules_jaccard(
        candidate_rules, X_train, y_train, max_rules_seed, jaccard_max_sim, espresso_expand_seeds
    )

    if return_estimator:
        return curated_atoms, selected_rules, et
    return curated_atoms, selected_rules


def extract_figs_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_rules: int = 15,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 3,
    max_rules_seed: int = 15,
    jaccard_max_sim: float = 0.8,
    espresso_expand_seeds: bool = False,
    random_state: int | None = None,
    return_estimator: bool = False,
) -> tuple[Any, ...]:
    """Fits FIGS and extracts curated split atoms and rule seed conjunctions."""
    try:
        from imodels import FIGSClassifier
    except ImportError as exc:
        raise ImportError("imodels is required for FIGS warmstart.") from exc

    n_samples, n_features = X_train.shape
    figs = FIGSClassifier(max_rules=max_rules, random_state=random_state)
    figs.fit(X_train, y_train)

    splits_per_feature: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_features)}
    candidate_rules = []

    def _traverse(node: Any, path: list[tuple[int, str, float]]):
        if getattr(node, "left", None) is None and getattr(node, "right", None) is None:
            if path:
                n_samples_node = float(getattr(node, "n_samples_", 10))
                val = getattr(node, "value", [0.5, 0.5])
                purity = float(np.max(val)) if hasattr(val, "__iter__") else 0.5
                candidate_rules.append({"atoms": path, "importance": n_samples_node * purity})
            return

        fi = int(node.feature)
        thr = float(node.threshold)
        red = float(getattr(node, "impurity_reduction", 1.0))
        splits_per_feature[fi].append((thr, max(red, 0.01)))
        if getattr(node, "left", None) is not None:
            _traverse(node.left, path + [(fi, "<=", thr)])
        if getattr(node, "right", None) is not None:
            _traverse(node.right, path + [(fi, ">", thr)])

    for t in getattr(figs, "trees_", []):
        _traverse(t, [])

    curated_atoms = _cluster_thresholds(
        splits_per_feature, X_train, min_samples_split, max_thresholds_per_feature
    )
    selected_rules = _filter_rules_jaccard(
        candidate_rules, X_train, y_train, max_rules_seed, jaccard_max_sim, espresso_expand_seeds
    )

    if return_estimator:
        return curated_atoms, selected_rules, figs
    return curated_atoms, selected_rules


def extract_l1_logistic_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_rules: int = 30,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 3,
    max_rules_seed: int = 15,
    jaccard_max_sim: float = 0.8,
    espresso_expand_seeds: bool = False,
    random_state: int | None = None,
    return_estimator: bool = False,
) -> tuple[Any, ...]:
    """Fits L1 Logistic Regression and extracts 1D split atoms and rules."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    n_samples, n_features = X_train.shape
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.5,
            random_state=random_state,
            max_iter=1000,
        )
        lr.fit(X_train, y_train)

    coef = lr.coef_.ravel()
    active_indices = np.where(np.abs(coef) > 1e-4)[0]
    # If no features active, take top features with largest abs coefficients
    if len(active_indices) == 0:
        active_indices = np.argsort(np.abs(coef))[::-1][:min(10, n_features)]

    splits_per_feature: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_features)}
    candidate_rules = []

    for fi in active_indices[:min(max_rules, len(active_indices))]:
        weight = float(abs(coef[fi])) if fi < len(coef) else 1.0
        # Optimal 1D stump split
        stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
        stump.fit(X_train[:, [fi]], y_train)
        thr = float(stump.tree_.threshold[0])
        splits_per_feature[fi].append((thr, weight))

        candidate_rules.append({"atoms": [(fi, "<=", thr)], "importance": weight})
        candidate_rules.append({"atoms": [(fi, ">", thr)], "importance": weight})

    curated_atoms = _cluster_thresholds(
        splits_per_feature, X_train, min_samples_split, max_thresholds_per_feature
    )
    selected_rules = _filter_rules_jaccard(
        candidate_rules, X_train, y_train, max_rules_seed, jaccard_max_sim, espresso_expand_seeds
    )

    if return_estimator:
        return curated_atoms, selected_rules, lr
    return curated_atoms, selected_rules


def extract_ensemble_rich_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_rules: int = 30,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 5,
    max_rules_seed: int = 200,
    jaccard_max_sim: float = 0.7,
    espresso_expand_seeds: bool = False,
    random_state: int | None = None,
    return_estimator: bool = False,
    top_k_features: int = 30,
    n_quantiles: int = 20,
    pareto_parsimony_fraction: float = 0.25,
    pareto_accuracy_fraction: float = 0.25,
    pareto_multirule_fraction: float = 0.25,
    scouting_max_depth: int = 4,
) -> tuple[Any, ...]:
    """Ensemble-rich warmstart: FS-level atom pool + Pareto-diverse population seeds.

    This strategy addresses the two core weaknesses of standard warmstart:

    1. **Restricted atom pool** (the 'atom-pool hijacking' problem):
       Instead of replacing the GP atom pool with the ensemble's specific split
       thresholds, we use the ensemble only for *feature ranking* and then build
       the *full* GP atom pool for those top-k features from the training data
       using quantile-based splits.  This gives ~k * n_quantiles * 2 atoms
       (e.g. 30 features × 20 quantiles × 2 directions = 1200 atoms) instead of
       the ~30 atoms produced by standard warmstart.

    2. **Lack of initial population diversity** (the 'Rule Seeds have zero effect'
       problem): Instead of keeping only 15 Jaccard-filtered paths as seeds, we
       explicitly construct a *Pareto-front-spanning* initial population:

       - **Parsimony-Pol** (``pareto_parsimony_fraction``):
         One 1-atom stump per top feature at its best univariate threshold.
         These are minimal (1 rule, 1 condition) and cover the high-parsimony
         end of the Pareto front from the start.

       - **Accuracy-Pol** (``pareto_accuracy_fraction``):
         Full-depth tree paths (all atoms in the path) sampled from ensemble
         leaves, filtered for diversity.  These are complex but highly accurate
         and anchor the high-accuracy end of the Pareto front.

       - **Mid-Range** (remaining fraction):
         2-3 atom rules from ensemble paths — the natural "sweet spot" that the
         GP typically converges to.

       - **Multi-Rule individuals** (``pareto_multirule_fraction``):
         Two complementary rules combined into a single rule set: one rule
         targeting class-1 (e.g. high purity for positives) and one targeting
         class-0 (high purity for negatives).  This immediately covers both
         sides of the decision boundary.

    The atom pool returned by this function is NOT the ensemble's specific
    thresholds but the *full* quantile-based pool for the top-k features.
    The returned seeds populate the initial population with diverse individuals.

    Parameters
    ----------
    top_k_features : int
        Number of top features selected by ensemble importance. Default 30.
    n_quantiles : int
        Number of quantile breakpoints per feature for atom pool construction.
        Default 20 → ~1200 atoms for top_k=30.
    pareto_parsimony_fraction : float
        Fraction of seed budget allocated to 1-atom parsimony anchors.
    pareto_accuracy_fraction : float
        Fraction of seed budget allocated to full-depth accuracy anchors.
    pareto_multirule_fraction : float
        Fraction of seed budget allocated to multi-rule complementary individuals.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(random_state)
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    n_classes = len(classes)

    # ------------------------------------------------------------------
    # Step 1: Fit ensemble for feature ranking + path extraction
    # ------------------------------------------------------------------
    et = ExtraTreesClassifier(
        n_estimators=max(max_rules, 50),
        max_depth=scouting_max_depth,
        max_features="sqrt",
        random_state=random_state,
    )
    et.fit(X_train, y_train)

    importances = et.feature_importances_
    # Top-k features by ensemble importance
    top_k = min(top_k_features, n_features)
    top_feat_indices = np.argsort(importances)[::-1][:top_k].tolist()

    # ------------------------------------------------------------------
    # Step 2: Build FULL quantile-based atom pool for top-k features
    #         (not restricted to the ensemble's specific thresholds)
    # ------------------------------------------------------------------
    full_atoms: list[tuple[int, str, float]] = []
    for fi in top_feat_indices:
        col = X_train[:, fi]
        unique_vals = np.unique(col)
        if len(unique_vals) <= 1:
            continue
        quantiles = np.quantile(unique_vals, np.linspace(0.05, 0.95, n_quantiles))
        quantiles = np.unique(quantiles)
        for thr in quantiles:
            full_atoms.append((fi, "<=", float(thr)))
            full_atoms.append((fi, ">", float(thr)))

    # ------------------------------------------------------------------
    # Step 3: Extract ALL leaf-path rules from ensemble trees
    # ------------------------------------------------------------------
    all_leaf_paths: list[dict[str, Any]] = []

    def _traverse(tree, node_id: int, path: list[tuple[int, str, float]]):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == -1:  # leaf
            counts = tree.value[node_id][0]
            total = float(np.sum(counts))
            if total < min_samples_split:
                return
            dom_class = int(np.argmax(counts))
            purity = float(counts[dom_class] / total) if total > 0 else 0.5
            # Only keep rules for features in the top-k set
            clean_path = [(fi, op, thr) for fi, op, thr in path if fi in top_feat_indices]
            if clean_path:
                all_leaf_paths.append({
                    "atoms": clean_path,
                    "full_path": path,
                    "purity": purity,
                    "n_samples": total,
                    "dom_class": dom_class,
                    "depth": len(path),
                    "importance": total * purity,
                })
            return
        fi = int(tree.feature[node_id])
        thr = float(tree.threshold[node_id])
        if left != -1:
            _traverse(tree, left, path + [(fi, "<=", thr)])
        if right != -1:
            _traverse(tree, right, path + [(fi, ">", thr)])

    for tree_est in et.estimators_:
        _traverse(tree_est.tree_, 0, [])

    # Sort by importance descending
    all_leaf_paths.sort(key=lambda r: r["importance"], reverse=True)

    # ------------------------------------------------------------------
    # Step 4: Build Pareto-diverse seed population
    # ------------------------------------------------------------------
    n_budget = max_rules_seed
    n_parsimony = max(1, int(n_budget * pareto_parsimony_fraction))
    n_accuracy  = max(1, int(n_budget * pareto_accuracy_fraction))
    n_multirule = max(1, int(n_budget * pareto_multirule_fraction))
    n_midrange  = n_budget - n_parsimony - n_accuracy - n_multirule

    seed_rules: list[list[tuple[int, str, float]]] = []

    # --- Parsimony-Pol: 1-atom stumps for each top feature ---
    parsimony_added = 0
    for fi in top_feat_indices:
        if parsimony_added >= n_parsimony:
            break
        stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
        stump.fit(X_train[:, [fi]], y_train)
        thr = float(stump.tree_.threshold[0])
        if thr < -1e10:  # leaf node, no split
            continue
        seed_rules.append([(fi, "<=", thr)])
        seed_rules.append([(fi, ">", thr)])
        parsimony_added += 2
        if parsimony_added >= n_parsimony:
            break

    # --- Accuracy-Pol: full-depth paths (depth >= 3) ---
    accuracy_candidates = [r for r in all_leaf_paths if r["depth"] >= 3 and r["purity"] >= 0.8]
    accuracy_selected = _filter_rules_jaccard(
        accuracy_candidates, X_train, y_train,
        max_rules_seed=n_accuracy, jaccard_max_sim=jaccard_max_sim,
    )
    seed_rules.extend(accuracy_selected)

    # --- Mid-range: 2-atom rules ---
    mid_candidates = [r for r in all_leaf_paths if r["depth"] == 2]
    mid_selected = _filter_rules_jaccard(
        mid_candidates, X_train, y_train,
        max_rules_seed=n_midrange, jaccard_max_sim=jaccard_max_sim,
    )
    seed_rules.extend(mid_selected)

    # --- Multi-rule individuals: pair complementary class rules ---
    # Find best rule per class and combine into a 2-rule individual.
    # We encode multi-rule individuals as a list of lists (each inner list = one rule).
    multirule_seeds: list[list[list[tuple[int, str, float]]]] = []
    rules_per_class: dict[int, list[dict]] = {int(c): [] for c in classes}
    for r in all_leaf_paths:
        dc = r["dom_class"]
        if dc in rules_per_class:
            rules_per_class[dc].append(r)

    # Take the top n_multirule complementary pairs
    cls_0_rules = rules_per_class.get(0, [])[:n_multirule * 2]
    cls_1_rules = rules_per_class.get(1, [])[:n_multirule * 2]
    for i in range(min(n_multirule, len(cls_0_rules), len(cls_1_rules))):
        r0 = cls_0_rules[i]["atoms"]
        r1 = cls_1_rules[i]["atoms"]
        if r0 and r1:
            multirule_seeds.append([r0, r1])

    if return_estimator:
        return full_atoms, seed_rules, multirule_seeds, et
    return full_atoms, seed_rules, multirule_seeds


def extract_warmstart_components(
    strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_rules: int = 30,
    min_samples_split: int = 2,
    max_thresholds_per_feature: int = 3,
    max_rules_seed: int = 15,
    jaccard_max_sim: float = 0.8,
    espresso_expand_seeds: bool = False,
    random_state: int | None = None,
    return_estimator: bool = False,
    ladder: bool = False,
    ladder_max_rungs: int = 60,
    scouting_max_depth: int | None = None,
) -> tuple[Any, ...]:
    """Unified dispatcher for all warmstart extraction strategies.

    With ``ladder=True`` the returned seed rules are expanded into the size
    ladders of their prefixes (see ``_expand_ladder``), so the initial
    population spans several model sizes per seed lineage instead of only
    full-depth leaf paths.
    """
    strat = strategy.lower().strip()
    if strat in ("rulefit", "rulefit_atoms_only", "rulefit_seeds_only"):
        result = extract_rulefit_components(
            X_train, y_train, feature_names, max_rules, min_samples_split,
            max_thresholds_per_feature, max_rules_seed, jaccard_max_sim,
            espresso_expand_seeds, random_state, return_estimator
        )
    elif strat in ("extratrees", "extratrees_atoms_only", "extratrees_seeds_only"):
        result = extract_extratrees_components(
            X_train, y_train, feature_names, max_rules, min_samples_split,
            max_thresholds_per_feature, max_rules_seed, jaccard_max_sim,
            espresso_expand_seeds, random_state, return_estimator,
            **({} if scouting_max_depth is None else {"scouting_max_depth": scouting_max_depth}),
        )
    elif strat in ("figs", "figs_atoms_only", "figs_seeds_only"):
        result = extract_figs_components(
            X_train, y_train, feature_names, min(max_rules, 15), min_samples_split,
            max_thresholds_per_feature, max_rules_seed, jaccard_max_sim,
            espresso_expand_seeds, random_state, return_estimator
        )
    elif strat in ("l1_logistic", "l1_logistic_atoms_only", "l1_logistic_seeds_only", "linear_l1"):
        result = extract_l1_logistic_components(
            X_train, y_train, feature_names, max_rules, min_samples_split,
            max_thresholds_per_feature, max_rules_seed, jaccard_max_sim,
            espresso_expand_seeds, random_state, return_estimator
        )
    elif strat in ("ensemble_rich", "ensemble_rich_atoms_only", "ensemble_rich_seeds_only"):
        result = extract_ensemble_rich_components(
            X_train, y_train, feature_names, max_rules, min_samples_split,
            max_thresholds_per_feature, max_rules_seed, jaccard_max_sim,
            espresso_expand_seeds, random_state, return_estimator,
            **({} if scouting_max_depth is None else {"scouting_max_depth": scouting_max_depth}),
        )
    else:
        raise ValueError(
            f"Unknown warmstart strategy '{strategy}'. "
            f"Supported: 'rulefit', 'extratrees', 'figs', 'l1_logistic', 'ensemble_rich'."
        )
    if ladder:
        seed_rules = _expand_ladder(list(result[1]), max_rungs=ladder_max_rungs)
        result = (result[0], seed_rules) + result[2:]
    return result
