"""
ExSTraCS rule shrinking and filtering.

This module provides algorithms to reduce ExSTraCS rule populations:

1. Conservative rule pruning: only safe atom removals
2. Aggressive rule pruning: uses validation data, accepts small F1 drops
3. Rule filtering: removes weak rules based on fitness/score
4. Rule consolidation: merges similar rules
5. Lossy Rule Compaction (LRC): interval merge + conservative pruning,
    a non-equivalent compaction that typically has low F1 loss
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score

from ..schema import Rule, ScoredRuleSet


@dataclass
class ExSTraCSPruningParams:
    """Parameters for ExSTraCS rule shrinking."""
    
    # Conservative pruning (guarantees no degradation)
    conservative_prune: bool = False
    
    # Aggressive pruning (with validation data)
    aggressive_prune: bool = False
    max_f1_loss: float = 0.01  # Accept up to 1% F1 loss
    
    # Rule Filtering
    filter_weak_rules: bool = False
    min_fitness_percentile: float = 0.2  # Keep top 80% only
    
    # Rule Consolidation
    consolidate_similar: bool = False
    similarity_threshold: float = 0.8  # Merge if > 80% similar

    # Lossy Rule Compaction (LRC): non-equivalent compaction by
    # merging rules with overlapping feature intervals and
    # the same class. Scores are summed (fitness x numerosity),
    # intervals are unioned. Typically 0-6% F1 loss with 29-98%
    # reduction in rule count.
    interval_merge: bool = False
    interval_merge_iou_threshold: float = 0.3  # Minimum IoU for merge (LRC)

    # Safety limits for large populations
    max_pruning_seconds: float = 120.0  # Timeout for pruning loops (seconds)
    pre_filter_threshold: int = 100  # Auto-filter if rule count exceeds this value


def exstracs_prune_conservative(
    ruleset: ScoredRuleSet,
    X_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    *,
    max_seconds: float = 120.0,
) -> ScoredRuleSet:
    """
    Conservative atom pruning for ExSTraCS rules.
    
    Removes only atoms that are guaranteed not to change predictions
    (similar to the tree-based algorithm).
    
    Safety criteria:
    - Fewer atoms
    - Scores remain positive
    - No complete collapse

    Parameters
    ----------
    max_seconds : float
        Maximum runtime for the pruning loop (seconds).
        If the budget is exceeded, remaining rules are kept unchanged.
    """
    from ..runtime import predict as predict_from_ruleset

    pruned_rules: list[Rule] = []

    baseline_f1: float | None = None
    if X_ref is not None and y_ref is not None:
        y_pred_baseline = predict_from_ruleset(ruleset, X_ref)
        baseline_f1 = f1_score(y_ref, y_pred_baseline, average='macro', zero_division=0)

    t_start = time.monotonic()
    timed_out = False
    
    for rule_idx, rule in enumerate(ruleset.rules):
        # Timeout check (once per rule)
        if time.monotonic() - t_start > max_seconds:
            timed_out = True
            # Keep remaining rules unchanged
            pruned_rules.extend(ruleset.rules[rule_idx:])
            break

        if not rule.atoms:  # Skip default rule
            pruned_rules.append(rule)
            continue
        
        # Try removing atoms (reverse order)
        changed = True
        current_rule = rule
        
        while changed:
            changed = False

            # Also check timeout in the inner loop
            if time.monotonic() - t_start > max_seconds:
                timed_out = True
                break
            
            for atom_idx in range(len(current_rule.atoms) - 1, -1, -1):
                candidate_atoms = current_rule.atoms[:atom_idx] + current_rule.atoms[atom_idx + 1:]
                
                # Check safety criteria
                # IMPORTANT: non-default rules must never become atoms=[].
                if not (len(candidate_atoms) < len(current_rule.atoms) and
                        any(s > 0 for s in current_rule.scores) and
                        len(candidate_atoms) > 0):
                    continue

                candidate_rule = Rule(
                    atoms=candidate_atoms,
                    scores=current_rule.scores,
                    rule_id=current_rule.rule_id,
                    metadata=current_rule.metadata,
                )

                # Without reference data: structurally conservative, but no F1 guarantee.
                if baseline_f1 is None:
                    current_rule = candidate_rule
                    changed = True
                    break

                # With reference data: accept only if F1 does not get worse.
                candidate_rules = pruned_rules + [candidate_rule] + ruleset.rules[len(pruned_rules) + 1:]
                candidate_ruleset = _create_pruned_ruleset(ruleset, candidate_rules, "conservative_prune")
                y_pred_candidate = predict_from_ruleset(candidate_ruleset, X_ref)
                f1_candidate = f1_score(y_ref, y_pred_candidate, average='macro', zero_division=0)
                if f1_candidate + 1e-12 >= baseline_f1:
                    current_rule = candidate_rule
                    baseline_f1 = f1_candidate
                    changed = True
                    break
        
        pruned_rules.append(current_rule)

    meta_extra = {"timed_out": timed_out} if timed_out else {}
    result = _create_pruned_ruleset(ruleset, pruned_rules, "conservative_prune")
    if timed_out:
        result.metadata = {**result.metadata, **meta_extra}
    return result


def exstracs_prune_aggressive(
    ruleset: ScoredRuleSet,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_f1_loss: float = 0.01,
    *,
    max_seconds: float = 120.0,
) -> ScoredRuleSet:
    """
    Aggressive atom pruning for ExSTraCS using validation data.
    
    Removes atoms iteratively, while allowing up to max_f1_loss F1 drop
    on validation data.
    
    Process:
    1. Baseline F1 on validation data
    2. Iteratively remove atoms
    3. Check F1 after each removal
    4. Accept if F1 does not drop more than max_f1_loss

    Parameters
    ----------
    max_seconds : float
        Maximum runtime for the pruning loop (seconds).
    """
    # Import runtime functions
    from ..runtime import predict_proba as predict_proba_from_ruleset
    
    # Baseline F1
    y_pred_baseline = np.argmax(predict_proba_from_ruleset(ruleset, X_val), axis=1)
    f1_baseline = f1_score(y_val, y_pred_baseline, average='macro', zero_division=0)
    
    pruned_rules = []
    rules_removed = 0

    t_start = time.monotonic()
    timed_out = False
    
    for rule_idx, rule in enumerate(ruleset.rules):
        # Timeout check
        if time.monotonic() - t_start > max_seconds:
            timed_out = True
            pruned_rules.extend(ruleset.rules[rule_idx:])
            break

        if not rule.atoms:  # Skip default rule
            pruned_rules.append(rule)
            continue
        
        current_rule = rule
        changed = True
        
        while changed:
            changed = False

            if time.monotonic() - t_start > max_seconds:
                timed_out = True
                break
            
            for atom_idx in range(len(current_rule.atoms) - 1, -1, -1):
                candidate_atoms = current_rule.atoms[:atom_idx] + current_rule.atoms[atom_idx + 1:]
                
                # Check basic safety
                # IMPORTANT: non-default rules must never become atoms=[].
                if not (len(candidate_atoms) < len(current_rule.atoms) and
                        any(s > 0 for s in current_rule.scores) and
                        len(candidate_atoms) > 0):
                    continue
                
                # Build candidate ruleset
                candidate_rules = pruned_rules + [
                    Rule(
                        atoms=candidate_atoms,
                        scores=current_rule.scores,
                        rule_id=current_rule.rule_id,
                        metadata=current_rule.metadata,
                    )
                ] + ruleset.rules[rule_idx + 1:]
                
                candidate_ruleset = _create_pruned_ruleset(ruleset, candidate_rules, "aggressive_prune")
                
                # Check F1 on validation data
                y_pred_candidate = np.argmax(predict_proba_from_ruleset(candidate_ruleset, X_val), axis=1)
                f1_candidate = f1_score(y_val, y_pred_candidate, average='macro', zero_division=0)
                
                # Accept if F1 loss is acceptable
                if f1_baseline - f1_candidate <= max_f1_loss:
                    current_rule = Rule(
                        atoms=candidate_atoms,
                        scores=current_rule.scores,
                        rule_id=current_rule.rule_id,
                        metadata=current_rule.metadata,
                    )
                    rules_removed += 1
                    changed = True
                    break
        
        pruned_rules.append(current_rule)

    result = _create_pruned_ruleset(ruleset, pruned_rules, "aggressive_prune")
    if timed_out:
        result.metadata = {**result.metadata, "timed_out": True}
    return result


def exstracs_filter_weak_rules(
    ruleset: ScoredRuleSet,
    min_fitness_percentile: float = 0.2,
) -> ScoredRuleSet:
    """
    Filter weak rules based on fitness value.
    
    Keep only rules with score >= min_fitness_percentile of the score distribution.
    
    min_fitness_percentile=0.2 -> keep top 80% of rules
    """
    # Extract max score per rule (as fitness proxy)
    scores = []
    for rule in ruleset.rules:
        if rule.atoms:  # Skip default rule
            max_score = max(rule.scores) if rule.scores else 0.0
            scores.append(max_score)
    
    if not scores:
        return ruleset
    
    # Compute threshold
    scores_array = np.array(scores)
    threshold = np.percentile(scores_array, min_fitness_percentile * 100)
    
    # Filter rules
    filtered_rules = []
    for rule in ruleset.rules:
        if not rule.atoms:  # Keep default rule
            filtered_rules.append(rule)
        else:
            max_score = max(rule.scores) if rule.scores else 0.0
            if max_score >= threshold:
                filtered_rules.append(rule)
    
    return _create_pruned_ruleset(ruleset, filtered_rules, "filter_weak_rules")


def exstracs_consolidate_similar_rules(
    ruleset: ScoredRuleSet,
    similarity_threshold: float = 0.8,
) -> ScoredRuleSet:
    """
    Merge similar rules by averaging their scores.
    
    Similarity = number of shared atoms / max(atom count)
    
    If similarity >= threshold: merge by score averaging
    """
    consolidated_rules = []
    merged = set()
    
    for i, rule_i in enumerate(ruleset.rules):
        if i in merged or not rule_i.atoms:
            consolidated_rules.append(rule_i)
            continue
        
        # Find similar rules
        similar_rules = [rule_i]
        
        for j, rule_j in enumerate(ruleset.rules[i + 1:], start=i + 1):
            if j in merged or not rule_j.atoms:
                continue
            
            # Compute similarity
            similarity = _calculate_rule_similarity(rule_i, rule_j)
            
            if similarity >= similarity_threshold:
                similar_rules.append(rule_j)
                merged.add(j)
        
        # Merge similar rules
        if len(similar_rules) > 1:
            merged_rule = _merge_rules(similar_rules, rule_i.rule_id)
            consolidated_rules.append(merged_rule)
        else:
            consolidated_rules.append(rule_i)
    
    return _create_pruned_ruleset(ruleset, consolidated_rules, "consolidate_similar")


def exstracs_apply_all_shrinking(
    ruleset: ScoredRuleSet,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: ExSTraCSPruningParams | None = None,
) -> ScoredRuleSet:
    """
    Apply multiple shrinking strategies sequentially.
    
    Order:
    0. Auto pre-filter on large populations (> pre_filter_threshold)
    1. Filter weak rules (if explicitly configured)
    1b. Interval merge (if configured) before pruning, because it can
        strongly reduce rule count and speed up subsequent pruning steps.
    2. Conservative Pruning
    3. Aggressive Pruning (if X_val is available)
    4. Consolidation
    """
    if params is None:
        params = ExSTraCSPruningParams()
    
    current_ruleset = ruleset

    # 0. Auto pre-filter for large populations
    n_non_default = sum(1 for r in current_ruleset.rules if r.atoms)
    if n_non_default > params.pre_filter_threshold:
        current_ruleset = exstracs_filter_weak_rules(
            current_ruleset,
            min_fitness_percentile=params.min_fitness_percentile,
        )
    
    # 1. Filter weak rules (explicit)
    if params.filter_weak_rules:
        current_ruleset = exstracs_filter_weak_rules(
            current_ruleset,
            min_fitness_percentile=params.min_fitness_percentile,
        )

    # 1b. Interval merge
    if params.interval_merge:
        current_ruleset = exstracs_merge_intervals(
            current_ruleset,
            iou_threshold=params.interval_merge_iou_threshold,
        )
    
    # 2. Conservative Pruning
    if params.conservative_prune:
        current_ruleset = exstracs_prune_conservative(
            current_ruleset,
            X_ref=X_val,
            y_ref=y_val,
            max_seconds=params.max_pruning_seconds,
        )
    
    # 3. Aggressive Pruning
    if params.aggressive_prune and X_val is not None and y_val is not None:
        current_ruleset = exstracs_prune_aggressive(
            current_ruleset,
            X_val,
            y_val,
            max_f1_loss=params.max_f1_loss,
            max_seconds=params.max_pruning_seconds,
        )
    
    # 4. Consolidation
    if params.consolidate_similar:
        current_ruleset = exstracs_consolidate_similar_rules(
            current_ruleset,
            similarity_threshold=params.similarity_threshold,
        )
    
    return current_ruleset


# ---------------------------------------------------------------------------
# Lossy Rule Compaction (LRC): non-equivalent compaction by
# merging rules with overlapping feature intervals
# ---------------------------------------------------------------------------


def exstracs_merge_intervals(
    ruleset: ScoredRuleSet,
    iou_threshold: float = 0.3,
) -> ScoredRuleSet:
    """Lossy Rule Compaction (LRC): compact ExSTraCS rule sets by merging interval-similar rules.

    Core idea:
    1. Group rules by *class* (argmax of scores) and *feature schema*
       (set of specified features).
    2. Within each group, greedily merge rules with strong interval overlap
       (IoU >= ``iou_threshold``).
    3. During merge, scores are **summed** (preserving total weight),
       and intervals are expanded to their **union** (min/max bounds).

    This is a *non-equivalent* transformation: predictions may differ slightly,
    but it often produces far fewer rules with typically low F1 loss.

    Parameters
    ----------
    ruleset : ScoredRuleSet
        Input rule set (e.g. from ``exstracs_to_scored_ruleset``).
    iou_threshold : float
        Minimum average feature IoU required to merge two rules
        (0 = merge all, 1 = merge only identical intervals).

    Returns
    -------
    ScoredRuleSet
        Compacted rule set.
    """
    default_rules = [r for r in ruleset.rules if not r.atoms]
    non_default_rules = [r for r in ruleset.rules if r.atoms]

    if len(non_default_rules) <= 1:
        return ruleset

    # 1. Group by (predicted class, feature schema)
    groups: dict[tuple[int, frozenset[str]], list[Rule]] = {}
    for rule in non_default_rules:
        class_idx = int(np.argmax(rule.scores))
        schema = _rule_feature_schema(rule)
        key = (class_idx, schema)
        groups.setdefault(key, []).append(rule)

    # 2. Greedy interval merge within each group
    merged_rules: list[Rule] = []
    merge_counter = 0
    total_merged_from = 0

    for (class_idx, schema), group_rules in groups.items():
        cluster_result = _greedy_interval_cluster(group_rules, iou_threshold)
        for cluster in cluster_result:
            merged = _merge_interval_cluster(cluster, merge_counter)
            merged_rules.append(merged)
            total_merged_from += len(cluster)
            merge_counter += 1

    all_rules = default_rules + merged_rules
    result = _create_pruned_ruleset(ruleset, all_rules, "interval_merge")
    result.metadata = {
        **result.metadata,
        "interval_merge_iou_threshold": iou_threshold,
        "rules_before_merge": len(non_default_rules),
        "rules_after_merge": len(merged_rules),
        "total_clusters": merge_counter,
    }
    return result


def _rule_feature_schema(rule: Rule) -> frozenset[str]:
    """Extract the set of specified feature names from a rule."""
    return frozenset(str(a.feature) for a in rule.atoms)


def _extract_feature_intervals(rule: Rule) -> dict[str, tuple[float, float]]:
    """Extract effective [lower, upper] interval per feature from atoms.

    - ``>`` / ``>=`` atoms define lower bound.
    - ``<`` / ``<=`` atoms define upper bound.
    - ``==`` atoms define ``(value, value)`` as degenerate interval.
    - If multiple bounds exist per feature, keep the tightest interval.
    """
    lowers: dict[str, float] = {}
    uppers: dict[str, float] = {}

    for atom in rule.atoms:
        fname = str(atom.feature)
        try:
            val = float(atom.value)
        except (TypeError, ValueError):
            continue

        if atom.op in (">", ">="):
            if fname not in lowers or val > lowers[fname]:
                lowers[fname] = val
        elif atom.op in ("<", "<="):
            if fname not in uppers or val < uppers[fname]:
                uppers[fname] = val
        elif atom.op == "==":
            lowers.setdefault(fname, val)
            uppers.setdefault(fname, val)

    features = set(lowers.keys()) | set(uppers.keys())
    intervals: dict[str, tuple[float, float]] = {}
    for f in features:
        lo = lowers.get(f, -np.inf)
        hi = uppers.get(f, np.inf)
        intervals[f] = (lo, hi)
    return intervals


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Compute intersection-over-union for two 1D intervals.

    Unbounded intervals (+/-inf) are clipped to a large finite range for
    denominator stability.
    """
    a_lo, a_hi = a
    b_lo, b_hi = b

    inter_lo = max(a_lo, b_lo)
    inter_hi = min(a_hi, b_hi)
    intersection = max(0.0, inter_hi - inter_lo)

    # Clip infinite bounds for union-length computation
    _CLIP = 1e6
    al = max(a_lo, -_CLIP)
    ah = min(a_hi, _CLIP)
    bl = max(b_lo, -_CLIP)
    bh = min(b_hi, _CLIP)

    len_a = max(0.0, ah - al)
    len_b = max(0.0, bh - bl)
    union = len_a + len_b - intersection

    if union <= 0.0:
        # Both intervals are identical points or empty
        return 1.0 if intersection >= 0.0 and a_lo == b_lo else 0.0
    return intersection / union


def _mean_feature_iou(
    intervals_a: dict[str, tuple[float, float]],
    intervals_b: dict[str, tuple[float, float]],
) -> float:
    """Mean IoU across all shared features of two rules."""
    common_features = set(intervals_a.keys()) & set(intervals_b.keys())
    if not common_features:
        return 0.0
    total = 0.0
    for f in common_features:
        total += _interval_iou(intervals_a[f], intervals_b[f])
    return total / len(common_features)


def _greedy_interval_cluster(
    rules: list[Rule],
    iou_threshold: float,
) -> list[list[Rule]]:
    """Greedy clustering: assign each rule to the first cluster whose
    representative has sufficient IoU overlap. O(n x k), k = number of clusters."""
    if not rules:
        return []

    intervals_cache = [_extract_feature_intervals(r) for r in rules]
    clusters: list[list[int]] = []
    cluster_representatives: list[dict[str, tuple[float, float]]] = []

    for idx, ivs in enumerate(intervals_cache):
        assigned = False
        for ci, rep_ivs in enumerate(cluster_representatives):
            if _mean_feature_iou(ivs, rep_ivs) >= iou_threshold:
                clusters[ci].append(idx)
                # Update representative to interval union
                cluster_representatives[ci] = _union_intervals(rep_ivs, ivs)
                assigned = True
                break
        if not assigned:
            clusters.append([idx])
            cluster_representatives.append(dict(ivs))

    return [[rules[i] for i in c] for c in clusters]


def _union_intervals(
    a: dict[str, tuple[float, float]],
    b: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Union two feature-interval dicts into combined intervals."""
    result: dict[str, tuple[float, float]] = {}
    for f in set(a.keys()) | set(b.keys()):
        a_iv = a.get(f, (np.inf, -np.inf))
        b_iv = b.get(f, (np.inf, -np.inf))
        lo = min(a_iv[0], b_iv[0])
        hi = max(a_iv[1], b_iv[1])
        result[f] = (lo, hi)
    return result


def _merge_interval_cluster(
    cluster: list[Rule],
    cluster_id: int,
) -> Rule:
    """Merge one cluster of rules into a single rule.

    - Scores are **summed** (preserving total weight).
    - Intervals are expanded to their **union** (min lower, max upper).
    """
    if len(cluster) == 1:
        return cluster[0]

    # Sum scores
    n_scores = len(cluster[0].scores)
    summed_scores = [0.0] * n_scores
    for r in cluster:
        for i, s in enumerate(r.scores):
            summed_scores[i] += s

    # Compute union intervals
    union_ivs: dict[str, tuple[float, float]] = {}
    for r in cluster:
        r_ivs = _extract_feature_intervals(r)
        union_ivs = _union_intervals(union_ivs, r_ivs) if union_ivs else dict(r_ivs)

    # Build atoms from union intervals
    from ..schema import Atom
    atoms: list[Atom] = []
    for fname in sorted(union_ivs.keys()):
        lo, hi = union_ivs[fname]
        if lo == hi:
            # Degenerate interval (==)
            atoms.append(Atom(feature=fname, op="==", value=lo))
        else:
            if np.isfinite(lo):
                atoms.append(Atom(feature=fname, op=">", value=lo))
            if np.isfinite(hi):
                atoms.append(Atom(feature=fname, op="<", value=hi))

    # Collect original metadata
    orig_ids = [r.rule_id or "?" for r in cluster]
    total_numerosity = sum(
        float(r.metadata.get("numerosity", 1)) for r in cluster
    )

    return Rule(
        atoms=atoms,
        scores=summed_scores,
        rule_id=f"imerge_{cluster_id}",
        metadata={
            "source": "interval_merge",
            "merged_count": len(cluster),
            "merged_ids": orig_ids[:10],  # Limit size
            "total_numerosity": total_numerosity,
        },
    )


def _calculate_rule_similarity(rule1: Rule, rule2: Rule) -> float:
    """
    Compute similarity between two rules.
    
    Similarity = |shared atoms| / max(|atoms1|, |atoms2|)
    """
    if not rule1.atoms or not rule2.atoms:
        return 0.0
    
    # Convert atoms to strings for comparison
    atoms1 = set((a.feature, a.op, str(a.value)) for a in rule1.atoms)
    atoms2 = set((a.feature, a.op, str(a.value)) for a in rule2.atoms)
    
    # Compute Jaccard similarity
    intersection = len(atoms1 & atoms2)
    union = len(atoms1 | atoms2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def _merge_rules(rules: list[Rule], rule_id: str) -> Rule:
    """
    Merge multiple similar rules by averaging scores.
    """
    # Keep first rule's atoms (they are similar)
    merged_atoms = rules[0].atoms
    
    # Average scores
    merged_scores = np.mean([np.array(r.scores) for r in rules], axis=0).tolist()
    
    # Combine metadata
    merged_metadata = {"source": "consolidated", "merged_count": len(rules)}
    
    return Rule(
        atoms=merged_atoms,
        scores=merged_scores,
        rule_id=rule_id,
        metadata=merged_metadata,
    )


def _create_pruned_ruleset(
    original: ScoredRuleSet,
    pruned_rules: list[Rule],
    method: str,
) -> ScoredRuleSet:
    """
    Create a new ruleset with pruned rules.
    """
    # Ensure at most one default rule exists.
    # If multiple empty rules exist, sum their scores
    # and use a single fallback rule.
    default_rules = [r for r in pruned_rules if not r.atoms]
    non_default_rules = [r for r in pruned_rules if r.atoms]
    if len(default_rules) > 1:
        summed_scores = np.sum([np.asarray(r.scores, dtype=float) for r in default_rules], axis=0)
        merged_default = Rule(
            atoms=[],
            scores=summed_scores.tolist(),
            rule_id="default",
            metadata={"source": "merged_default", "merged_count": len(default_rules)},
        )
        pruned_rules = non_default_rules + [merged_default]

    return ScoredRuleSet(
        class_labels=original.class_labels,
        feature_names=original.feature_names,
        rules=pruned_rules,
        aggregation=original.aggregation,
        metadata={
            **original.metadata,
            "pruning_method": method,
            "original_rules": len(original.rules),
            "pruned_rules": len(pruned_rules),
        },
    )

