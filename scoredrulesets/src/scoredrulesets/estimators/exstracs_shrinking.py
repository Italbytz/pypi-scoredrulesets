"""
ExSTraCS Rule-Shrinking und Filterung

Dieser Modul bietet Algorithmen zur Reduktion von ExSTraCS Rule-Populationen:

1. Conservative Rule Pruning: Nur sichere Atom-Entfernungen
2. Aggressive Rule Pruning: Mit Validierungs-Daten, akzeptiert leichte F1-Verluste
3. Rule Filtering: Entfernt schwache Regeln basierend auf Fitness/Score
4. Rule Consolidation: Mergt ähnliche Regeln
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score

from ..schema import Rule, ScoredRuleSet


@dataclass
class ExSTraCSPruningParams:
    """Parameter für ExSTraCS Rule-Shrinking"""
    
    # Conservative Pruning (garantiert keine Verschlechterung)
    conservative_prune: bool = False
    
    # Aggressive Pruning (mit Validierungs-Daten)
    aggressive_prune: bool = False
    max_f1_loss: float = 0.01  # Akzeptiere bis zu 1% F1-Verlust
    
    # Rule Filtering
    filter_weak_rules: bool = False
    min_fitness_percentile: float = 0.2  # Behalte nur top 80%
    
    # Rule Consolidation
    consolidate_similar: bool = False
    similarity_threshold: float = 0.8  # Merge wenn > 80% ähnlich

    # Interval-Merge: Nicht-aequivalente Kompaktierung durch Zusammenfassung
    # von Regeln mit ueberlappenden Feature-Intervallen und gleicher Klasse.
    # Scores werden aufsummiert (fitness × numerosity), Intervalle vereinigt.
    interval_merge: bool = False
    interval_merge_iou_threshold: float = 0.3  # Minimale IoU fuer Merge

    # Sicherheitslimits fuer grosse Populationen
    max_pruning_seconds: float = 120.0  # Timeout fuer Pruning-Schleifen (Sekunden)
    pre_filter_threshold: int = 200  # Auto-Filter wenn Regelzahl diesen Wert uebersteigt


def exstracs_prune_conservative(
    ruleset: ScoredRuleSet,
    X_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    *,
    max_seconds: float = 120.0,
) -> ScoredRuleSet:
    """
    Conservative Atom-Pruning für ExSTraCS Regeln.
    
    Entfernt nur Atome, die garantiert keine Vorhersage-Änderung bringen
    (ähnlich wie der Tree-basierte Algorithmus).
    
    Sicherheitskriterien:
    - Weniger Atome
    - Scores bleiben positiv
    - Keine komplette Auflösung

    Parameters
    ----------
    max_seconds : float
        Maximale Laufzeit fuer die Pruning-Schleife (Sekunden).
        Wird das Budget ueberschritten, werden die verbleibenden Regeln
        unveraendert uebernommen.
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
        # Timeout-Pruefung (einmal pro Regel)
        if time.monotonic() - t_start > max_seconds:
            timed_out = True
            # Verbleibende Regeln unveraendert uebernehmen
            pruned_rules.extend(ruleset.rules[rule_idx:])
            break

        if not rule.atoms:  # Überspringe Default-Regel
            pruned_rules.append(rule)
            continue
        
        # Versuche Atome zu entfernen (rückwärts)
        changed = True
        current_rule = rule
        
        while changed:
            changed = False

            # Timeout auch in der inneren Schleife pruefen
            if time.monotonic() - t_start > max_seconds:
                timed_out = True
                break
            
            for atom_idx in range(len(current_rule.atoms) - 1, -1, -1):
                candidate_atoms = current_rule.atoms[:atom_idx] + current_rule.atoms[atom_idx + 1:]
                
                # Überprüfe Sicherheitskriterien
                # WICHTIG: Non-default Regeln dürfen nie zu atoms=[] werden.
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

                # Ohne Referenzdaten: strukturell konservativ, aber ohne F1-Garantie.
                if baseline_f1 is None:
                    current_rule = candidate_rule
                    changed = True
                    break

                # Mit Referenzdaten: nur akzeptieren, wenn F1 nicht schlechter wird.
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
    Aggressive Atom-Pruning für ExSTraCS mit Validierungs-Daten.
    
    Entfernt Atome iterativ, akzeptiert aber bis zu max_f1_loss F1-Verlust
    auf Validierungs-Daten.
    
    Prozess:
    1. Baseline-F1 auf Validierungs-Daten
    2. Iterativ Atome entfernen
    3. Nach jeder Entfernung: F1 überprüfen
    4. Akzeptieren wenn F1 nicht > max_f1_loss sinkt

    Parameters
    ----------
    max_seconds : float
        Maximale Laufzeit fuer die Pruning-Schleife (Sekunden).
    """
    # Importiere Runtime-Funktionen
    from ..runtime import predict_proba as predict_proba_from_ruleset
    
    # Baseline F1
    y_pred_baseline = np.argmax(predict_proba_from_ruleset(ruleset, X_val), axis=1)
    f1_baseline = f1_score(y_val, y_pred_baseline, average='macro', zero_division=0)
    
    pruned_rules = []
    rules_removed = 0

    t_start = time.monotonic()
    timed_out = False
    
    for rule_idx, rule in enumerate(ruleset.rules):
        # Timeout-Pruefung
        if time.monotonic() - t_start > max_seconds:
            timed_out = True
            pruned_rules.extend(ruleset.rules[rule_idx:])
            break

        if not rule.atoms:  # Überspringe Default-Regel
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
                
                # Überprüfe grundsätzliche Sicherheit
                # WICHTIG: Non-default Regeln dürfen nie zu atoms=[] werden.
                if not (len(candidate_atoms) < len(current_rule.atoms) and
                        any(s > 0 for s in current_rule.scores) and
                        len(candidate_atoms) > 0):
                    continue
                
                # Erstelle Kandidaten-Ruleset
                candidate_rules = pruned_rules + [
                    Rule(
                        atoms=candidate_atoms,
                        scores=current_rule.scores,
                        rule_id=current_rule.rule_id,
                        metadata=current_rule.metadata,
                    )
                ] + ruleset.rules[rule_idx + 1:]
                
                candidate_ruleset = _create_pruned_ruleset(ruleset, candidate_rules, "aggressive_prune")
                
                # Überprüfe F1 auf Validierungs-Daten
                y_pred_candidate = np.argmax(predict_proba_from_ruleset(candidate_ruleset, X_val), axis=1)
                f1_candidate = f1_score(y_val, y_pred_candidate, average='macro', zero_division=0)
                
                # Akzeptiere wenn F1-Verlust akzeptabel
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
    Filtere schwache Regeln basierend auf Fitness-Wert.
    
    Behält nur Regeln, deren Score ≥ min_fitness_percentile der Verteilung liegt.
    
    min_fitness_percentile=0.2 → behalte nur top 80% der Regeln
    """
    # Extrahiere max Score pro Regel (als Proxy für Fitness)
    scores = []
    for rule in ruleset.rules:
        if rule.atoms:  # Überspringe Default-Regel
            max_score = max(rule.scores) if rule.scores else 0.0
            scores.append(max_score)
    
    if not scores:
        return ruleset
    
    # Berechne Threshold
    scores_array = np.array(scores)
    threshold = np.percentile(scores_array, min_fitness_percentile * 100)
    
    # Filtere Regeln
    filtered_rules = []
    for rule in ruleset.rules:
        if not rule.atoms:  # Behalte Default-Regel
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
    Merge ähnliche Regeln durch Durchschnitt ihrer Scores.
    
    Ähnlichkeit = Anzahl gemeinsamer Atome / max(Atom-Anzahl)
    
    Wenn ähnlich_keitsgrad ≥ threshold: Merge durch Durchschnitt der Scores
    """
    consolidated_rules = []
    merged = set()
    
    for i, rule_i in enumerate(ruleset.rules):
        if i in merged or not rule_i.atoms:
            consolidated_rules.append(rule_i)
            continue
        
        # Finde ähnliche Regeln
        similar_rules = [rule_i]
        
        for j, rule_j in enumerate(ruleset.rules[i + 1:], start=i + 1):
            if j in merged or not rule_j.atoms:
                continue
            
            # Berechne Ähnlichkeit
            similarity = _calculate_rule_similarity(rule_i, rule_j)
            
            if similarity >= similarity_threshold:
                similar_rules.append(rule_j)
                merged.add(j)
        
        # Merge ähnliche Regeln
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
    Wende mehrere Shrinking-Strategien nacheinander an.
    
    Reihenfolge:
    0. Auto-Vorfilterung bei grossen Populationen (> pre_filter_threshold)
    1. Filter schwache Regeln (falls explizit konfiguriert)
    1b. Interval-Merge (falls konfiguriert) – vor Pruning, da es die Regelzahl
        grob reduziert und die nachfolgenden Pruning-Schritte beschleunigt.
    2. Conservative Pruning
    3. Aggressive Pruning (falls X_val vorhanden)
    4. Consolidation
    """
    if params is None:
        params = ExSTraCSPruningParams()
    
    current_ruleset = ruleset

    # 0. Auto-Vorfilterung bei grossen Populationen
    n_non_default = sum(1 for r in current_ruleset.rules if r.atoms)
    if n_non_default > params.pre_filter_threshold:
        current_ruleset = exstracs_filter_weak_rules(
            current_ruleset,
            min_fitness_percentile=params.min_fitness_percentile,
        )
    
    # 1. Filter schwache Regeln (explizit)
    if params.filter_weak_rules:
        current_ruleset = exstracs_filter_weak_rules(
            current_ruleset,
            min_fitness_percentile=params.min_fitness_percentile,
        )

    # 1b. Interval-Merge
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
# Interval-Merge: Nicht-aequivalente Kompaktierung durch Zusammenfassung
# von Regeln mit ueberlappenden Feature-Intervallen
# ---------------------------------------------------------------------------


def exstracs_merge_intervals(
    ruleset: ScoredRuleSet,
    iou_threshold: float = 0.3,
) -> ScoredRuleSet:
    """Kompaktiere ExSTraCS-Regelmenge durch Zusammenfassung intervall-aehnlicher Regeln.

    Kernidee:
    1. Regeln werden nach *Klasse* (argmax der Scores) und *Feature-Schema*
       (Menge der spezifizierten Features) gruppiert.
    2. Innerhalb jeder Gruppe werden Regeln mit hoher Intervall-Ueberlappung
       (IoU ≥ ``iou_threshold``) gierig zusammengefuehrt.
    3. Beim Merge werden die Scores **aufsummiert** (bewahrt das Gesamtgewicht),
       und die Intervalle zur **Vereinigung** (min/max der Grenzen) erweitert.

    Dies ist eine *nicht-aequivalente* Transformation: Die resultierende Regelmenge
    kann sich in der Vorhersage leicht unterscheiden, erzeugt aber massiv weniger
    Regeln bei typischerweise sehr geringem F1-Verlust.

    Parameters
    ----------
    ruleset : ScoredRuleSet
        Eingabe-Regelmenge (z.B. aus ``exstracs_to_scored_ruleset``).
    iou_threshold : float
        Minimale durchschnittliche Feature-IoU, damit zwei Regeln zusammengefasst
        werden (0 = alles mergen, 1 = nur identische Intervalle).

    Returns
    -------
    ScoredRuleSet
        Kompaktierte Regelmenge.
    """
    default_rules = [r for r in ruleset.rules if not r.atoms]
    non_default_rules = [r for r in ruleset.rules if r.atoms]

    if len(non_default_rules) <= 1:
        return ruleset

    # 1. Gruppiere nach (vorhergesagte Klasse, Feature-Schema)
    groups: dict[tuple[int, frozenset[str]], list[Rule]] = {}
    for rule in non_default_rules:
        class_idx = int(np.argmax(rule.scores))
        schema = _rule_feature_schema(rule)
        key = (class_idx, schema)
        groups.setdefault(key, []).append(rule)

    # 2. Greedy Interval-Merge innerhalb jeder Gruppe
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
    """Extrahiere die Menge der spezifizierten Feature-Namen einer Regel."""
    return frozenset(str(a.feature) for a in rule.atoms)


def _extract_feature_intervals(rule: Rule) -> dict[str, tuple[float, float]]:
    """Extrahiere pro Feature das effektive Intervall [lower, upper] aus den Atomen.

    - ``>`` / ``>=`` Atome liefern die Untergrenze.
    - ``<`` / ``<=`` Atome liefern die Obergrenze.
    - ``==`` Atome liefern ``(value, value)`` als degeneriertes Intervall.
    - Falls mehrere Grenzen pro Feature existieren, wird das engste Intervall genommen.
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
    """Berechne Intersection-over-Union zweier 1D-Intervalle.

    Unbeschraenkte Intervalle (±inf) werden fuer den IoU-Nenner auf einen
    grossen aber endlichen Bereich geklemmt, damit der Wert sinnvoll bleibt.
    """
    a_lo, a_hi = a
    b_lo, b_hi = b

    inter_lo = max(a_lo, b_lo)
    inter_hi = min(a_hi, b_hi)
    intersection = max(0.0, inter_hi - inter_lo)

    # Klemme unendliche Grenzen fuer die Berechnung der Union-Laenge
    _CLIP = 1e6
    al = max(a_lo, -_CLIP)
    ah = min(a_hi, _CLIP)
    bl = max(b_lo, -_CLIP)
    bh = min(b_hi, _CLIP)

    len_a = max(0.0, ah - al)
    len_b = max(0.0, bh - bl)
    union = len_a + len_b - intersection

    if union <= 0.0:
        # Beide Intervalle sind Punkte am gleichen Ort oder leer
        return 1.0 if intersection >= 0.0 and a_lo == b_lo else 0.0
    return intersection / union


def _mean_feature_iou(
    intervals_a: dict[str, tuple[float, float]],
    intervals_b: dict[str, tuple[float, float]],
) -> float:
    """Mittlere IoU ueber alle gemeinsamen Features zweier Regeln."""
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
    """Greedy Clustering: Weise jede Regel dem ersten Cluster zu, dessen
    Repraesentant ausreichend IoU-Ueberlappung hat.  O(n × k) mit k = Anzahl Cluster."""
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
                # Aktualisiere Repraesentant auf Union-Intervall
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
    """Vereinige zwei Feature-Intervall-Dicts zu ihren Union-Intervallen."""
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
    """Fasse einen Cluster von Regeln zu einer einzigen Regel zusammen.

    - Scores werden **aufsummiert** (bewahrt Gesamtgewicht).
    - Intervalle werden zur **Vereinigung** (min lower, max upper) erweitert.
    """
    if len(cluster) == 1:
        return cluster[0]

    # Scores aufsummieren
    n_scores = len(cluster[0].scores)
    summed_scores = [0.0] * n_scores
    for r in cluster:
        for i, s in enumerate(r.scores):
            summed_scores[i] += s

    # Union-Intervalle berechnen
    union_ivs: dict[str, tuple[float, float]] = {}
    for r in cluster:
        r_ivs = _extract_feature_intervals(r)
        union_ivs = _union_intervals(union_ivs, r_ivs) if union_ivs else dict(r_ivs)

    # Atome aus Union-Intervallen aufbauen
    from ..schema import Atom
    atoms: list[Atom] = []
    for fname in sorted(union_ivs.keys()):
        lo, hi = union_ivs[fname]
        if lo == hi:
            # Degeneriertes Intervall (==)
            atoms.append(Atom(feature=fname, op="==", value=lo))
        else:
            if np.isfinite(lo):
                atoms.append(Atom(feature=fname, op=">", value=lo))
            if np.isfinite(hi):
                atoms.append(Atom(feature=fname, op="<", value=hi))

    # Sammle originale Metadaten
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
            "merged_ids": orig_ids[:10],  # Begrenze Groesse
            "total_numerosity": total_numerosity,
        },
    )


def _calculate_rule_similarity(rule1: Rule, rule2: Rule) -> float:
    """
    Berechne Ähnlichkeit zwischen zwei Regeln.
    
    Ähnlichkeit = |gemeinsame Atome| / max(|Atome1|, |Atome2|)
    """
    if not rule1.atoms or not rule2.atoms:
        return 0.0
    
    # Konvertiere Atome zu Strings für Vergleich
    atoms1 = set((a.feature, a.op, str(a.value)) for a in rule1.atoms)
    atoms2 = set((a.feature, a.op, str(a.value)) for a in rule2.atoms)
    
    # Berechne Jaccard-Ähnlichkeit
    intersection = len(atoms1 & atoms2)
    union = len(atoms1 | atoms2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def _merge_rules(rules: list[Rule], rule_id: str) -> Rule:
    """
    Merge mehrere ähnliche Regeln durch Durchschnitt der Scores.
    """
    # Behalte erste Regel's Atome (sie sind ähnlich)
    merged_atoms = rules[0].atoms
    
    # Durchschnittliche Scores
    merged_scores = np.mean([np.array(r.scores) for r in rules], axis=0).tolist()
    
    # Kombiniere Metadaten
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
    Erstelle neues Ruleset mit pruned rules.
    """
    # Stelle sicher, dass maximal eine Default-Regel vorhanden ist.
    # Falls mehrere leere Regeln existieren, werden deren Scores aufsummiert
    # und als eine einzige Fallback-Regel genutzt.
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

