# -*- coding: utf-8 -*-
"""Trace exactly why multi-monomial seeds are eliminated from Pareto front.

Fortfuehrung: Warum werden die class-discriminativen Seeds dominiert?
- Klassenverteilung im car-Datensatz
- Wie aggregiert predict_classes die Gewichte bei Multi-Monom?
- Woher kommt recalls=[0,0,1,0] für die Dominierer?
- Warum können Seeds nicht mithalten?
"""
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from scoredrulesets.benchmarking.datasets import load_dataset_registry
from scoredrulesets.estimators.logicgp import (
    LogicGPClassifier,
    _compute_weights,
    _evaluate_fitness,
    _discretize_features,
    _generate_literals,
    _pareto_front,
    _Polynomial,
    _Monomial,
    _Fitness,
)

reg = load_dataset_registry(include_online_uci=True, include_synthetic=True)
b = reg["uci_car_evaluation"]
X, y = b.X, b.y
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

class_labels = sorted(np.unique(y_tr))
class_to_idx = {label: idx for idx, label in enumerate(class_labels)}
y_idx = np.array([class_to_idx[v] for v in y_tr])
n_classes = len(class_labels)

X_disc, binners, cat_masks = _discretize_features(X_tr, n_bins=5)
all_lits = _generate_literals(X_disc)

clf = LogicGPClassifier(trainer="flcw", n_bins=5, random_state=0)
clf._rng_ = np.random.default_rng(0)
clf.feature_names_in_ = np.array([f"f{i}" for i in range(X_disc.shape[1])])
population = clf._init_population(all_lits, n_classes, X_disc, y_idx)

for poly in population:
    _compute_weights(poly, X_disc, y_idx, n_classes)

single_mono = [(p, _evaluate_fitness(p, X_disc, y_idx, n_classes))
               for p in population if len(p.monomials) == 1]
multi_mono = [(p, _evaluate_fitness(p, X_disc, y_idx, n_classes))
              for p in population if len(p.monomials) > 1]

# =====================================================================
# 0) Grundlegende Statistiken
# =====================================================================
print("=" * 70)
print("0) KLASSENVERTEILUNG & GRUNDLAGEN")
print("=" * 70)
class_counts = np.bincount(y_idx, minlength=n_classes)
print(f"Klassen:        {class_labels}")
print(f"Klassenindizes: {list(range(n_classes))}")
print(f"Klassenverteilung: {class_counts}")
print(f"Klassenanteile:    {np.round(class_counts / len(y_idx), 4)}")
print(f"Majority-Klasse:   Klasse {np.argmax(class_counts)} "
      f"({class_labels[np.argmax(class_counts)]}) mit {class_counts.max()}/{len(y_idx)} "
      f"({class_counts.max()/len(y_idx)*100:.1f}%)")
print(f"\nAnzahl Literale:        {len(all_lits)}")
print(f"Single-monomial Modelle: {len(single_mono)}")
print(f"Multi-monomial Seeds:    {len(multi_mono)}")

# =====================================================================
# 1) Warum predizieren die "Gewinner" nur Klasse 2?
# =====================================================================
print("\n" + "=" * 70)
print("1) WARUM SAGEN DIE PARETO-ÜBERLEBENDEN NUR KLASSE 2 VORHER?")
print("=" * 70)

# Die Grundmechanik: Ein single-monomial Modell hat 1 Monomial mit bestimmten
# Gewichten. Wo es nicht feuert, greifen die default_weights.
# predict_classes = argmax(scores). Wenn nur 1 Monom UND die default_weights
# Klasse 2 bevorzugen, wird *überall* Klasse 2 gesagt.

# Schaue uns an, was bei single-mono Modellen passiert
print("\nBeispiel-Analyse: Welche Vorhersage hat ein typisches single-mono Modell?")
# Finde ein Modell mit recalls=[0,0,1,0]
dominant_pattern = None
for poly, fit in single_mono:
    if np.allclose(fit.objectives, [0, 0, 1, 0], atol=0.01):
        dominant_pattern = (poly, fit)
        break

if dominant_pattern:
    poly, fit = dominant_pattern
    mon = poly.monomials[0]
    mask = mon.fires_mask(X_disc)
    n_fire = int(mask.sum())
    n_nofire = len(y_idx) - n_fire
    print(f"  Monom feuert: {n_fire}/{len(y_idx)}")
    print(f"  Monom-Gewichte (wo feuert):   {np.round(mon.weights, 4)} → predicts class {np.argmax(mon.weights)}")
    print(f"  Default-Gewichte (wo nicht):   {np.round(poly.default_weights, 4)} → predicts class {np.argmax(poly.default_weights)}")
    preds = poly.predict_classes(X_disc)
    pred_dist = np.bincount(preds, minlength=n_classes)
    print(f"  Vorhersage-Verteilung: {pred_dist}")
    print(f"  Recalls: {np.round(fit.objectives, 4)}")
    print(f"  Size: {fit.size}")

    # Detailliert: Was passiert für jede Klasse?
    for c in range(n_classes):
        c_mask = y_idx == c
        n_c = int(c_mask.sum())
        correct = int((preds[c_mask] == c).sum())
        print(f"    Klasse {c} ({class_labels[c]}): {correct}/{n_c} korrekt → recall={correct/max(n_c,1):.4f}")

# =====================================================================
# 2) SCHLÜSSELDIAGNOSE: Wie sieht predict_classes bei Multi-Monom aus?
# =====================================================================
print("\n" + "=" * 70)
print("2) PREDICT_CLASSES BEI MULTI-MONOM SEEDS - DAS AGGREGATIONSPROBLEM")
print("=" * 70)

print("\nDie Vorhersage-Logik bei Multi-Monom-Polynomen:")
print("  scores[i] = SUM(weights) aller Monome die für Sample i feuern")
print("  Falls KEIN Monom feuert: scores[i] = default_weights")
print("  prediction = argmax(scores[i])")
print()

for i, (poly, fit) in enumerate(multi_mono):
    n_mon = len(poly.monomials)
    print(f"--- Multi-Monom Seed {i}: {n_mon} Monome, size={fit.size} ---")

    # Welche Monome feuern für welche Samples?
    fire_masks = []
    for j, mon in enumerate(poly.monomials):
        mask = mon.fires_mask(X_disc)
        fire_masks.append(mask)
        n_fire = int(mask.sum())
        dominant_class = int(np.argmax(mon.weights))
        # Wie viele der feuernden Samples gehören zur dominanten Klasse?
        if n_fire > 0:
            class_dist_in_fire = np.bincount(y_idx[mask], minlength=n_classes)
        else:
            class_dist_in_fire = np.zeros(n_classes, dtype=int)
        print(f"  Monom {j}: fires={n_fire}, weights={np.round(mon.weights, 4)}, "
              f"dominant_class={dominant_class}, class_dist_fired={class_dist_in_fire}")

    # Überlappungsanalyse
    fire_count_per_sample = np.zeros(len(y_idx), dtype=int)
    any_fired = np.zeros(len(y_idx), dtype=bool)
    for mask in fire_masks:
        fire_count_per_sample += mask.astype(int)
        any_fired |= mask

    overlap_counts = Counter(fire_count_per_sample)
    print(f"  Feuert-Verteilung: {dict(sorted(overlap_counts.items()))} "
          f"(0=default, 1=single, 2+=overlap)")
    print(f"  Default (kein Monom): {int((~any_fired).sum())}/{len(y_idx)}")
    print(f"  Default-Gewichte: {np.round(poly.default_weights, 4)} → class {np.argmax(poly.default_weights)}")

    # KERN: Was passiert bei Überlappung?
    # Wenn mehrere Monome gleichzeitig feuern, werden Gewichte ADDIERT
    # → argmax der Summe bestimmt die Klasse!
    preds = poly.predict_classes(X_disc)
    pred_dist = np.bincount(preds, minlength=n_classes)
    print(f"  Vorhersage: {pred_dist}")
    print(f"  Recalls: {np.round(fit.objectives, 4)}, consolidated={fit.consolidated:.4f}")

    # Zeige, was bei Samples passiert die von >1 Monom abgedeckt werden
    overlap_samples = fire_count_per_sample >= 2
    if overlap_samples.any():
        n_overlap = int(overlap_samples.sum())
        overlap_preds = preds[overlap_samples]
        overlap_y = y_idx[overlap_samples]
        overlap_pred_dist = np.bincount(overlap_preds, minlength=n_classes)
        overlap_true_dist = np.bincount(overlap_y, minlength=n_classes)
        print(f"  Überlappende Samples ({n_overlap}): "
              f"vorhergesagt={overlap_pred_dist}, wahr={overlap_true_dist}")

        # Zeige aggregierte Scores für ein paar Overlap-Samples
        overlap_idx = np.where(overlap_samples)[0][:3]
        for si in overlap_idx:
            accum = np.zeros(n_classes, dtype=float)
            for j, mon in enumerate(poly.monomials):
                if fire_masks[j][si]:
                    accum += mon.weights
            print(f"    Sample {si}: aggregiert={np.round(accum, 4)} → "
                  f"pred={np.argmax(accum)}, true={y_idx[si]}")

    print()

# =====================================================================
# 3) DOMINANZ-ANALYSE: Wer dominiert die Seeds und WARUM?
# =====================================================================
print("=" * 70)
print("3) DOMINANZ-ANALYSE: WARUM DIE SEEDS VERLIEREN")
print("=" * 70)

for i, (poly, fit) in enumerate(multi_mono):
    print(f"\n--- Seed {i}: recalls={np.round(fit.objectives, 4)}, size={fit.size} ---")

    # Finde alle Dominierer
    dominators = []
    for j, (other_poly, other_fit) in enumerate(single_mono):
        if other_fit.dominates(fit):
            dominators.append((j, other_poly, other_fit))

    if dominators:
        print(f"  Dominiert von {len(dominators)} single-monomial Modellen:")
        # Zeige die 3 kleinsten und 3 mit bester consolidated
        by_size = sorted(dominators, key=lambda x: x[2].size)
        by_cons = sorted(dominators, key=lambda x: -x[2].consolidated)
        print(f"  Kleinster Dominierer:")
        for j, _, df in by_size[:3]:
            print(f"    Single[{j}]: recalls={np.round(df.objectives, 4)}, size={df.size}")
        print(f"  Bester consolidated Dominierer:")
        for j, _, df in by_cons[:3]:
            print(f"    Single[{j}]: recalls={np.round(df.objectives, 4)}, size={df.size}, "
                  f"cons={df.consolidated:.4f}")

        # Kernfrage: Warum ist der Seed schlechter?
        # Der Seed hat size >= 2 (mehrere Monome), ein Single hat size=1
        # → Wenn der Single in JEDER recall-Komponente >= dem Seed ist,
        #   UND size <= dem Seed, dann dominiert er.
        ex = by_size[0]
        _, _, d_fit = ex
        print(f"\n  Dominanz-Erklärung (Seed vs kleinster Dominierer):")
        print(f"    Seed  : recalls={np.round(fit.objectives, 4)}, size={fit.size}")
        print(f"    Dom.  : recalls={np.round(d_fit.objectives, 4)}, size={d_fit.size}")
        print(f"    size: {d_fit.size} <= {fit.size} ? {d_fit.size <= fit.size}")
        for c in range(n_classes):
            r_seed = fit.objectives[c]
            r_dom = d_fit.objectives[c]
            print(f"    recall[{c}]: dom={r_dom:.4f} >= seed={r_seed:.4f} ? {r_dom >= r_seed - 1e-12} "
                  f"({'GLEICH' if abs(r_dom - r_seed) < 1e-12 else 'DOM BESSER' if r_dom > r_seed else '*** SEED BESSER ***'})")
    else:
        print(f"  NICHT dominiert von single-monomial Modellen!")
        # Prüfe ob von anderem multi-mono dominiert
        for j2, (_, other_fit2) in enumerate(multi_mono):
            if i != j2 and other_fit2.dominates(fit):
                print(f"  → Dominiert von multi-mono Seed {j2}")

# =====================================================================
# 4) DAS KERNPROBLEM: Warum haben Seeds schlechte Recalls?
# =====================================================================
print("\n" + "=" * 70)
print("4) DAS KERNPROBLEM: GEWICHTSAGGREGATION BEI MULTI-MONOM")
print("=" * 70)

print("""
Hypothese: Bei Multi-Monom-Polynomen mit initialen Gewichten (berechnet
durch _compute_weights) können sich die Gewichte destruktiv überlagern.

Wenn Monom A und Monom B beide für Sample x feuern:
  score(x) = weights_A + weights_B
  → argmax(score) ist NICHT notwendig das gleiche wie argmax(weights_A)
  oder argmax(weights_B)

Dies kann dazu führen, dass ein Seed, der pro Klasse ein spezifisches Monom
hat, trotzdem fast überall die Majority-Klasse vorhersagt.
""")

# Detaillierte Analyse eines 4-Monom Seeds
for i, (poly, fit) in enumerate(multi_mono):
    if len(poly.monomials) == n_classes:  # Multi-Monom = ein Monom pro Klasse
        print(f"=== Seed {i}: Ein Monom pro Klasse ===")
        print(f"Recalls: {np.round(fit.objectives, 4)}")

        # Für jeden Sample: Berechne den Score manuell
        n = len(y_idx)
        all_scores = np.zeros((n, n_classes), dtype=float)
        masks = []
        any_fired = np.zeros(n, dtype=bool)

        for j, mon in enumerate(poly.monomials):
            mask = mon.fires_mask(X_disc)
            masks.append(mask)
            any_fired |= mask
            all_scores[mask] += mon.weights

        # Default für non-fired
        all_scores[~any_fired] = poly.default_weights

        preds = np.argmax(all_scores, axis=1)

        # Analysiere pro Klasse warum recall schlecht ist
        for c in range(n_classes):
            c_samples = np.where(y_idx == c)[0]
            n_c = len(c_samples)
            correct = int((preds[c_samples] == c).sum())
            print(f"\n  Klasse {c} ({class_labels[c]}): {correct}/{n_c} korrekt")

            # Warum werden die Samples von Klasse c falsch vorhergesagt?
            wrong = c_samples[preds[c_samples] != c]
            if len(wrong) > 0:
                wrong_preds = preds[wrong]
                wrong_pred_dist = Counter(wrong_preds.tolist())
                print(f"    {len(wrong)} falsch, davon vorhergesagt als: {dict(wrong_pred_dist)}")

                # Zeige was bei ein paar falsch vorhergesagten Samples passiert
                for si in wrong[:3]:
                    n_firing = sum(1 for m in masks if m[si])
                    score = all_scores[si]
                    print(f"    Sample {si}: {n_firing} Monome feuern, "
                          f"score={np.round(score, 4)}, pred={preds[si]}, true={c}")
                    for j, mon in enumerate(poly.monomials):
                        if masks[j][si]:
                            print(f"      Monom {j} feuert: weights={np.round(mon.weights, 4)}")
        break  # Nur das erste Multi-Klassen-Seed detailliert

# =====================================================================
# 5) VERGLEICH: Bester Recall der Single-Mono Modelle PRO KLASSE
# =====================================================================
print("\n" + "=" * 70)
print("5) BESTER RECALL PRO KLASSE: SINGLE-MONO VS SEEDS")
print("=" * 70)

for c in range(n_classes):
    best_single = max(single_mono, key=lambda x: x[1].objectives[c])
    _, bf = best_single
    print(f"\nKlasse {c} ({class_labels[c]}): {class_counts[c]} Samples")
    print(f"  Bester single-mono recall: {bf.objectives[c]:.4f} (size={bf.size})")
    print(f"  Volle recalls: {np.round(bf.objectives, 4)}")

    # Bester Seed für diese Klasse
    if multi_mono:
        best_seed = max(multi_mono, key=lambda x: x[1].objectives[c])
        _, sf = best_seed
        print(f"  Bester Seed recall:        {sf.objectives[c]:.4f} (size={sf.size})")
        print(f"  Volle recalls: {np.round(sf.objectives, 4)}")

# =====================================================================
# 6) SZENARIEN: Was müsste ein Seed leisten um zu überleben?
# =====================================================================
print("\n" + "=" * 70)
print("6) PARETO-FRONT ANALYSE: WAS ÜBERLEBT UND WARUM?")
print("=" * 70)

all_evaluated = single_mono + multi_mono
front = _pareto_front(all_evaluated)
front_singles = [(p, f) for p, f in front if len(p.monomials) == 1]
front_multis = [(p, f) for p, f in front if len(p.monomials) > 1]

print(f"Pareto-Front: {len(front)} (singles={len(front_singles)}, multis={len(front_multis)})")

# Einzigartige Recall-Muster in der Front
front_patterns = Counter()
for _, f in front:
    key = tuple(np.round(f.objectives, 3))
    front_patterns[key] += 1

print(f"\nEinzigartige Recall-Muster in der Front:")
for pattern, count in front_patterns.most_common(15):
    sizes = [f.size for _, f in front if tuple(np.round(f.objectives, 3)) == pattern]
    print(f"  recalls={pattern} : {count} Modelle, sizes={sorted(set(sizes))}")

# Klasse 2 Recall = 1.0 Modelle genauer ansehen
class2_perfect = [(p, f) for p, f in front if f.objectives[2] > 0.99]
if class2_perfect:
    print(f"\nModelle mit recall[2] ≈ 1.0 in der Front: {len(class2_perfect)}")
    # Warum dominieren die? Weil Klasse 2 die Majority ist!
    # Ein Modell das ALLES als Klasse 2 vorhersagt hat recall[2]=1.0
    # und recall[0,1,3]=0.0 aber size kann =1 sein.
    # Ein Seed mit z.B. recalls=[0.1, 0.1, 0.5, 0.1] und size=4
    # wird dominiert weil: size 1 <= 4, und [0,0,1,0] >= [0.1,0.1,0.5,0.1]
    # in den Komponenten 0,1,3 (die sind 0 bei beiden oder 0>=0.1 - NEIN!)
    # Warte... 0 >= 0.1 ist FALSCH. Dann kann [0,0,1,0] nicht [0.1,0.1,0.5,0.1] dominieren!
    #
    # → Prüfe das genau!
    print("\n  Prüfe ob recalls=[0,0,1,0] wirklich Seeds dominieren kann...")
    # Die Seeds müssten recalls haben wo Component 0,1,3 alle =0 sind
    # UND recall[2] <= 1 UND size > dominator.size
    for i, (poly, fit) in enumerate(multi_mono):
        print(f"\n  Seed {i}: recalls={np.round(fit.objectives, 4)}, size={fit.size}")
        # Checke jedes [0,0,1,0] Modell
        n_dom = 0
        for _, sf in single_mono:
            if sf.dominates(fit):
                n_dom += 1
        print(f"    Dominiert von {n_dom} singles")
        if n_dom > 0:
            # Zeige den ersten Dominator
            for _, sf in single_mono:
                if sf.dominates(fit):
                    print(f"    Dominator: recalls={np.round(sf.objectives, 4)}, size={sf.size}")
                    print(f"    Vergleich per Komponente:")
                    for c in range(n_classes):
                        print(f"      recall[{c}]: dom={sf.objectives[c]:.4f} >= seed={fit.objectives[c]:.4f}? "
                              f"{sf.objectives[c] >= fit.objectives[c] - 1e-12}")
                    break

# =====================================================================
# 7) FAZIT: Ursache & mögliche Fixes
# =====================================================================
print("\n" + "=" * 70)
print("7) ZUSAMMENFASSUNG")
print("=" * 70)
print("""
Frage: Warum werden multi-monomial Seeds im FLCW-Pareto eliminiert?

Zu prüfende Hypothesen:
A) Seeds haben recall[0,1,3]=0 weil die Gewichtsaggregation die Klasse 2
   bevorzugt → dann dominiert jedes size=1 Modell mit recall[2]=1.0
B) Seeds haben positive recalls für minority-Klassen aber Gewichts-
   überlappung reduziert recall[2] unter 1.0 → sie sind "strikt schlechter"
   in recall[2] UND "nicht besser genug" in recall[0,1,3]
C) Die Seeds haben tatsächlich bessere minority-recalls, aber die
   Dominanz-Relation sieht nur recall >= in ALLEN Komponenten
""")

