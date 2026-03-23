"""Diagnose 2: Vergleich consolidated (mean recall) vs actual macro-F1 bei Modellauswahl."""
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scoredrulesets.estimators.logicgp import (
    LogicGPClassifier,
    _final_model_selection,
    _compute_weights,
    _evaluate_fitness,
    _discretize_features,
    _Polynomial,
    _Fitness,
)

import warnings
warnings.filterwarnings("ignore")


def _get_all_candidates_and_best(clf, X_train, y_train, X_val, y_val):
    """Monkey-patch _run_gp to also collect all candidates, then evaluate them all."""
    from sklearn.utils.validation import check_X_y
    from sklearn.utils.multiclass import unique_labels

    # Use internal _run_gp and intercept all_candidates
    clf.fit(X_train, y_train)

    # Re-run to collect all candidates by re-patching
    # Stattdessen: wir bewerten alle Individuen der FINALEN Population.
    # Dafuer muessen wir den gesamten GP-Lauf mit gesammelten Kandidaten wiederholen.
    # Einfacher: wir testen die Top-Kandidaten aus _final_model_selection
    # Alternative: Wir evaluieren das ausgewaehlte Modell + zusaetzlich alle Polynome
    # die in der letzten Generation ueberlebt haben.

    # Rufe predict direkt auf dem Polynom auf (Training und Validation)
    pass


def diagnose_selection(name, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # Um an die alle Kandidaten zu kommen, patchen wir _run_gp temporaer
    original_run_gp = LogicGPClassifier._run_gp
    all_collected = []

    def patched_run_gp(self, population, all_literals, X_disc, y_idx, n_classes):
        # Kopie des Original-Codes, aber wir speichern all_candidates
        use_rlcw = self.trainer.lower().startswith("rlcw")
        from scoredrulesets.estimators.logicgp import (
            _evaluate_fitness, _evaluate_fitness_rlcw,
            _pareto_front, _pareto_front_rlcw,
            _tournament_trim, _crossover,
            _mut_insert_literal, _mut_delete_literal,
            _mut_replace_literal, _mut_insert_monomial, _mut_delete_monomial,
        )
        for poly in population:
            _compute_weights(poly, X_disc, y_idx, n_classes)
        if use_rlcw:
            evaluated = [(poly, _evaluate_fitness_rlcw(poly, X_disc, y_idx, n_classes)) for poly in population]
            evaluated = _pareto_front_rlcw(evaluated)
        else:
            evaluated = [(poly, _evaluate_fitness(poly, X_disc, y_idx, n_classes)) for poly in population]
            evaluated = _pareto_front(evaluated)
        if self.population_size is not None and len(evaluated) > self.population_size:
            evaluated = _tournament_trim(evaluated, self.population_size, self.tournament_size, self._rng_)
        best_consolidated = max(f.consolidated for _, f in evaluated)
        stagnation_count = 0
        all_candidates = [(poly, fit, fit.consolidated) for poly, fit in evaluated]
        _MUT_OPS = [
            lambda p: _mut_insert_literal(p, all_literals, self._rng_),
            lambda p: _mut_delete_literal(p, self._rng_),
            lambda p: _mut_replace_literal(p, all_literals, self._rng_),
            lambda p: _mut_insert_monomial(p, all_literals, n_classes, self._rng_),
            lambda p: _mut_delete_monomial(p, self._rng_),
        ]
        n_adapt = max(1, self.n_adaptations_per_gen)
        for _gen in range(self.max_generations):
            new_polys = []
            for i in range(n_adapt):
                if i == 0 and len(evaluated) >= 2:
                    p_a, p_b = self._select_two_parents(evaluated)
                    new_polys.append(_crossover(p_a, p_b, self._rng_))
                else:
                    parent = self._select_parent(evaluated)
                    new_polys.append(_MUT_OPS[(i - 1) % 5](parent))
            new_evaluated = []
            for poly in new_polys:
                if not poly.monomials:
                    continue
                if self.max_model_size is not None and poly.size > self.max_model_size:
                    continue
                _compute_weights(poly, X_disc, y_idx, n_classes)
                if use_rlcw:
                    fit = _evaluate_fitness_rlcw(poly, X_disc, y_idx, n_classes)
                else:
                    fit = _evaluate_fitness(poly, X_disc, y_idx, n_classes)
                new_evaluated.append((poly, fit))
            combined = evaluated + new_evaluated
            if use_rlcw:
                evaluated = _pareto_front_rlcw(combined)
            else:
                evaluated = _pareto_front(combined)
            if self.population_size is not None and len(evaluated) > self.population_size:
                evaluated = _tournament_trim(evaluated, self.population_size, self.tournament_size, self._rng_)
            current_best = max(f.consolidated for _, f in evaluated)
            if current_best > best_consolidated + 1e-10:
                best_consolidated = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            all_candidates.extend((poly, fit, fit.consolidated) for poly, fit in evaluated)
            if stagnation_count >= self.stagnation_generations:
                break

        # Speichere ALLE Kandidaten fuer spaetere Analyse
        all_collected.clear()
        all_collected.extend(all_candidates)
        # Auch die finale Population separat
        all_collected.append(("__final_pop__", evaluated))

        return _final_model_selection(all_candidates, min_improvement=self.min_improvement_pct)

    LogicGPClassifier._run_gp = patched_run_gp

    try:
        clf = LogicGPClassifier(
            trainer="flcw_macro",
            max_generations=200,
            stagnation_generations=80,
            population_size=30,
            n_adaptations_per_gen=6,
            n_bins=5,
            random_state=0,
        )
        clf.fit(X_tr, y_tr)

        # Ausgewaehltes Modell: F1
        y_pred_sel = clf.predict(X_te)
        f1_selected = f1_score(y_te, y_pred_sel, average="macro")

        # Discretize test data for direct polynomial evaluation
        X_te_disc, _, _ = _discretize_features(
            np.asarray(X_te), fitted_binners=clf._binners_, cat_masks=clf._cat_masks_
        )
        X_tr_disc, _, _ = _discretize_features(
            np.asarray(X_tr), fitted_binners=clf._binners_, cat_masks=clf._cat_masks_
        )

        # Finale Population extrahieren
        final_pop_entry = all_collected[-1]
        assert final_pop_entry[0] == "__final_pop__"
        final_pop = final_pop_entry[1]

        # Dedupliziere Kandidaten nach Modellgroesse und Consolidated
        seen = set()
        unique_candidates = []
        for item in all_collected[:-1]:
            poly, fit, consolidated = item
            key = (fit.size, round(consolidated, 6))
            if key not in seen:
                seen.add(key)
                unique_candidates.append((poly, fit, consolidated))

        print(f"\n=== {name} ===")
        print(f"  Insgesamt {len(unique_candidates)} unique Kandidaten (von {len(all_collected)-1})")
        print(f"  Finale Population: {len(final_pop)} Individuen")
        print(f"  Ausgewaehltes Modell: F1(test)={f1_selected:.4f}")

        # Bewerte Top-Kandidaten nach tatsaechlicher F1
        results = []
        for poly, fit, consolidated in unique_candidates:
            # Training F1
            preds_tr = poly.predict_classes(X_tr_disc)
            f1_tr = f1_score(
                np.asarray(y_tr, dtype=int) if not np.issubdtype(np.asarray(y_tr).dtype, np.integer) else y_tr,
                preds_tr, average="macro", labels=list(range(len(clf.classes_)))
            )
            # Test F1
            preds_te = poly.predict_classes(X_te_disc)
            y_te_idx = np.array([list(clf.classes_).index(c) for c in y_te])
            f1_te = f1_score(y_te_idx, preds_te, average="macro", labels=list(range(len(clf.classes_))))
            results.append((poly, fit, consolidated, f1_tr, f1_te))

        # Sortiere nach Test-F1
        results.sort(key=lambda x: x[4], reverse=True)

        print(f"\n  Top-5 nach tatsaechlicher Test-F1:")
        for i, (poly, fit, cons, f1_tr, f1_te) in enumerate(results[:5]):
            print(f"    #{i+1}: F1(test)={f1_te:.4f}  F1(train)={f1_tr:.4f}  consolidated={cons:.4f}  size={fit.size}")

        # Was waehlt _final_model_selection?
        selected_cons = max(results, key=lambda x: x[2])
        print(f"\n  consolidated-Auswahl: F1(test)={selected_cons[4]:.4f}  consolidated={selected_cons[2]:.4f}  size={selected_cons[1].size}")

        best_f1 = results[0]
        print(f"  Beste Test-F1:        F1(test)={best_f1[4]:.4f}  consolidated={best_f1[2]:.4f}  size={best_f1[1].size}")

        # Bewerte nur finale Population
        final_results = []
        for poly, fit in final_pop:
            preds_te = poly.predict_classes(X_te_disc)
            y_te_idx = np.array([list(clf.classes_).index(c) for c in y_te])
            f1_te = f1_score(y_te_idx, preds_te, average="macro", labels=list(range(len(clf.classes_))))
            final_results.append((poly, fit, f1_te))
        final_results.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  Beste F1 in finaler Population: {final_results[0][2]:.4f}  (size={final_results[0][1].size})")
        print(f"  Schlechteste F1 in finaler Pop: {final_results[-1][2]:.4f}  (size={final_results[-1][1].size})")

        gap = best_f1[4] - f1_selected
        print(f"\n  >>> GAP (best possible - selected): {gap:+.4f}")

    finally:
        LogicGPClassifier._run_gp = original_run_gp


for name, loader in [("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer)]:
    X, y = loader(return_X_y=True)
    diagnose_selection(name, X, y)

