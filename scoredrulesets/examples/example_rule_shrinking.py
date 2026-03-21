#!/usr/bin/env python3
"""
Beispiel: Rule-Shrinking-Algorithmus für CART und HS Modelle

Dieses Skript demonstriert, wie der Atom-Pruning-Algorithmus verwendet wird,
um die Modellgröße zu reduzieren bei gleichzeitiger Erhaltung der Vorhersage-Qualität.
"""

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


def benchmark_pruning_on_dataset(name, X, y):
    """Benchmark Atom-Pruning auf einem Datensatz"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    n_samples, n_features = X_train.shape
    n_classes = len(set(y_train))
    
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"  n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}")
    print(f"{'='*70}")
    
    # Baseline: Ohne Pruning
    print("\n1. BASELINE: Ohne Atom-Pruning (λ=None)")
    print("-" * 70)
    clf_baseline = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 4},
        random_state=42,
    )
    clf_baseline.fit(X_train, y_train)
    ruleset_baseline = clf_baseline.to_ruleset()
    
    n_rules_baseline = len(ruleset_baseline.rules)
    n_atoms_baseline = sum(len(r.atoms) for r in ruleset_baseline.rules)
    f1_baseline = f1_score(y_test, clf_baseline.predict(X_test), average='macro')
    
    print(f"  Rules:        {n_rules_baseline}")
    print(f"  Total Atoms:  {n_atoms_baseline}")
    print(f"  F1 (macro):   {f1_baseline:.4f}")
    
    # Pruning mit verschiedenen Lambda-Werten
    results = []
    for prune_lambda in [1.5, 2.0, 3.0]:
        print(f"\n2. Mit Atom-Pruning (λ={prune_lambda})")
        print("-" * 70)
        
        clf_pruned = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 4},
            transform_params={
                "prune_atoms": True,
                "prune_lambda": prune_lambda,
            },
            random_state=42,
        )
        
        try:
            clf_pruned.fit(X_train, y_train)
            ruleset_pruned = clf_pruned.to_ruleset()
            
            n_rules_pruned = len(ruleset_pruned.rules)
            n_atoms_pruned = sum(len(r.atoms) for r in ruleset_pruned.rules)
            f1_pruned = f1_score(y_test, clf_pruned.predict(X_test), average='macro')
            
            atoms_removed = n_atoms_baseline - n_atoms_pruned
            pct_removed = (atoms_removed / n_atoms_baseline * 100) if n_atoms_baseline > 0 else 0
            f1_delta = f1_pruned - f1_baseline
            
            print(f"  Rules:        {n_rules_pruned}")
            print(f"  Total Atoms:  {n_atoms_pruned} (↓ {atoms_removed}, -{pct_removed:.1f}%)")
            print(f"  F1 (macro):   {f1_pruned:.4f} (Δ {f1_delta:+.4f})")
            
            results.append({
                'lambda': prune_lambda,
                'n_atoms': n_atoms_pruned,
                'atoms_removed': atoms_removed,
                'pct_removed': pct_removed,
                'f1': f1_pruned,
                'f1_delta': f1_delta,
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Zusammenfassung
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"  Baseline: {n_atoms_baseline} atoms, F1={f1_baseline:.4f}")
    for r in results:
        print(f"  λ={r['lambda']}: {r['n_atoms']} atoms ({r['pct_removed']:.0f}% reduction), "
              f"F1={r['f1']:.4f} ({r['f1_delta']:+.4f})")


def main():
    # Lade mehrere Datensätze
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
    ]
    
    for name, X, y in datasets:
        benchmark_pruning_on_dataset(name, X, y)
    
    print(f"\n{'='*70}")
    print("Benchmark abgeschlossen!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

