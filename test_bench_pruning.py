#!/usr/bin/env python3
"""
Benchmark-Test für Rule-Shrinking mit verschiedenen Lambda-Werten
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from sklearn.datasets import load_iris, load_wine
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    
    from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier
    
    print("=" * 80)
    print("Benchmark: Rule-Shrinking mit verschiedenen Lambda-Werten")
    print("=" * 80)
    
    # Lade Iris-Datensatz
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42
    )
    
    print("\nDataset: Iris (n_samples={}, n_features={}, n_classes={})".format(
        X_train.shape[0], X_train.shape[1], len(set(y_train))
    ))
    
    # Test ohne Pruning
    print("\n" + "-" * 80)
    print("BASELINE: Ohne Atom-Pruning")
    print("-" * 80)
    
    clf_baseline = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 4},
        random_state=42,
    )
    clf_baseline.fit(X_train, y_train)
    ruleset_baseline = clf_baseline.to_ruleset()
    
    n_rules_baseline = len(ruleset_baseline.rules)
    n_atoms_baseline = sum(len(rule.atoms) for rule in ruleset_baseline.rules)
    f1_baseline = f1_score(y_test, clf_baseline.predict(X_test), average='macro')
    
    print(f"Rules:   {n_rules_baseline}")
    print(f"Atoms:   {n_atoms_baseline}")
    print(f"F1 (macro): {f1_baseline:.4f}")
    
    # Test mit verschiedenen Lambda-Werten
    for prune_lambda in [1.5, 2.0, 3.0]:
        print("\n" + "-" * 80)
        print(f"Mit Atom-Pruning (λ={prune_lambda})")
        print("-" * 80)
        
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
            n_atoms_pruned = sum(len(rule.atoms) for rule in ruleset_pruned.rules)
            f1_pruned = f1_score(y_test, clf_pruned.predict(X_test), average='macro')
            
            atoms_removed = n_atoms_baseline - n_atoms_pruned
            pct_removed = (atoms_removed / n_atoms_baseline * 100) if n_atoms_baseline > 0 else 0
            
            print(f"Rules:   {n_rules_pruned}")
            print(f"Atoms:   {n_atoms_pruned} (↓ {atoms_removed}, -{pct_removed:.1f}%)")
            print(f"F1 (macro): {f1_pruned:.4f} (Δ {f1_pruned - f1_baseline:+.4f})")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Benchmark abgeschlossen")
    print("=" * 80)

