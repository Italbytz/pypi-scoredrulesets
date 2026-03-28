#!/usr/bin/env python3
"""
Test-Skript zur Validierung des Atom-Pruning-Algorithmus mit verschiedenen Lambda-Werten
"""

import sys
from pathlib import Path

# Füge src zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier
from scoredrulesets.estimators.tree_transform import TreeTransformParams


def test_pruning():
    """Test Atom-Pruning mit verschiedenen Lambda-Werten"""
    
    # Lade Iris-Datensatz
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("=" * 80)
    print("Test: Atom-Pruning mit verschiedenen Lambda-Werten")
    print("=" * 80)
    
    # Test ohne Pruning
    print("\n1. BASELINE: Ohne Atom-Pruning")
    print("-" * 80)
    estimator_baseline = ScoredRuleSetClassifier(
        backend="cart",
        backend_params={"max_depth": 4},
        random_state=42,
    )
    estimator_baseline.fit(X_train, y_train)
    ruleset_baseline = estimator_baseline.to_ruleset()
    
    n_rules_baseline = len(ruleset_baseline.rules)
    n_atoms_baseline = sum(len(rule.atoms) for rule in ruleset_baseline.rules)
    score_baseline = estimator_baseline.score(X_test, y_test)
    
    print(f"  Rules:         {n_rules_baseline}")
    print(f"  Total Atoms:   {n_atoms_baseline}")
    print(f"  Accuracy:      {score_baseline:.4f}")
    
    # Test mit verschiedenen Lambda-Werten
    for prune_lambda in [1.5, 2.0, 3.0]:
        print(f"\n2. Mit Atom-Pruning (λ={prune_lambda})")
        print("-" * 80)
        
        estimator_pruned = ScoredRuleSetClassifier(
            backend="cart",
            backend_params={"max_depth": 4},
            transform_params={
                "prune_atoms": True,
                "prune_lambda": prune_lambda,
            },
            random_state=42,
        )
        
        try:
            estimator_pruned.fit(X_train, y_train)
            ruleset_pruned = estimator_pruned.to_ruleset()
            
            n_rules_pruned = len(ruleset_pruned.rules)
            n_atoms_pruned = sum(len(rule.atoms) for rule in ruleset_pruned.rules)
            score_pruned = estimator_pruned.score(X_test, y_test)
            
            atoms_removed = n_atoms_baseline - n_atoms_pruned
            atoms_pct = (atoms_removed / n_atoms_baseline * 100) if n_atoms_baseline > 0 else 0
            
            print(f"  Rules:         {n_rules_pruned}")
            print(f"  Total Atoms:   {n_atoms_pruned} (↓ {atoms_removed}, -{atoms_pct:.1f}%)")
            print(f"  Accuracy:      {score_pruned:.4f} (Δ {score_pruned - score_baseline:+.4f})")
            
            # Zeige Details der Regeln
            print(f"\n  Rule Details:")
            for i, rule in enumerate(ruleset_pruned.rules):
                print(f"    Rule {i}: {len(rule.atoms)} atoms, scores={rule.scores}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test abgeschlossen!")
    print("=" * 80)


if __name__ == "__main__":
    test_pruning()

