#!/usr/bin/env python3
"""
Beispiel: RuleKit und ExSTraCS als Backends

Dieses Skript demonstriert:
1. RuleKit Integration (mit Java-Validierung)
2. ExSTraCS Integration (skExSTraCS)
3. Vergleich aller Backends
4. Fehlerbehandlung
"""

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


def test_backend(name, backend, dataset_name, X_train, y_train, X_test, y_test):
    """Test ein Backend"""
    print(f"\n  {name:20}", end=" ", flush=True)
    
    try:
        clf = ScoredRuleSetClassifier(
            backend=backend,
            backend_params={},
            random_state=42,
        )
        
        # Trainiere
        clf.fit(X_train, y_train)
        
        # Vorhersage
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Ruleset Statistiken
        ruleset = clf.to_ruleset()
        n_rules = len(ruleset.rules)
        n_atoms = sum(len(r.atoms) for r in ruleset.rules)
        
        print(f"✓ F1={f1:.4f} | Rules={n_rules} | Atoms={n_atoms}")
        return True
        
    except ImportError as e:
        print(f"✗ ImportError: {str(e)[:60]}...")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)[:60]}...")
        return False


def main():
    print("=" * 80)
    print("Test: RuleKit und ExSTraCS als Backends")
    print("=" * 80)
    
    # Lade Datensätze
    iris = load_iris()
    wine = load_wine()
    
    datasets = [
        ("Iris", iris.data, iris.target),
        ("Wine", wine.data, wine.target),
    ]
    
    # Test Backends
    backends = [
        ("CART (Tree)", "cart"),
        ("HS (Optimized Tree)", "hs"),
        ("RuleKit", "rulekit"),
        ("ExSTraCS", "exstracs"),
    ]
    
    for dataset_name, X, y in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name} (n={len(X_train)}, features={X.shape[1]})")
        print(f"{'='*80}")
        
        results = []
        for name, backend in backends:
            ok = test_backend(name, backend, dataset_name, X_train, y_train, X_test, y_test)
            results.append((name, ok))
        
        # Zusammenfassung
        print(f"\n{'-'*80}")
        print("Zusammenfassung:")
        for name, ok in results:
            status = "✓" if ok else "✗"
            print(f"  {status} {name}")
    
    print(f"\n{'='*80}")
    print("Test abgeschlossen!")
    print(f"{'='*80}\n")
    
    print("Hinweise:")
    print("- RuleKit benötigt Java (JDK 11+) und 'pip install rulekit'")
    print("- ExSTraCS benötigt 'pip install scikit-exstracs'")
    print("- CART und HS sollten immer verfügbar sein (sklearn)")


if __name__ == "__main__":
    main()

