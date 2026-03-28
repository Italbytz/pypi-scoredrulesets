#!/usr/bin/env python3
"""
Validierungs-Skript für Rule-Shrinking Implementation
"""

import sys
from pathlib import Path

# Füge src zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_imports():
    """Überprüfe ob alle neuen Module importierbar sind"""
    print("=" * 70)
    print("IMPORT-ÜBERPRÜFUNG")
    print("=" * 70)
    
    checks = [
        ("TreeTransformParams", "scoredrulesets.estimators.tree_transform"),
        ("_aggressive_atom_pruning", "scoredrulesets.estimators.tree_transform"),
        ("_can_remove_atom_safely", "scoredrulesets.estimators.tree_transform"),
        ("ScoredRuleSetClassifier", "scoredrulesets.estimators.sklearn_wrapper"),
        ("default_estimator_specs", "scoredrulesets.benchmarking.estimators"),
    ]
    
    all_ok = True
    for name, module_path in checks:
        try:
            module = __import__(module_path, fromlist=[name])
            getattr(module, name)
            print(f"✓ {name:30} von {module_path}")
        except Exception as e:
            print(f"✗ {name:30} von {module_path}")
            print(f"  Error: {e}")
            all_ok = False
    
    return all_ok


def check_estimators():
    """Überprüfe ob neue Estimator-Specs vorhanden sind"""
    print("\n" + "=" * 70)
    print("ESTIMATOR-ÜBERPRÜFUNG")
    print("=" * 70)
    
    from scoredrulesets.benchmarking.estimators import default_estimator_specs
    
    specs = default_estimator_specs()
    new_specs = [k for k in specs.keys() if "pruned" in k]
    
    print(f"Gesamt Estimators: {len(specs)}")
    print(f"Mit Pruning: {len(new_specs)}")
    print("\nNeu hinzugefügte Estimators:")
    
    for name in sorted(new_specs):
        print(f"  ✓ {name}")
    
    return len(new_specs) >= 4  # Sollte mindestens 4 neue haben


def check_parameters():
    """Überprüfe TreeTransformParams"""
    print("\n" + "=" * 70)
    print("PARAMETER-ÜBERPRÜFUNG")
    print("=" * 70)
    
    from scoredrulesets.estimators.tree_transform import TreeTransformParams
    
    # Erstelle Instanz mit neuen Parametern
    try:
        params = TreeTransformParams(
            depth_decay_lambda=2.0,
            prune_atoms=True,
            prune_lambda=2.0,
        )
        print(f"✓ TreeTransformParams mit neuen Parametern:")
        print(f"  - prune_atoms: {params.prune_atoms}")
        print(f"  - prune_lambda: {params.prune_lambda}")
        print(f"  - depth_decay_lambda: {params.depth_decay_lambda}")
        return True
    except Exception as e:
        print(f"✗ Fehler beim Erstellen von TreeTransformParams: {e}")
        return False


def main():
    print("\n🔍 Rule-Shrinking Implementation - Validierungs-Skript\n")
    
    results = []
    
    # Überprüfungen
    results.append(("Imports", check_imports()))
    results.append(("Estimators", check_estimators()))
    results.append(("Parameter", check_parameters()))
    
    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:10} - {name}")
    
    all_ok = all(ok for _, ok in results)
    
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ ALLE ÜBERPRÜFUNGEN ERFOLGREICH")
        print("Implementation ist bereit für Benchmarks!")
    else:
        print("❌ EINIGE ÜBERPRÜFUNGEN FEHLGESCHLAGEN")
        print("Bitte überprüfe die Implementierung")
    print("=" * 70 + "\n")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

