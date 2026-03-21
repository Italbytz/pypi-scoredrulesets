#!/usr/bin/env python3
"""
Validierungs-Skript für RuleKit und ExSTraCS Integration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_backends():
    """Überprüfe ob neue Backends verfügbar sind"""
    print("=" * 70)
    print("BACKEND-VALIDIERUNG")
    print("=" * 70)
    
    from scoredrulesets.estimators.backends import build_backend_estimator
    
    backends_to_test = [
        ("cart", "Decision Tree (sklearn)"),
        ("hs", "Hierarchical Shrinkage (imodels)"),
        ("rulekit", "RuleKit (Java erforderlich)"),
        ("exstracs", "ExSTraCS (skExSTraCS)"),
    ]
    
    print("\nBackend-Verfügbarkeit:")
    print("-" * 70)
    
    results = {}
    for backend_name, description in backends_to_test:
        try:
            # Versuche Backend zu bauen
            cls = build_backend_estimator(backend_name, {}, random_state=0)
            print(f"✓ {backend_name:12} - {description:40} VERFÜGBAR")
            results[backend_name] = True
        except ImportError as e:
            print(f"✗ {backend_name:12} - {description:40}")
            print(f"    Fehler: {str(e)[:60]}...")
            results[backend_name] = False
        except Exception as e:
            print(f"✗ {backend_name:12} - {description:40}")
            print(f"    Unerwarteter Fehler: {type(e).__name__}")
            results[backend_name] = False
    
    return results


def check_estimators():
    """Überprüfe ob Estimator-Specs vorhanden sind"""
    print("\n" + "=" * 70)
    print("ESTIMATOR-SPECS VALIDIERUNG")
    print("=" * 70)
    
    from scoredrulesets.benchmarking.estimators import default_estimator_specs
    
    specs = default_estimator_specs()
    
    required_specs = [
        ("wrapper_cart", "CART Wrapper"),
        ("wrapper_hs", "HS Wrapper"),
        ("wrapper_rulekit", "RuleKit Wrapper (NEU)"),
        ("wrapper_exstracs", "ExSTraCS Wrapper (NEU)"),
    ]
    
    print("\nEstimator-Specs:")
    print("-" * 70)
    
    results = {}
    for spec_name, description in required_specs:
        if spec_name in specs:
            print(f"✓ {spec_name:20} - {description}")
            results[spec_name] = True
        else:
            print(f"✗ {spec_name:20} - {description}")
            results[spec_name] = False
    
    print(f"\nGesamt verfügbare Estimators: {len(specs)}")
    return results


def check_transformations():
    """Überprüfe ob Transformationsfunktionen vorhanden sind"""
    print("\n" + "=" * 70)
    print("TRANSFORMATIONS-VALIDIERUNG")
    print("=" * 70)
    
    print("\nTransformationsfunktionen:")
    print("-" * 70)
    
    results = {}
    
    # Überprüfe tree_transform
    try:
        from scoredrulesets.estimators.tree_transform import (
            estimator_to_scored_ruleset,
            _aggressive_atom_pruning,
            TreeTransformParams,
        )
        print("✓ tree_transform.py:")
        print("  - estimator_to_scored_ruleset")
        print("  - _aggressive_atom_pruning (Rule-Shrinking)")
        print("  - TreeTransformParams")
        results["tree_transform"] = True
    except Exception as e:
        print(f"✗ tree_transform.py - Fehler: {e}")
        results["tree_transform"] = False
    
    # Überprüfe ruleset_transform (NEU)
    try:
        from scoredrulesets.estimators.ruleset_transform import (
            rulekit_to_scored_ruleset,
            exstracs_to_scored_ruleset,
        )
        print("✓ ruleset_transform.py (NEU):")
        print("  - rulekit_to_scored_ruleset")
        print("  - exstracs_to_scored_ruleset")
        results["ruleset_transform"] = True
    except Exception as e:
        print(f"✗ ruleset_transform.py - Fehler: {e}")
        results["ruleset_transform"] = False
    
    return results


def check_sklearn_wrapper():
    """Überprüfe ob Wrapper richtig aktualisiert ist"""
    print("\n" + "=" * 70)
    print("SKLEARN_WRAPPER VALIDIERUNG")
    print("=" * 70)
    
    try:
        from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier
        
        # Überprüfe dass Imports vorhanden sind
        import inspect
        source = inspect.getsource(ScoredRuleSetClassifier.fit)
        
        checks = [
            ("rulekit_to_scored_ruleset Import", "rulekit_to_scored_ruleset" in source),
            ("exstracs_to_scored_ruleset Import", "exstracs_to_scored_ruleset" in source),
            ("Backend-Selection", "backend_lower" in source),
        ]
        
        print("\nSklearn Wrapper Aktualisierungen:")
        print("-" * 70)
        
        results = {}
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"{status} {check_name}")
            results[check_name] = passed
        
        return results
    except Exception as e:
        print(f"✗ Fehler beim Überprüfen: {e}")
        return {}


def main():
    print("\n🔍 RuleKit & ExSTraCS Integration - Validierungs-Skript\n")
    
    all_results = {}
    
    # Durchführe alle Überprüfungen
    all_results["Backends"] = check_backends()
    all_results["Estimators"] = check_estimators()
    all_results["Transformations"] = check_transformations()
    all_results["Wrapper"] = check_sklearn_wrapper()
    
    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    
    total_checks = sum(len(results) for results in all_results.values())
    passed_checks = sum(
        sum(1 for v in results.values() if v) 
        for results in all_results.values()
    )
    
    print(f"\nBestanden: {passed_checks}/{total_checks}")
    
    # Details pro Kategorie
    for category, results in all_results.items():
        if results:
            category_passed = sum(1 for v in results.values() if v)
            category_total = len(results)
            status = "✓" if category_passed == category_total else "⚠"
            print(f"{status} {category:20} {category_passed}/{category_total}")
    
    # Final-Status
    print("\n" + "=" * 70)
    if passed_checks == total_checks:
        print("✅ ALLE VALIDIERUNGEN ERFOLGREICH!")
        print("Integration ist bereit für Benchmarks!")
    else:
        print(f"⚠️  {total_checks - passed_checks} Fehler gefunden")
        print("Bitte überprüfen Sie die fehlgeschlagenen Checks")
    print("=" * 70 + "\n")
    
    return 0 if passed_checks == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())

