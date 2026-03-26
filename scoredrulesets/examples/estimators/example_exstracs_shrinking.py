#!/usr/bin/env python3
"""
ExSTraCS rule-shrinking example

Demonstrates different strategies to reduce large ExSTraCS rule sets:
1. Conservative pruning - guarantees no degradation
2. Aggressive pruning - uses validation data, accepts up to 1% F1 loss
3. Weak rule filtering - removes weak rules
4. Rule consolidation - merges similar rules
5. All - combines all strategies
"""

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier


def benchmark_exstracs_shrinking():
    """Compare ExSTraCS with different shrinking strategies."""
    
    print("=" * 80)
    print("ExSTraCS Rule-Shrinking Benchmark")
    print("=" * 80)
    
    datasets = [
        ("Iris", load_iris(), 100),
        ("Wine", load_wine(), 100),
        ("Breast Cancer", load_breast_cancer(), 200),
    ]
    
    shrinking_variants = [
        ("exstracs_baseline", None, "Baseline (no shrinking)"),
        ("exstracs_conservative", {"conservative_prune": True}, "Conservative Pruning"),
        ("exstracs_filter", {"filter_weak_rules": True, "min_fitness_percentile": 0.2}, "Filter Weak Rules"),
        ("exstracs_aggressive", {"aggressive_prune": True, "max_f1_loss": 0.01}, "Aggressive Pruning (1% loss)"),
        ("exstracs_all", {
            "conservative_prune": True,
            "filter_weak_rules": True,
            "consolidate_similar": True,
            "aggressive_prune": True,
            "max_f1_loss": 0.01,
        }, "All Strategies"),
    ]
    
    for dataset_name, dataset, train_size in datasets:
        X, y = dataset.data, dataset.target

        # Stratified split: ensure all classes are represented in training
        if train_size < len(X):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, stratify=y, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = np.empty((0, X.shape[1])), np.empty((0,))

        if len(X_test) == 0 or len(y_test) == 0:
            print(f"[WARN] Too few test samples for {dataset_name}, skipping.")
            continue

        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"{'='*80}")
        
        results = []
        
        for variant_name, shrinking_params, description in shrinking_variants:
            print(f"\n{description:40}", end=" ", flush=True)
            
            try:
                clf = ScoredRuleSetClassifier(
                    backend="exstracs",
                    backend_params={},
                    exstracs_params=shrinking_params,
                    random_state=42,
                )
                # Train
                clf.fit(X_train, y_train)

                # F1 with native ExSTraCS model (before transformation)
                estimator = getattr(clf, "estimator_", None)
                if estimator is not None and hasattr(estimator, "predict"):
                    try:
                        y_pred_native = estimator.predict(X_test)
                        # Debug output when F1=0.0 or an error occurs
                        import numpy as np
                        debug_native = False
                        try:
                            f1_native = f1_score(y_test, y_pred_native, average='macro', zero_division=0)
                            if f1_native == 0.0:
                                debug_native = True
                        except Exception as e:
                            print(f"  [F1 native ExSTraCS: Error: {e}]", end=" ")
                            debug_native = True
                            f1_native = None
                        if debug_native:
                            print("\n  [DEBUG native] y_test:", y_test)
                            print("  [DEBUG native] y_pred_native:", y_pred_native)
                            print("  [DEBUG native] y_test type:", type(y_test), "len:", len(y_test), "unique:", np.unique(y_test))
                            print("  [DEBUG native] y_pred_native type:", type(y_pred_native), "len:", len(y_pred_native), "unique:", np.unique(y_pred_native))
                        if f1_native is not None:
                            print(f"  [F1 native ExSTraCS: {f1_native:.4f}]", end=" ")
                    except Exception as e:
                        print(f"  [F1 native ExSTraCS: Error: {e}]", end=" ")

                # Prediction with ScoredRuleSet
                y_pred = clf.predict(X_test)
                # Debug output when F1=0.0 or an error occurs
                import numpy as np
                debug_scored = False
                try:
                    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    if f1 == 0.0:
                        debug_scored = True
                except Exception as e:
                    print(f"  [F1 scoredruleset: Error: {e}]", end=" ")
                    debug_scored = True
                    f1 = None
                if debug_scored:
                    print("\n  [DEBUG scoredruleset] y_test:", y_test)
                    print("  [DEBUG scoredruleset] y_pred:", y_pred)
                    print("  [DEBUG scoredruleset] y_test type:", type(y_test), "len:", len(y_test), "unique:", np.unique(y_test))
                    print("  [DEBUG scoredruleset] y_pred type:", type(y_pred), "len:", len(y_pred), "unique:", np.unique(y_pred))

                # Ruleset statistics
                ruleset = clf.to_ruleset()
                n_rules = len(ruleset.rules)
                n_atoms = sum(len(r.atoms) for r in ruleset.rules)
                avg_atoms = n_atoms / max(n_rules, 1)

                # Debug: count rules with atoms=0 and with atoms>0
                n_default = sum(1 for r in ruleset.rules if len(r.atoms) == 0)
                n_nondefault = sum(1 for r in ruleset.rules if len(r.atoms) > 0)
                print(f"  [DEBUG] Default rules: {n_default}, non-default rules: {n_nondefault}")

                print(f"✓ F1 scoredruleset={f1:.4f} | Rules={n_rules:3d} | Atoms={n_atoms:4d} | AvgAtoms={avg_atoms:.1f}")

                results.append({
                    'variant': variant_name,
                    'f1': f1,
                    'n_rules': n_rules,
                    'n_atoms': n_atoms,
                    'avg_atoms': avg_atoms,
                })

            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}...")
        
        # Summary
        if results:
            print(f"\n{'-'*80}")
            print("Summary:")
            
            # Comparison against baseline
            baseline = results[0]
            print(f"\nBaseline (no shrinking):")
            print(f"  F1={baseline['f1']:.4f}, Rules={baseline['n_rules']}, Atoms={baseline['n_atoms']}")
            
            print(f"\nReductions:")
            for result in results[1:]:
                atom_reduction = (1 - result['n_atoms'] / baseline['n_atoms']) * 100 if baseline['n_atoms'] > 0 else 0
                f1_diff = result['f1'] - baseline['f1']
                print(f"  {result['variant']:35s}: Atoms {atom_reduction:5.1f}% ↓, F1 {f1_diff:+.4f}")


def test_individual_strategy():
    """Test a single strategy with details."""
    
    print("\n" + "=" * 80)
    print("Test: Aggressive Pruning Details")
    print("=" * 80)
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("\nBaseline (no shrinking):")
    clf_baseline = ScoredRuleSetClassifier(backend="exstracs", random_state=42)
    clf_baseline.fit(X_train, y_train)
    ruleset_baseline = clf_baseline.to_ruleset()
    y_pred_baseline = clf_baseline.predict(X_test)
    import numpy as np
    debug_baseline = False
    try:
        f1_baseline = f1_score(y_test, y_pred_baseline, average='macro')
        if f1_baseline == 0.0:
            debug_baseline = True
    except Exception as e:
        print(f"  [F1 baseline: Error: {e}]", end=" ")
        debug_baseline = True
        f1_baseline = None
    if debug_baseline:
        print("\n  [DEBUG baseline] y_test:", y_test)
        print("  [DEBUG baseline] y_pred:", y_pred_baseline)
        print("  [DEBUG baseline] y_test type:", type(y_test), "len:", len(y_test), "unique:", np.unique(y_test))
        print("  [DEBUG baseline] y_pred type:", type(y_pred_baseline), "len:", len(y_pred_baseline), "unique:", np.unique(y_pred_baseline))
    print(f"  Rules: {len(ruleset_baseline.rules)}")
    print(f"  Total Atoms: {sum(len(r.atoms) for r in ruleset_baseline.rules)}")
    if f1_baseline is not None:
        print(f"  F1: {f1_baseline:.4f}")
    
    print("\nWith aggressive pruning (max_f1_loss=1%):")
    clf_aggressive = ScoredRuleSetClassifier(
        backend="exstracs",
        exstracs_params={
            "aggressive_prune": True,
            "max_f1_loss": 0.01,
        },
        random_state=42,
    )
    clf_aggressive.fit(X_train, y_train)
    ruleset_aggressive = clf_aggressive.to_ruleset()
    y_pred_aggressive = clf_aggressive.predict(X_test)
    debug_aggressive = False
    try:
        f1_aggressive = f1_score(y_test, y_pred_aggressive, average='macro')
        if f1_aggressive == 0.0:
            debug_aggressive = True
    except Exception as e:
        print(f"  [F1 aggressive: Error: {e}]", end=" ")
        debug_aggressive = True
        f1_aggressive = None
    if debug_aggressive:
        print("\n  [DEBUG aggressive] y_test:", y_test)
        print("  [DEBUG aggressive] y_pred:", y_pred_aggressive)
        print("  [DEBUG aggressive] y_test type:", type(y_test), "len:", len(y_test), "unique:", np.unique(y_test))
        print("  [DEBUG aggressive] y_pred type:", type(y_pred_aggressive), "len:", len(y_pred_aggressive), "unique:", np.unique(y_pred_aggressive))
    print(f"  Rules: {len(ruleset_aggressive.rules)}")
    print(f"  Total Atoms: {sum(len(r.atoms) for r in ruleset_aggressive.rules)}")
    if f1_aggressive is not None:
        print(f"  F1: {f1_aggressive:.4f}")
    
    atom_reduction = (1 - sum(len(r.atoms) for r in ruleset_aggressive.rules) / 
                     sum(len(r.atoms) for r in ruleset_baseline.rules)) * 100
    f1_diff = f1_aggressive - f1_baseline
    
    print(f"\nSavings:")
    print(f"  Atoms: {atom_reduction:.1f}% Reduktion")
    print(f"  F1: {f1_diff:+.4f} (Ziel war max -0.01)")


if __name__ == "__main__":
    print("\nExSTraCS rule-shrinking examples\n")
    
    # Run benchmark
    try:
        benchmark_exstracs_shrinking()
    except Exception as e:
        print(f"\n[WARN] Benchmark error: {e}")
        print("Ensure skExSTraCS is installed: pip install scikit-exstracs")
    
    # Run individual test
    try:
        test_individual_strategy()
    except Exception as e:
        print(f"\n[WARN] Test error: {e}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80 + "\n")

