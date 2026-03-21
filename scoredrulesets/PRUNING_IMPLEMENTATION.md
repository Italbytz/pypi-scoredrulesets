# Rule-Shrinking Algorithmus Implementation

## Zusammenfassung

Ich habe den aggressiven Atom-Pruning-Algorithmus aus dem Paper implementiert, der auskommentiert war. Dieser Algorithmus reduziert die Modellgröße für CART und HS Estimators durch intelligente Atom-Entfernung.

## Was wurde hinzugefügt

### 1. **TreeTransformParams erweitert** (`tree_transform.py`)
- Neue Parameter:
  - `prune_atoms: bool` - Aktiviert Atom-Pruning
  - `prune_lambda: float | None` - Lambda-Decay-Parameter für Depth-Weighting

### 2. **_aggressive_atom_pruning() Funktion**
Implementiert den Hauptalgorithmus mit:
- Iterative Atom-Entfernung
- Sicherheitsüberprüfungen für Score-Validität
- Rückwärts-Iteration für stabile Indizes
- Konservative Heuristiken zur Vermeidung von Overpruning

### 3. **_can_remove_atom_safely() Funktion**
Validiert sichere Atom-Entfernung mit 3 Kriterien:
1. Weniger Atome im Kandidaten
2. Mindestens ein positiver Score bleibt
3. Regeln werden nicht komplett aufgelöst (außer Default-Regel)

### 4. **Benchmark-Estimators** (`benchmarking/estimators.py`)
Neue Estimator-Konfigurationen hinzugefügt:
- `wrapper_cart_pruned_λ1.5` - CART mit λ=1.5
- `wrapper_cart_pruned_λ2.0` - CART mit λ=2.0
- `wrapper_cart_pruned_λ3.0` - CART mit λ=3.0
- `wrapper_hs_pruned_λ2.0` - HS mit λ=2.0

## Erwartete Ergebnisse

Die neuen Estimators sollten deutlich **kompaktere Modelle** mit **weniger Atomen** produzieren:

### Baseline (ohne Pruning):
- CART: ~23-39 Atome
- HS: ~20-40+ Atome (abhängig von Datensatz)

### Mit Pruning (λ=2.0):
- **Reduktion von 30-60%** der Atome erwartet
- Vorhersage-Qualität sollte gleich bleiben oder sogar verbessern
- Modelle werden viel interpretierbarer

## Parameter-Abstimmung

- **λ=1.5**: Konservativ, entfernt weniger Atome, höhere Vorhersage-Äquivalenz
- **λ=2.0**: Balanced, gutes Verhältnis Kompaktheit/Qualität
- **λ=3.0**: Aggressiv, maximale Atom-Entfernung

Höhere λ-Werte erlauben mehr Atom-Entfernung, da tiefere Splits (mit kleinerem Gewicht) leichter optimiert werden können.

## Benchmark durchführen

```bash
cd /Users/nunkesser/repos/work/artifacts/pypi-scoredrulesets/scoredrulesets

# Nur neue pruned Modelle testen
python examples/benchmark_runner.py \
  --datasets sklearn_iris,sklearn_wine,sklearn_breast_cancer \
  --estimators wrapper_cart,wrapper_cart_pruned_λ1.5,wrapper_cart_pruned_λ2.0,wrapper_cart_pruned_λ3.0 \
  --repeats 3 \
  --aggregate-repeats \
  --output-csv benchmarks/2026-03-pruning/benchmark_results.csv \
  --output-markdown benchmarks/2026-03-pruning/benchmark_report.md

# Mit HS auch vergleichen
python examples/benchmark_runner.py \
  --datasets sklearn_iris,sklearn_wine,sklearn_breast_cancer \
  --estimators wrapper_hs,wrapper_hs_pruned_λ2.0 \
  --repeats 3 \
  --aggregate-repeats \
  --output-csv benchmarks/2026-03-hs-pruning/benchmark_results.csv
```

## Änderungen-Übersicht

### Geänderte Dateien:
1. `src/scoredrulesets/estimators/tree_transform.py`
   - TreeTransformParams erweitert
   - Pruning-Algorithmen hinzugefügt
   - Integration in estimator_to_scored_ruleset()

2. `src/scoredrulesets/benchmarking/estimators.py`
   - 5 neue Estimator-Konfigurationen mit Pruning-Parametern

### Neue Test-Dateien (optional):
- `test_pruning.py` - Umfassendes Test-Skript
- `test_simple_pruning.py` - Einfacher Test
- `test_bench_pruning.py` - Benchmark-Vergleich

## Nächste Schritte

1. Führe Benchmarks durch, um Atom-Reduktion zu messen
2. Vergleiche F1-Scores für verschiedene λ-Werte
3. Optimiere λ-Wert basierend auf Accuraccy/Kompaktheit Trade-off
4. Erwäge adaptive λ-Bestimmung basierend auf Datensatz-Charakteristika

