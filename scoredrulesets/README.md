# scoredrulesets

Scikit-learn-kompatible Grundstruktur fuer Klassifikation mit **Scored Rule Sets**.

## Ziele

- Einheitliches, allgemeines JSON-Format fuer Scored Rule Sets
- Import bestehender `logicGP`-Modelle in dieses Format
- sklearn-kompatibler Wrapper, der externe Schaetzer trainiert und post-hoc in Scored Rule Sets transformiert
- einfacher Standardaufruf (`backend="hs"`, konfigurierbar)
- Erweiterungspunkt fuer kuenftige native Schaetzer im Paket

## Installation

```bash
pip install -e .
```

Mit HS-Backend (imodels):

```bash
pip install -e '.[hs]'
```

Fuer Entwicklung/Tests:

```bash
pip install -e '.[dev]'
```

Fuer Benchmarking:

```bash
pip install -e '.[benchmark]'
```

Optional fuer lokale UCI-CSV-Benchmarks:

```bash
pip install -e '.[benchmark-uci]'
```

## Schnellstart

```python
from sklearn.datasets import load_iris
from scoredrulesets import ScoredRuleSetClassifier, dump_ruleset_json

X, y = load_iris(return_X_y=True)

# Standard ist backend="hs" (benoetigt imodels)
clf = ScoredRuleSetClassifier(backend="cart")
clf.fit(X, y)

ruleset = clf.to_ruleset()
dump_ruleset_json(ruleset, "iris_ruleset.json")
```

Konsolenlesbare Tabellen-Ausgabe:

```python
from scoredrulesets import format_ruleset_table

print(format_ruleset_table(ruleset))
```

## logicGP-Import

```python
from scoredrulesets import import_logicgp_json, dump_ruleset_json

ruleset = import_logicgp_json("ext/logicgp_model.json")
dump_ruleset_json(ruleset, "converted_ruleset.json")
```

## Nativer Estimator (Skeleton)

```python
from sklearn.datasets import load_iris
from scoredrulesets import NativeScoredRuleSetClassifier

X, y = load_iris(return_X_y=True)

clf = NativeScoredRuleSetClassifier()
clf.fit(X, y)
print(clf.predict(X[:3]))

# Optional: einfache native Regelkomplexitaet steuern
# NativeScoredRuleSetClassifier(max_rules=6, min_samples_leaf=5)
# Optional: kategoriale Regelgenerierung abschalten
# NativeScoredRuleSetClassifier(enable_categorical_rules=False)
```

Der native Lerner erzeugt je nach Datenlage Atome mit `<=`, `>`, `between`, `==` und `in`.

## Nativer GP-Estimator

```python
from sklearn.datasets import load_iris
from scoredrulesets import GeneticScoredRuleSetClassifier

X, y = load_iris(return_X_y=True)

gp = GeneticScoredRuleSetClassifier(
    population_size=40,
    generations=20,
    max_rules=6,
    score_mode="auto",  # "auto" | "log_proba" | "proba"
    selection_mode="fitness",  # "fitness" | "pareto"
    validation_fraction=0.2,
    early_stopping_rounds=5,
    random_state=0,
)
gp.fit(X, y)
print(gp.predict(X[:3]))
```

Der GP-Lerner durchsucht ebenfalls Atome mit `<=`, `>`, `between`, `==` und `in`.

## Benchmarking

Benchmarking vergleicht die Estimatoren ueber:

- `f1_macro`
- Modellgroesse (`n_rules`, `n_atoms`, `ruleset_json_bytes`)
- Laufzeit (`fit_seconds`, `predict_seconds`)

Datensaetze:

- Standard: `sklearn_iris`, `sklearn_wine`, `sklearn_breast_cancer`
- Optional lokal: CSV-Dateien aus `SCORERULESETS_UCI_DIR` (letzte Spalte = Ziel)

Beispielaufruf:

```bash
python examples/benchmark_runner.py \
  --datasets sklearn_iris,sklearn_wine \
  --estimators wrapper_cart,native,gp,wrapper_hs \
  --repeats 2 \
  --aggregate-repeats \
  --error-bar std \
  --leaderboard-primary-metric f1_macro_mean \
  --output-markdown benchmark_leaderboard.md \
  --output-html benchmark_report.html \
  --output-csv benchmark_results.csv \
  --output-json benchmark_results.json \
  --output-plot-base benchmark_results \
  --plot-size-metric n_rules
```

Dabei werden standardmaessig folgende Dateien geschrieben:

- `benchmark_results.csv`
- `benchmark_results.json`
- `benchmark_results_aggregated.csv` (bei `--aggregate-repeats`)
- `benchmark_results_aggregated.json` (bei `--aggregate-repeats`)
- `benchmark_leaderboard.md` (bei `--aggregate-repeats`)
- `benchmark_report.html` (optional)
- `benchmark_results.png`
- `benchmark_results.pdf`

Der Plot zeigt `f1_macro` gegen eine gewaehlte Modellgroessenmetrik (`n_rules`, `n_atoms` oder `ruleset_json_bytes`); die Farbe kodiert die Fit-Zeit.
Mit `--aggregate-repeats` zeigt der Plot Mittelwerte pro `(dataset, estimator)` und Fehlerbalken (`std` oder `sem`).
Das Leaderboard sortiert aggregierte Ergebnisse standardmaessig nach `f1_macro_mean`.
`--output-markdown` erzeugt einen kompakten Markdown-Report mit Konfiguration, Artefaktverweisen und eingebettetem Leaderboard.
Zusaetzlich enthaelt der Report eine globale "Top per Dataset"-Uebersicht vor dem Gesamt-Leaderboard.
Pro Datensatz enthaelt der Report zusaetzlich automatische Summary-Punkte fuer bestes Modell, kleinste Modellgroesse und schnellstes Modell.
Der Report enthaelt zusaetzlich eine kurze Summary mit Top-1-Modell, eine eingebettete Plot-Vorschau und separate Abschnitte pro Datensatz.

Direkter Vergleich zweier aggregierter Benchmark-Laeufe:

```bash
python examples/benchmark_compare.py \
  --core-aggregated-json benchmarks/2026-03-core/benchmark_results_aggregated.json \
  --compare-aggregated-json benchmarks/2026-03-hs-compare/benchmark_results_aggregated.json \
  --output-markdown benchmarks/2026-03-core-vs-hs/benchmark_comparison_report.md \
  --output-html benchmarks/2026-03-core-vs-hs/benchmark_comparison_report.html
```

Kombinierter Meta-Report ueber beide Laeufe und ihren Vergleich:

```bash
python examples/benchmark_meta_report.py \
  --core-aggregated-json benchmarks/2026-03-core/benchmark_results_aggregated.json \
  --compare-aggregated-json benchmarks/2026-03-hs-compare/benchmark_results_aggregated.json \
  --output-markdown benchmarks/2026-03-meta/benchmark_meta_report.md \
  --output-html benchmarks/2026-03-meta/benchmark_meta_report.html
```

## JSON-Format (Kurz)

```json
{
  "format": "scoredrulesets",
  "version": "0.1",
  "class_labels": ["A", "B", "C"],
  "feature_names": ["f0", "f1"],
  "aggregation": {"type": "argmax_sum"},
  "rules": [
    {
      "atoms": [{"feature": "f0", "op": "<=", "value": 1.5}],
      "scores": [1.0, 0.0, 0.0]
    }
  ]
}
```

## Hinweis zum HS-Backend

`backend="hs"` ist als Default gesetzt. Wenn `imodels` nicht installiert ist oder keine passende HS-Klasse gefunden wird, gibt das Paket eine klare Fehlermeldung und nennt die Installationsoption.

Optionaler Integrationstest fuer HS:

```bash
pytest -q -m hs
```

Sklearn-Kompatibilitaet (Estimator-Checks) pruefen:

```bash
pytest -q tests/test_estimator_checks.py
```

