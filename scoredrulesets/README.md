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
from scoredrulesets import ScoredRuleSetClassifier, format_ruleset_table
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = ScoredRuleSetClassifier(backend="cart")
clf.fit(X, y)
ruleset = clf.to_ruleset()

print(format_ruleset_table(ruleset))
```

## Examples

See `examples/README.md` for runnable demos:

- direct estimator demo: `examples/estimators/example_pittsburgh_backend.py`
- wrapper backend demo: `examples/estimators/example_pittsburgh_wrapper.py`

## logicGP-Import

```python
from scoredrulesets import import_logicgp_json, dump_ruleset_json

ruleset = import_logicgp_json("ext/logicgp_model.json")
dump_ruleset_json(ruleset, "converted_ruleset.json")
```

## Pittsburgh-Estimator

```python
from sklearn.datasets import load_iris
from scoredrulesets import PittsburghRuleSetClassifier

X, y = load_iris(return_X_y=True)

clf = PittsburghRuleSetClassifier(max_rules=5, random_state=42)
clf.fit(X, y)
print(clf.predict(X[:3]))

# Optional: Beam-Search-Parameter steuern
# PittsburghRuleSetClassifier(max_rules=6, beam_width=10, max_iterations=20)
```

Der Pittsburgh-Lerner erzeugt je nach Datenlage Atome mit `<=`, `>`, `between`, `==` und `in`.

## Benchmarking

Die stabilen Entrypoints liegen unter `examples/benchmarks/` und sind ueber
das Makefile gekapselt.

Schnellstart:

```bash
make benchmark
make benchmark-standard
make benchmark-normal-lite
```

Reports aus vorhandenen Ergebnissen regenerieren:

```bash
make reports-standard
make reports-normal-lite
make reports-full
```

Direkte Python-Aufrufe sind ebenfalls moeglich:

```bash
python3 examples/benchmarks/benchmark_full_report.py
python3 examples/benchmarks/benchmark_standard.py
python3 examples/benchmarks/benchmark_normal_lite.py
python3 examples/benchmarks/generate_reports.py normal-lite
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

