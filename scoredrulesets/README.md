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
    random_state=0,
)
gp.fit(X, y)
print(gp.predict(X[:3]))
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

