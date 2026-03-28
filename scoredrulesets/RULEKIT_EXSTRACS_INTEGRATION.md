# RuleKit und ExSTraCS Integration

## Übersicht

RuleKit und ExSTraCS wurden als neue Backends integriert. Beide Algorithmen lassen sich direkt zu Scored Rule Sets transformieren, wie im Paper beschrieben.

## Backends

### 1. RuleKit
- **Typ:** Regel-Listen-Lerner (Voting Scheme)
- **Sprache:** Java-basiert
- **Anforderung:** Java JDK 11+ erforderlich
- **Paket:** `rulekit`
- **Transformation:** Jede Regel hat Conditions + Zielklasse → Score Vektor

#### Installation
```bash
pip install rulekit
# Benötigt Java (JDK 11+)
java -version
```

#### Verwendung
```python
clf = ScoredRuleSetClassifier(backend="rulekit")
clf.fit(X_train, y_train)
```

### 2. ExSTraCS
- **Typ:** Genetischer Algorithmus für Regel-Population (LCS-Familie)
- **Sprache:** Python
- **Paket:** `scikit-exstracs` (skExSTraCS)
- **Transformation:** Regeln mit Fitness/Numerosity + Intervals → Score Vektor

#### Installation
```bash
pip install scikit-exstracs
```

#### Verwendung
```python
clf = ScoredRuleSetClassifier(backend="exstracs")
clf.fit(X_train, y_train)
```

## Transformationen

### RuleKit → Scored Rule Set

RuleKit produziert eine Regel-Liste mit:
- **Conditions:** Atome (Feature OP Value)
- **Class Label:** Zielklasse für diese Regel
- **Voting:** Jede passende Regel gibt 1 Stimme ab

**Transformation:**
```
RuleKit Regel:
  conditions: [x1 <= 5.0, x2 > 3.0]
  class: setosa

→ Scored Rule Set:
  atoms: [Atom(x1, <=, 5.0), Atom(x2, >, 3.0)]
  scores: [1.0, 0.0, 0.0]  # setosa = Klasse 0
```

### ExSTraCS → Scored Rule Set

ExSTraCS produziert eine Population mit:
- **Conditions:** Intervals [lower, upper] pro Feature
- **Class Label:** Zielklasse
- **Fitness × Numerosity:** Regelstärke

**Transformation:**
```
ExSTraCS Regel:
  conditions: [# (don't care), # (don't care), [5.2,6.9], [1.9,2.5]]
  phenotype: virginica
  fitness: 0.85
  numerosity: 12

→ Scored Rule Set:
  atoms: [Atom(petal_length, in, [5.2,6.9]), 
          Atom(petal_width, in, [1.9,2.5])]
  scores: [0.0, 0.0, 10.2]  # virginica = Klasse 2, score = 0.85*12
```

## Fehlerbehandlung

### Java nicht gefunden (RuleKit)
```
ImportError: backend='rulekit' benötigt Java, aber kein Java gefunden oder funktionsfähig.
Bitte installiere Java (JDK 11+).
```

**Lösung:**
```bash
# macOS (Homebrew)
brew install openjdk@11

# Linux (Ubuntu/Debian)
sudo apt-get install openjdk-11-jdk

# Windows
# Download von https://www.oracle.com/java/technologies/javase-downloads.html
```

### skExSTraCS nicht installiert
```
ImportError: backend='exstracs' braucht das 'skexstracs' Paket.
Installiere mit: pip install scikit-exstracs.
```

**Lösung:**
```bash
pip install scikit-exstracs
```

## Benchmark-Integration

Die neuen Backends sind sofort in Benchmarks verfügbar:

```bash
python3 examples/benchmarks/benchmark_full_report.py \
  --estimators wrapper_cart,wrapper_hs,wrapper_rulekit,wrapper_exstracs,native,gp \
  --datasets sklearn_iris,sklearn_wine,sklearn_breast_cancer \
  --repeats 3
```

## Vergleich der Backends

| Backend | Typ | Model | F1 | Rules | Atoms | Training |
|---------|-----|-------|-----|-------|-------|----------|
| CART | Tree | klein | gut | gering | viele | schnell |
| HS | Tree (optimiert) | sehr klein | gut | gering | wenige | schnell |
| RuleKit | Rule List | kompakt | gut | mittel | mittel | mittel |
| ExSTraCS | GA/LCS | mittel | sehr gut | hoch | hoch | langsam |
| logicGP | GA | sehr kompakt | sehr gut | mittel | wenige | langsam |

## Bekannte Besonderheiten

### RuleKit
- Benötigt Java - wird bei Import validiert
- Produziert interpretierbare Regel-Listen
- Voting-Schema (jede Regel: +1 Stimme für Zielklasse)
- Kann für multi-class Probleme optimiert werden

### ExSTraCS
- Genetischer Algorithmus - variabel in Anzahl Generationen
- Produziert Regel-Population (nicht sortiert wie RuleKit)
- Fitness/Numerosity für Regelstärke
- Braucht mehr Training für gute Ergebnisse

## Technische Details

### Datei: `ruleset_transform.py`

Neue Transformationsfunktionen:
- `rulekit_to_scored_ruleset()` - RuleKit → Scored Rule Set
- `exstracs_to_scored_ruleset()` - ExSTraCS → Scored Rule Set
- `_condition_to_atom()` - RuleKit Condition → Atom
- `_interval_to_atom()` - ExSTraCS Interval → Atom

### Datei: `backends.py`

Neue Backend-Resolver:
- `_resolve_rulekit_class()` - Mit Java-Validierung
- `_resolve_exstracs_class()` - Mit aussagekräftiger Fehlerbehandlung

### Datei: `sklearn_wrapper.py`

Aktualisiert:
- `fit()` - Wählt passende Transformation basierend auf Backend

## Performance-Erwartungen

### RuleKit
- **Trainingszeit:** Mittel (Java Overhead)
- **Modellgröße:** Mittel (kompakte Regel-Listen)
- **Vorhersage-Qualität:** Gut (interpretierbare Regeln)

### ExSTraCS
- **Trainingszeit:** Lange (GA)
- **Modellgröße:** Groß (Population)
- **Vorhersage-Qualität:** Sehr gut (evolvierte Regeln)

## Beispiele

### RuleKit verwenden
```python
from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = ScoredRuleSetClassifier(
    backend="rulekit",
    backend_params={},
)
clf.fit(X[:100], y[:100])
print(clf.to_ruleset())
```

### ExSTraCS verwenden
```python
clf = ScoredRuleSetClassifier(
    backend="exstracs",
    backend_params={
        # skExSTraCS Parameter
    },
)
clf.fit(X[:100], y[:100])
print(clf.to_ruleset())
```

### Alle Backends vergleichen
```python
backends = ["cart", "hs", "rulekit", "exstracs"]
for backend in backends:
    try:
        clf = ScoredRuleSetClassifier(backend=backend)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        ruleset = clf.to_ruleset()
        print(f"{backend:12} F1={score:.4f} Rules={len(ruleset.rules)}")
    except Exception as e:
        print(f"{backend:12} ERROR: {e}")
```

## Status

✅ **RuleKit-Integration:** Implementiert mit Java-Validierung
✅ **ExSTraCS-Integration:** Implementiert mit Fehlerbehandlung
✅ **Transformationen:** Beide implementiert
✅ **Benchmarks:** Beide verfügbar
✅ **Fehlerbehandlung:** Aussagekräftige Fehlermeldungen

