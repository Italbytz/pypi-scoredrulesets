# Bewertung: Reimplementierung externer Schätzer – Lohnt es sich?

## Kurze Antwort: **Nein** – für die geplanten Features ist eine Reimplementierung nicht nötig und sogar kontraproduktiv.

---

## Die aktuelle Architektur

Das Package hat bereits ein klares Zwei-Schichten-Modell:

```
┌─────────────────────────────────────────────────────────────────┐
│  ScoredRuleSetClassifier (sklearn_wrapper.py)                   │
│  ┌───────────┐    ┌──────────┐    ┌───────────────────────────┐ │
│  │ X, y      │───▶│ Backend  │───▶│ ScoredRuleSet             │ │
│  │ (Rohdaten)│    │ .fit()   │    │ (einheitl. Regelformat)   │ │
│  └───────────┘    └──────────┘    └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**6 von 10 Backends sind bereits nativ** (michigan, pittsburgh, logicgp, gp_native, native, nln, rulefit) – nur CART/HS (sklearn/imodels), RuleKit (Java) und ExSTraCS (skExSTraCS) sind echte externe Dependencies.

---

## Warum Reimplementierung NICHT nötig ist

### 1. Atom-Reduktion (Preprocessing) funktioniert ohne Reimplementierung

Die Atom-Reduktion hat **zwei Ebenen**, die beide Backend-agnostisch sind:

| Ebene | Wo eingreifen | Wirkung | Geht mit externen Backends? |
|-------|---------------|---------|----------------------------|
| **Feature-Selektion** | Vor `backend.fit(X, y)` – Spalten aus X entfernen | Weniger Features → weniger Atome | ✅ Ja – alle Backends bekommen nur die reduzierten Features |
| **Threshold-Budget** | Nur für native Backends – `_build_feature_specs()` einschränken | Weniger Schwellenwerte pro Feature | ⚠️ Nur native Backends |

Die Feature-Selektion (Ebene 1) reicht für alle Backends und ist die wirkungsvollste Reduktion. Der Code dafür existiert bereits in `utils_feature_selection.py`:

```python
# Bereits vorhanden – muss nur in ScoredRuleSetClassifier.fit() integriert werden:
X_reduced, selected_names = select_features(X, y, method="kbest", k=20)
self.estimator_.fit(X_reduced, y)
```

Das funktioniert mit ExSTraCS, RuleKit, CART – **identisch**, ohne deren Interna zu kennen.

### 2. AutoML Backend-Selektion funktioniert über die bestehende API

```python
class AutoScoredRuleSetClassifier(BaseRuleSetEstimator):
    def fit(self, X, y):
        candidates = ["cart", "hs", "rulekit", "exstracs", "gp", "pittsburgh"]
        best_score, best_clf = -1, None
        for backend in candidates:
            clf = ScoredRuleSetClassifier(backend=backend, ...)
            score = cross_val_score(clf, X, y, scoring="f1_macro").mean()
            if score > best_score:
                best_score, best_clf = score, clf
        best_clf.fit(X, y)
        self.ruleset_ = best_clf.to_ruleset()
```

Das funktioniert **exakt gleich** für native und externe Backends – weil die Schnittstelle `ScoredRuleSetClassifier(backend=...)` einheitlich ist.

### 3. Reimplementierung ExSTraCS wäre enormer Aufwand mit fraglichem Nutzen

ExSTraCS hat ~15.000 Zeilen mit:
- Population Management (N=1000, covering, subsumption, compaction)
- Accuracy-based Fitness mit Power-Parameter
- Attribute Tracking & Expert Knowledge Feedback
- GA mit Crossover + Mutation + Niche-basiertem Selection
- Correct-Set / Action-Set Partitionierung
- Rule Compaction (QRF, QRC, PDRC, CRA2)
- Interval-Repräsentation für kontinuierliche Features

Eine Reimplementierung würde **Monate** dauern und wäre immer ein Schritt hinter dem Original.

---

## Was STATTDESSEN sinnvoll ist

### Architektur-Erweiterung in 3 Schritten:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  AutoScoredRuleSetClassifier                    ← NEU: Schritt 3       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ScoredRuleSetClassifier                                        │   │
│  │  ┌──────────────┐  ┌──────────┐  ┌───────────────────────────┐ │   │
│  │  │ AtomBudget   │  │ Backend  │  │ ScoredRuleSet             │ │   │
│  │  │ (Preproc.)   │─▶│ .fit()   │─▶│                           │ │   │
│  │  │ ← Schritt 1  │  └──────────┘  └───────────────────────────┘ │   │
│  │  └──────────────┘       ↑                                       │   │
│  │                    Schritt 2:                                    │   │
│  │                    allowed_atoms für native Backends             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Schritt 1: Atom-Budget als Preprocessing** (alle Backends)
```python
ScoredRuleSetClassifier(
    backend="exstracs",
    preprocessing={"feature_selection": "kbest", "k": 20},  # NEU
)
```
→ Wirkt vor `fit()` auf X – alle Backends profitieren.

**Schritt 2: `allowed_atoms` für native Backends** (michigan, pittsburgh, logicgp, gp, nln)
```python
ScoredRuleSetClassifier(
    backend="gp",
    preprocessing={"max_thresholds_per_feature": 5},  # NEU
)
```
→ Schränkt `_build_feature_specs()` ein – nur für native Backends, aber dort am wirkungsvollsten.

**Schritt 3: AutoML Meta-Estimator**
```python
AutoScoredRuleSetClassifier(
    candidate_backends=["hs", "rulekit", "exstracs", "gp"],
    cv=5,
    scoring="f1_macro",
)
```
→ Nutzt die einheitliche API – keine Backend-spezifische Logik nötig.

---

## Zusammenfassung: Wann reimplementieren?

| Situation | Reimplementierung? | Stattdessen |
|-----------|-------------------|-------------|
| Atom-Reduktion für alle Backends | ❌ Nein | Feature-Selektion vor `fit()` |
| Atom-Budget für native Backends | ❌ Nein | `_build_feature_specs()` parametrisieren |
| AutoML Backend-Selektion | ❌ Nein | Meta-Estimator über einheitliche API |
| ExSTraCS-Dependency fällt weg | ⚠️ Dann ggf. | Michigan aufwerten ODER Fork |
| Neues LCS-Feature (z.B. Transfer Learning) | ⚠️ Dann ggf. | Nur wenn ExSTraCS es nicht unterstützt |
| Paper-Reproduzierbarkeit ohne Dependencies | ⚠️ Dann ggf. | Vendoring statt Reimplementierung |

**Die bestehende Architektur ist bereits richtig aufgestellt.** Die einheitliche Schnittstelle `backend → ScoredRuleSet` macht Cross-Cutting Concerns Backend-agnostisch implementierbar. ExSTraCS reimplementieren würde den Aufwand nicht rechtfertigen – Feature-Selektion vor `fit()` und ein AutoML-Meta-Estimator darüber lösen die beschriebenen Anforderungen eleganter.
