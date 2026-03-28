# Übersicht: Integrierte Schätzer in `scoredrulesets`

## 1. Schätzer-Architektur

Das Paket hat eine **zweischichtige Architektur**:
- **`ScoredRuleSetClassifier`** (sklearn_wrapper.py) – generischer Wrapper, der über `backend=` verschiedene Backends ansteuert und deren Ergebnis in ein `ScoredRuleSet` transformiert
- **Eigenständige Schätzer** – implementieren `BaseRuleSetEstimator` direkt und erzeugen intern ein `ScoredRuleSet`
- **`AutoScoredRuleSetClassifier`** – Meta-Schätzer, der per CV das beste Backend auswählt

---

## 2. Alle Backends / Schätzer im Detail

### A. **CART** (Backend `"cart"`)
- **Implementierung:** `sklearn.tree.DecisionTreeClassifier` (extern)
- **Transformation:** `tree_transform.py` → extrahiert Regeln aus `tree_` mit Tiefengewichtung (`depth_decay_lambda`)
- **Varianten/Optionen:**
  - `depth_decay_lambda` (Gewichtungsabfall mit Tiefe)
  - `include_default_rule`, `aggressive_prune`, `prune_atoms` (Atom-Pruning mit Äquivalenzvalidierung)
  - `prune_lambda_candidates` (Auto-Search über mehrere Lambda-Werte)
- **Stärken:** Schnell, robust, gut verstanden

### B. **HS** (Backend `"hs"`)
- **Implementierung:** `imodels.HSTreeClassifier` / `HSTreeClassifierCV` (extern)
- **Transformation:** Identisch zu CART (via `tree_transform.py`, gleicher Pfad)
- **Varianten:** Gleiche Transform-Params wie CART
- **Stärken:** Hierarchisches Shrinkage → oft bessere Interpretierbarkeit

### C. **RuleKit** (Backend `"rulekit"`)
- **Implementierung:** `rulekit.classification.RuleClassifier` (extern, Java-basiert via JPype)
- **Transformation:** `ruleset_transform.py` → `rulekit_to_scored_ruleset()` – parst Java-Rule-Objekte
- **Varianten:** Keine internen Varianten, nur RuleKit-eigene Parameter
- **Abhängigkeiten:** Java (JDK 11+), JPype, `rulekit`-Paket
- **⚠ Problematik:** Schwere externe Abhängigkeit (JVM), komplexer Setup-Code (`_ensure_java_home`, JVM-Architekturdetektion)

### D. **ExSTraCS** (Backend `"exstracs"`)
- **Implementierung:** `skExSTraCS.ExSTraCS` (extern)
- **Transformation:** `ruleset_transform.py` → `exstracs_to_scored_ruleset()`
- **Shrinking/Pruning:** `exstracs_shrinking.py` mit umfangreichen Optionen:
  - Conservative Pruning (ohne Verschlechterung)
  - Aggressive Pruning (mit max. F1-Verlust)
  - Rule Filtering (schwache Regeln entfernen)
  - Rule Consolidation (ähnliche Regeln mergen)
  - Interval-Merge (überlappende Intervalle zusammenführen)
- **⚠ Problematik:** Sehr komplexe Transformation (608 Zeilen `ruleset_transform.py` + 760 Zeilen `exstracs_shrinking.py`); Populationsformat schwer zu parsen

### E. **LogicGP** (Backend `"logicgp"`)
- **Implementierung:** Eigene Implementierung in `logicgp.py` (1321 Zeilen!)
- **Klasse:** `LogicGPClassifier`
- **Varianten (4 Trainer):**
  - `"flcw_macro"` – Full Literals Computed Weights, Macro-F1 (Standard)
  - `"flcw_micro"` – Full Literals Computed Weights, Micro-F1
  - `"rlcw_macro"` – Restricted Literals Computed Weights, Macro-F1
  - `"rlcw_micro"` – Restricted Literals Computed Weights, Micro-F1
- **Besonderheiten:**
  - Eigene Feature-Diskretisierung (KBinsDiscretizer)
  - Pareto-basierte Selektion
  - Class-bound Dominanz (RLCW)
  - Populationsgrößenbegrenzung mit Turnierselektion
  - 5 Mutationsoperatoren + Crossover
  - Klassen-diskriminatives Seeding
  - Elitismus-Tracking

### F. **Pittsburgh** (Backend `"pittsburgh"`)
- **Implementierung:** Eigene Implementierung in `pittsburgh.py` (547 Zeilen)
- **Klasse:** `PittsburghRuleSetClassifier`
- **Ansatz:** Beam-Search über Regel-Teilmengen
- **Atom-Typen:** `<=`, `>`, `between`, `==`, `in` (numerisch + kategorisch)
- **Besonderheiten:**
  - Kandidaten-Pool → Beam-Search mit Nachbarschaftssuche
  - Train/Val-Split für Selektion
  - Complexity-Penalty, Coverage-Bonus
  - `max_thresholds_per_feature` unterstützt

### G. **GeneticScoredRuleSetClassifier** (Backend `"gp"`)
- **Implementierung:** Eigene Implementierung in `gp_native.py` (1025 Zeilen)
- **Klasse:** `GeneticScoredRuleSetClassifier`
- **Varianten (per Parameter):**
  - `score_mode`: `"auto"`, `"log_proba"`, `"proba"`
  - `selection_mode`: `"fitness"`, `"pareto"`
  - `final_rule_selection`: `"fitness"`, `"diverse"`, `"contribution"`
  - `evolution_fitness_mode`: `"single_rule"`, `"residual_covering"`
  - `aggregation`: `"softmax_sum"` (Standard) oder `"argmax_sum"`
- **Besonderheiten:**
  - GA mit Turnierselektion, Crossover, Mutation
  - Hall-of-Fame, Early Stopping
  - Context-Gene-basierte Residual-Fitness
  - Diversity-Penalty, Class-Balance-Weight
  - `max_thresholds_per_feature` unterstützt

### H. **RuleFit** (Backend `"rulefit"`)
- **Implementierung:** Wrapper um `rulefit.RuleFit` (extern) in `rulefit.py` (291 Zeilen)
- **Klasse:** `RuleFitClassifier`
- **Ansatz:** Regeln aus Bäumen + Lasso-Regression (Friedman & Popescu, 2008)
- **Varianten:** Binär vs. Multi-Klasse (One-vs-Rest automatisch)
- **Transformation:** Nur "rule"-Typ-Regeln werden zu Atomen, "linear"-Terms werden ignoriert → **verlustbehaftete Transformation**

### I. **NLN – Neural Logic Network** (Backend `"nln"`)
- **Implementierung:** Eigene Implementierung in `nln.py` (563 Zeilen)
- **Klasse:** `NeuralLogicNetClassifier`
- **Ansatz:** Differenzierbares Logik-Netzwerk (nur numpy, kein PyTorch)
  - Conjunction Layer (lernbare Propositionsauswahl mit Sigmoid + L1)
  - Score Layer (per-class weights)
  - Adam-Optimizer, Mini-Batch-SGD
  - Pruning der nahe-Null-Gewichte → crisp Regeln
- **Besonderheiten:** `n_rules`, `n_bins`, `l1_conj`, `l1_score`, `atom_threshold`, `max_thresholds_per_feature`

### J. **NativeScoredRuleSetClassifier** (kein Backend-Key)
- **Implementierung:** `native.py` (384 Zeilen)
- **Klasse:** `NativeScoredRuleSetClassifier`
- **Ansatz:** Einfacher univariater Split-Learner (Gini-basiert)
- **Atom-Typen:** `<=`, `>`, `between`, `==`, `in`
- **Besonderheiten:** Feature-Gain-Ranking, adaptive Feature-Selektion, Gap-Detektion
- **⚠ Nicht als Backend im Wrapper registriert** – nur als eigenständiger Schätzer nutzbar

---

## 3. Zusammenfassungs- und Streichungskandidaten

### 🔴 Starke Streichungskandidaten

| Schätzer | Begründung |
|---|---|
| **NativeScoredRuleSetClassifier** | ✅ **GESTRICHEN** (Empfehlung 2 umgesetzt). Funktional eine Untermenge von CART. Datei `native.py` und alle Referenzen entfernt. |
| **ExSTraCS** (Backend) | ✅ **OPTIONALES PLUGIN** (Empfehlung 3 umgesetzt). Verfügbar über `pip install 'scoredrulesets[exstracs]'`. In Benchmarks weiterhin nutzbar. |
| **RuleKit** (Backend) | ✅ **OPTIONALES PLUGIN** (Empfehlung 3 umgesetzt). Verfügbar über `pip install 'scoredrulesets[rulekit]'`. In Benchmarks weiterhin nutzbar. |

### 🟡 Zusammenlegungskandidaten

| Varianten | Begründung |
|---|---|
| **GeneticScoredRuleSetClassifier** ↔ **PittsburghRuleSetClassifier** | Beide sind regelbasierte Suchverfahren über ähnlichen Kandidatenräumen (gleiche Atom-Typen: `<=`, `>`, `between`, `==`, `in`). Pittsburgh ist eine Beam-Search-Variante, GP-Native eine GA-Variante. Die Regelgenerierung (`_best_numeric_split`, `_categorical_splits`, `_numeric_interval_splits`, `_categorical_group_splits`, `_gini`) ist **nahezu identisch dupliziert** in beiden Dateien. → **Gemeinsame Basisklasse** für die Kandidatengenerierung extrahieren, dann GP und Beam-Search als zwei Suchstrategien. |
| **CART** ↔ **HS** | Teilen sich bereits die Transformation (`tree_transform.py`). Der einzige Unterschied ist der sklearn-Schätzer. **Keine weitere Zusammenlegung nötig, aber sauber.** |
| **LogicGP `flcw_macro`** ↔ **`flcw_micro`** | Unterscheiden sich nur im F1-Averaging (`"macro"` vs `"micro"`). **Könnten auf einen Trainer mit Parameter reduziert werden** (ist bereits so via `trainer`-String implementiert – also okay). |
| **LogicGP `rlcw_*`** ↔ **`flcw_*`** | RLCW ist eine eingeschränktere, effizientere Variante von FLCW. Beide teilen 80% des Codes. **Intern bereits gut vereint** über den `use_rlcw`-Flag in `_run_gp()` – hier ist das Design okay. |

### 🟢 Beibehaltungskandidaten (klar distinkt)

| Schätzer | Begründung |
|---|---|
| **CART** (+ HS) | Schnell, deterministisch, bewährt. Minimalfall. |
| **LogicGP** | Einzigartiger Ansatz (Polynom-/Monom-basierte Literale, Pareto-Selektion). Paper-Referenz-Implementierung. |
| **RuleFit** | Einzigartiger Ansatz (Baum-Regeln + Lasso). Leichte Abhängigkeit. |
| **NLN** | Einzigartiger Ansatz (differenzierbares Logik-Netzwerk, nur numpy). |
| **Auto** | Meta-Schätzer, sinnvolles Nutzer-Interface. |

### 🔧 Refactoring-Empfehlungen

1. ✅ **Duplizierten Code in GP-Native und Pittsburgh zusammengeführt:**
   - `_best_numeric_split`, `_categorical_splits`, `_numeric_interval_splits`, `_categorical_group_splits`, `_gini`, `_distribution_to_scores` → in gemeinsames Modul `_split_utils.py` extrahiert.
   - **~200 Zeilen** Code-Reduktion erreicht.

2. ✅ **Native Estimator gestrichen:** +384 Zeilen Reduktion, da funktional von CART abgedeckt.

3. ✅ **ExSTraCS + RuleKit als optionale Plugins:**
   - `pip install 'scoredrulesets[exstracs]'` → scikit-ExSTraCS
   - `pip install 'scoredrulesets[rulekit]'` → rulekit + jpype1
   - `pip install 'scoredrulesets[all]'` → alle optionalen Backends
   - Lazy-Imports in `sklearn_wrapper.py`, klare Fehlermeldungen in `backends.py`
   - Benchmarks funktionieren weiterhin (Extras müssen installiert sein)
   - Pytest-Marker `exstracs` und `rulekit` für Integrationstests registriert

4. **Ergebnis:** Core-Package reduziert sich auf **5 klar distinkte Schätzer** (CART/HS, LogicGP, RuleFit, NLN, Pittsburgh/GP) + den Auto-Meta-Schätzer. ExSTraCS und RuleKit sind optional verfügbar.
