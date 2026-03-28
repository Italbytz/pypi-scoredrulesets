# Benchmark-Analyse: 11 Schätzer × 10 Datasets × 3 Repeats

## Globale Übersicht (sortiert nach Ø F1)

| Schätzer | Ø F1 | Ø Atoms | Ø Rules | Ø Fit (s) | Ø Rang |
|---|---:|---:|---:|---:|---:|
| **wrapper_rulekit_native** | **0.822** | 61.4 | 17.5 | 3.62 | 3.20 |
| **wrapper_rulekit** | **0.815** | 102.7 | 17.1 | 0.44 | 3.30 |
| **wrapper_hs** | **0.786** | 92.6 | 18.4 | 0.01 | **3.00** |
| wrapper_cart | 0.715 | 47.3 | 12.5 | 0.003 | 5.20 |
| wrapper_nln_strong | 0.706 | 59.3 | 20.7 | 1.22 | 5.90 |
| wrapper_logicgp | 0.683 | 4.6 | 3.9 | 2.27 | 5.30 |
| wrapper_rulegp | 0.661 | 6.1 | 4.5 | 2.00 | 6.50 |
| wrapper_pittsburgh | 0.592 | 1.7 | 2.7 | 0.57 | 7.80 |
| nln_native | 0.589 | 41.3 | 13.0 | 0.40 | 7.50 |
| wrapper_nln | 0.589 | 41.3 | 13.0 | 0.41 | 8.50 |
| wrapper_nln_fast | 0.498 | 26.1 | 9.0 | 0.10 | 9.80 |

---

## 1. Kandidaten zum Entfernen (paretodominiert / redundant)

### 🔴 `wrapper_nln_fast` — **100% paretodominiert, klarer Entfernungskandidat**
- Wird auf **allen 10 Datasets** von mindestens einem anderen Schätzer dominiert (bessere F1 bei gleicher oder kleinerer Modellgröße).
- Letzter Platz im Ø-Rang (9.80), schlechteste Ø-F1 (0.498).
- Ist nicht einmal schneller als `nln_native`/`wrapper_nln` genug, um den massiven F1-Verlust zu rechtfertigen.
- **Empfehlung: Entfernen.**

### 🔴 `wrapper_nln` — **90% paretodominiert, identisch mit `nln_native`**
- Auf **allen 10 Datasets exakt identische F1 und Atoms** wie `nln_native`.
- Marginale Laufzeit-Differenz (0.41s vs 0.40s) – kein Informationsgewinn.
- **Empfehlung: Entfernen (reines Duplikat von `nln_native`).**

### 🟡 `nln_native` — **90% paretodominiert, aber einziger NLN-Basisvertreter**
- 90% dominiert, Ø-Rang 7.5, Ø-F1 nur 0.589.
- Wird fast überall von `wrapper_nln_strong` oder Tree-Methoden geschlagen.
- Wenn `wrapper_nln_strong` bleibt, ist `nln_native` redundant.
- **Option A: Entfernen (nur `wrapper_nln_strong` behalten).** Option B: Behalten als schnelle NLN-Baseline.

### 🟡 `wrapper_nln_strong` — **80% paretodominiert, aber einzige NLN-Variante mit Potenzial**
- Ø-F1 0.706 ist respektabel, aber bei 59.3 Atoms recht groß.
- Glänzt auf einzelnen Datasets (mux_11: 0.904, synth_xor: 0.924, synth_dnf: 0.999).
- Ist aber auf 80% der Datasets von kompakteren Schätzern dominiert.
- **Option A: Behalten als einziger NLN-Vertreter.** Option B: Entfernen falls NLN-Familie insgesamt nicht benötigt.

### 🟡 `wrapper_hs` — **80% paretodominiert, aber bester Ø-Rang**
- Trotz 80% Dominanz hat `wrapper_hs` den **besten Ø-Rang (3.0)** – d.h. HS ist selten der schlechteste, aber wegen ~93 Atoms fast nie Pareto-optimal.
- Sehr schnell (0.01s), konsistent solide.
- Problem: Große Modelle (Ø 92.6 Atoms), damit fast immer von kompakteren Schätzern dominiert.
- **Empfehlung: Behalten – konsistenter, schneller Allrounder.**

### ℹ️ `wrapper_cart` — **70% paretodominiert**
- Solide Basis (0.715 F1, 47.3 Atoms, 0.003s), aber auf den meisten Datasets von HS oder RuleKit übertroffen bei vergleichbarer Modellgröße.
- **Empfehlung: Behalten als ultraschnelle Baseline.**

---

## 2. Verbesserungspotenzial

### 🔧 `wrapper_pittsburgh` — **Nie paretodominiert (0%!), aber schwache F1**
Das ist der spannendste Befund: Pittsburgh wird auf **keinem einzigen Dataset paretodominiert** – weil es die **kompaktesten Modelle** erzeugt (Ø 1.7 Atoms, Ø 2.7 Rules). In der Pareto-Front ist es damit immer vertreten. Aber die Ø-F1 von 0.592 und der Ø-Rang 7.8 zeigen deutlich: die Modelle sind zu einfach.

**Verbesserungspotenzial:**
- Das neue OvR-Feature könnte helfen (mehr klassenspezifische Regeln).
- Mehr `max_rules` und `candidate_pool_size` (aktuell 7/36) könnten die F1 steigern, ohne die Pareto-Eigenschaft zu verlieren.
- Eine zweite Pittsburgh-Variante mit aggressiveren Parametern wäre sinnvoll (`max_rules=12`, `candidate_pool_size=48`).
- Besonders schwach auf uci_car_evaluation (0.206, nur Default-Regel!) und synth_monk3 (0.398) – hier finden die Kandidaten offenbar keine guten Splits.

### 🔧 `wrapper_logicgp` — **Nur 20% dominiert, sehr kompakt, aber mäßige F1**
- Ø 4.6 Atoms, Ø 3.9 Rules – zweikompaktester nach Pittsburgh.
- Ø-F1 0.683 ist ordentlich für die Modellgröße.
- Aber 2.27s Trainingszeit – deutlich langsamer als Alternativen.
- Schwach bei car_evaluation (0.391) und epistasis (0.672).
- **Potenzial: Population/Generationen erhöhen könnte F1 steigern.**

### 🔧 `wrapper_rulegp` — **50% dominiert, aber kompakt**
- Ø 6.1 Atoms bei Ø-F1 0.661 – akzeptable Kompaktheit.
- Aber 2.0s Trainingszeit bei mäßiger F1.
- **Potenzial begrenzt – ähnlich wie logicgp, aber mit weniger konsistenten Ergebnissen.**

### 🔧 `wrapper_rulekit_native` vs. `wrapper_rulekit`
Beide sind Top-3-Schätzer, ergänzen sich aber:

| | rulekit (Java) | rulekit_native |
|---|---|---|
| Ø F1 | 0.815 | **0.822** |
| Ø Atoms | 102.7 | **61.4** (40% weniger) |
| Ø Fit | **0.44s** | 3.62s (8× langsamer) |
| Bester Rang | 1 | 1 |

- `rulekit_native` hat bessere F1 *und* kleinere Modelle, ist aber deutlich langsamer.
- Problem von `rulekit_native`: extrem langsam auf manchen Datasets (checkerboard: 10.8s, overlap: 13.0s, highdim: 11.3s).
- **Empfehlung: Beide behalten – komplementäre Stärken.**

---

## 3. Gibt es klar dominante Schätzer?

### Ja, es gibt eine klare **Top-3-Gruppe**:

| Rang | Schätzer | Stärke |
|---:|---|---|
| 1 | **wrapper_hs** | Bester Ø-Rang (3.0), ultraschnell, konsistent |
| 2 | **wrapper_rulekit_native** | Beste Ø-F1 (0.822), kompakter als rulekit |
| 3 | **wrapper_rulekit** | Zweitbeste Ø-F1 (0.815), deutlich schneller als native |

Diese drei sind auf fast allen Datasets unter den Top-4. `wrapper_hs` besticht durch Konsistenz und Geschwindigkeit, die RuleKit-Varianten durch Top-F1.

### Mittelfeld mit speziellen Stärken:
- **wrapper_cart** (schnellste Baseline, robust)
- **wrapper_nln_strong** (Spitze auf logischen Problemen: mux, xor, dnf)
- **wrapper_logicgp** (sehr kompakte Modelle)

### Untere Hälfte:
- **wrapper_pittsburgh** (kompakteste Modelle, aber schwache F1)
- **nln_native / wrapper_nln** (redundant, mittelmäßig)
- **wrapper_nln_fast** (durchgängig schlecht)

---

## Zusammenfassung der Entscheidungen

| Schätzer | Empfehlung | Begründung |
|---|---|---|
| `wrapper_nln_fast` | 🔴 **Entfernen** | 100% dominiert, schlechteste F1 |
| `wrapper_nln` | 🔴 **Entfernen** | Exaktes Duplikat von `nln_native` |
| `nln_native` | 🟡 **Entfernen oder behalten** | 90% dominiert; redundant wenn `nln_strong` bleibt |
| `wrapper_nln_strong` | 🟢 Behalten | Einziger relevanter NLN-Vertreter |
| `wrapper_pittsburgh` | 🟢 Behalten + **tunen** | Nie dominiert, F1 muss hoch |
| `wrapper_logicgp` | 🟢 Behalten | Kompakt, nur 20% dominiert |
| `wrapper_rulegp` | 🟡 **Optional entfernen** | 50% dominiert, ähnliches Profil wie logicgp aber schwächer |
| `wrapper_cart` | 🟢 Behalten | Schnelle Baseline |
| `wrapper_hs` | 🟢 Behalten | Top-3, konsistent |
| `wrapper_rulekit` | 🟢 Behalten | Top-3, schnell |
| `wrapper_rulekit_native` | 🟢 Behalten | Top-3, beste F1 |
