# LogicGP Varianten-Analyse: Empfehlungen

**Basis:** 324 Läufe (12 Varianten × 9 Datensätze × 3 Repeats), alle erfolgreich abgeschlossen.

---

## 1. Varianten, die entfernt werden sollten (klar paretodominiert)

### ❌ Fast-Varianten (`lgp_rlcw_macro_fast`, `lgp_flcw_macro_fast`)
- **Budget RLCW:** Standard gewinnt **7 von 9** Datensätzen, Avg ΔF1 = +0.078
- **Budget FLCW:** Standard gewinnt **8 von 9** Datensätzen, Avg ΔF1 = +0.103
- Die Fast-Varianten sparen zwar ~1s Laufzeit, verlieren aber massiv an F1
- Besonders kritisch bei `synth_imbalanced_10pct`: F1 fällt von 0.88 auf 0.47
- **Empfehlung: Beide entfernen.** Falls ein Speed-Modus gewünscht ist, besser über einen Parameter `budget="fast"` lösen

### ❌ FLCW-Macro (`lgp_flcw_macro`)
- RLCW-Macro gewinnt **5 von 9** Datensätzen, Avg ΔF1 = +0.018
- Bei synth_xor_3bit besonders deutlich: RLCW 0.65 vs FLCW 0.54
- **Empfehlung: Entfernen als eigenständige Variante**, da RLCW konsistent besser ist

### ❌ `lgp_rlcw_macro` (als Macro zugunsten von Micro)
- RLCW-Micro gewinnt **3x**, Macro nur **1x**, bei 5 Ties
- Avg ΔF1 = -0.014 (Micro besser)
- Micro besonders stark bei Interaktionsdaten (xor_3bit: 0.77 vs 0.65)
- **Empfehlung: Macro-Averaging als Default durch Micro ersetzen**

---

## 2. Varianten, die zusammengefasst werden können (kein signifikanter Unterschied)

### 🔄 Literal-Filter (`min_max_weight` 0.0 / 0.1 / 0.25)
- `nofilt` vs `macro`: **9 von 9 Ties** (ΔF1 = 0.0000)
- `strongfilt` vs `macro`: **9 von 9 Ties** (ΔF1 = 0.0000)
- Der `min_max_weight`-Parameter hat **keinerlei messbaren Einfluss**
- **Empfehlung: Parameter als Konfigurationsoption behalten, aber alle 3 separaten Varianten streichen.** Default 0.1 beibehalten

### 🔄 Bins (3 / 5 / 7)
- 5 vs 3: **6 Ties**, ΔF1 = -0.001
- 5 vs 7: **8 Ties**, ΔF1 = +0.002
- Die Bin-Zahl hat nahezu keinen Einfluss – vermutlich weil die meisten Testdatensätze bereits kategorisch sind oder wenige kontinuierliche Features haben
- **Empfehlung: Parameter als Konfigurationsoption behalten, `n_bins=5` als Default, separate Varianten streichen**

### 🔄 Literal-Generator (Full vs Singleton)
- **4 Ties**, 3 Wins Full, 2 Wins Singleton, ΔF1 = -0.002
- Singleton besser bei epistasis (+0.11), Full besser bei car_evaluation (+0.08)
- **Empfehlung: Als Konfigurationsparameter behalten, `full` als Default, separate Variante streichen**

### 🔄 Populationsgröße (50 vs 80)
- **4 Ties**, 3:2 für pop=50, ΔF1 = +0.011
- Größere Population hilft nicht konsistent
- **Empfehlung: `population_size=50` als Default, Parameter behalten, Variante streichen**

---

## 3. Varianten, die behalten werden sollten

### ✅ `lgp_rlcw_micro` (RLCW + Micro-Averaging) → **Neuer Default**
- Konsistent gut über alle Datensätze
- Bester Trainer-Typ bei Interaktionsdaten
- Avg F1 über alle Datensätze: ~0.67

### ✅ `lgp_flcw_micro` (FLCW + Micro) → **Alternative behalten**
- 4:4 gegen RLCW-Micro → situationsabhängig
- Teils besser bei checkerboard, epistasis
- Bietet echte Diversität im Ansatz

---

## 4. Zusammenfassung: Von 12 auf 2-3 Varianten

| Behalten | Rolle | Begründung |
|----------|-------|------------|
| `lgp_rlcw_micro` | **Default** | Konsistent bester oder gleichwertiger F1 |
| `lgp_flcw_micro` | **Alternative** | Komplementär auf einigen Datensätzen |
| `lgp_rlcw_macro` | *Optional* | Nur falls Macro-F1 explizit gewünscht |

**9 Varianten können entfernt werden:**
- `lgp_rlcw_macro_fast` (Fast klar schlechter)
- `lgp_flcw_macro_fast` (Fast klar schlechter)
- `lgp_flcw_macro` (RLCW dominiert FLCW bei Macro)
- `lgp_flcw_micro` → behalten (situationsabhängig anders)
- `lgp_rlcw_macro_singleton` (kein Unterschied zu full)
- `lgp_rlcw_macro_bins3` (kein Unterschied zu bins5)
- `lgp_rlcw_macro_bins7` (kein Unterschied zu bins5)
- `lgp_rlcw_macro_nofilt` (kein Unterschied)
- `lgp_rlcw_macro_strongfilt` (kein Unterschied)
- `lgp_rlcw_macro_bigpop` (kein Unterschied)

---

## 5. Auffällige Beobachtungen & nächste Schritte

1. **F1-Werte insgesamt niedrig** (Ø ~0.67): LogicGP hat Schwächen bei einigen synthetischen Datensätzen (checkerboard: 0.52-0.58, car_evaluation: 0.37-0.39). Hier könnten längere Laufzeiten oder adaptive Parameteranpassung helfen.

2. **`min_max_weight` hat keinen Effekt**: Dies deutet darauf hin, dass der Literal-Filter-Mechanismus in der aktuellen Implementierung möglicherweise nicht korrekt greift oder die Testdatensätze keine Literale erzeugen, die gefiltert würden. **→ Implementierung prüfen.**

3. **Micro-Averaging gewinnt**: F1-Micro als Fitness-Ziel führt konsistent zu besseren F1-Macro-Werten auf dem Testset. Dies ist ein interessantes Ergebnis, das in einem Paper erwähnenswert wäre.

4. **Empfohlene Standard-Konfiguration für das Paket:**
   ```python
   ScoredRuleSetClassifier(
       backend="logicgp",
       backend_params={
           "trainer": "rlcw_micro",
           "max_generations": 200,
           "stagnation_generations": 40,
           "population_size": 50,
           "n_bins": 5,
           "literal_generator": "full",
       }
   )
   ```
