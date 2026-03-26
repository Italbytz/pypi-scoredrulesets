# Benchmark-Analyse: Standard-Benchmark (10 Datasets, 31 Schätzer)

## 1. Entfernungskandidaten (Pareto-dominiert / redundant)

### 🔴 Exakte Duplikate – sofort entfernen
| Redundant | Identisch mit | Grund |
|---|---|---|
| **`gp_diverse`** | `gp` | Exakt gleiche F1 auf allen 10 Datasets |
| **`wrapper_nln`** | `nln_native` | Exakt gleiche F1 auf allen 10 Datasets |
| **`wrapper_rulekit_native`** | `rulekit_native` | Exakt gleiche F1 auf allen 10 Datasets |
| **`wrapper_cart`** (unpruned) | `wrapper_cart_pruned` | Gleiche F1, aber mehr Atome (47 vs. 25) |

**Empfehlung: 4 Schätzer sofort streichen.** Cart unpruned durch cart_pruned ersetzen (gleiche F1, weniger Atome).

### 🟠 Durchgehend schlecht (100% Pareto-dominiert)
Diese Schätzer werden auf **allen 10 Datasets** von mindestens einem anderen Schätzer in F1 UND Modellgröße geschlagen:

| Schätzer | Ø F1 | Ø Atome | Problem |
|---|---|---|---|
| **`gp`** | 0.456 | 10.2 | Schlechteste GP-Variante, dominiert von gp_residual |
| **`gp_fast`** | 0.458 | 20.0 | Viele Atome, kaum F1 |
| **`nln_native`** | 0.454 | 38.6 | NLN generell sehr schwach |
| **`wrapper_nln_fast`** | 0.398 | 43.0 | Schlechteste NLN-Variante |
| **`wrapper_nln_strong`** | 0.452 | 63.8 | Viele Atome, kaum F1 |
| **`wrapper_hs`** (unpruned) | 0.784 | 92.9 | `wrapper_hs_pruned` hat gleiche F1, 1/3 Atome |
| **`wrapper_rulefit`** (full) | 0.743 | 64.3 | Hohe Atomzahl, von HS/CART/RuleKit dominiert |

**Empfehlung: Alle NLN-Varianten (`nln_native`, `wrapper_nln`, `wrapper_nln_fast`, `wrapper_nln_strong`) entfernen – NLN erreicht auf keinem Dataset kompetitive Ergebnisse (Ø F1=0.45, max 0.62). `gp` und `gp_fast` entfernen (gp_residual/gp_contrib sind besser). `wrapper_hs` unpruned durch pruned ersetzen.**

### 🟡 Innerhalb der Familie dominiert
| Familie | Entfernungskandidat | Begründung |
|---|---|---|
| **Pittsburgh** | `wrapper_pittsburgh_fast` | Marginal schlechter als `wrapper_pittsburgh` (0.559 vs. 0.570), kaum schneller. Alle 4 Pittsburgh-Varianten liegen nah beieinander. |
| **Pittsburgh** | `wrapper_pittsburgh` (Standard) | `diverse`/`strong` sind minimal besser (0.592/0.593 vs. 0.570), aber Pittsburgh generell schwach |
| **RuleKit Native** | `wrapper_rulekit_native_fast` | Deutlich weniger F1 (0.705 vs. 0.802), nur wenig weniger Atome |
| **ExSTraCS** | `wrapper_exstracs_compact` | F1 schlechter als pruned (0.805 vs. 0.879), Laufzeit fast gleich (138s vs. 151s), und **Validierungswarnungen** (Shrinking zerstört F1 auf 3 Datasets) |
| **RuleFit** | `wrapper_rulefit_compact` | Massiver F1-Verlust (0.604 vs. 0.743) bei mäßiger Atom-Reduktion |

## 2. Verbesserungspotential

### 🔧 GP-Familie (gp, gp_residual, gp_contrib)
- **gp_residual** ist die einzig brauchbare GP-Variante (Ø F1=0.611), aber immer noch deutlich unter CART/HS/RuleKit
- Auf `synth_dnf_3x2` erreicht `gp_residual` nur 0.849 wo andere 1.0 schaffen
- **Empfehlung:** GP-Varianten auf `gp_residual` und `gp_contrib` reduzieren, Laufzeit-Budget erhöhen

### 🔧 Pittsburgh LCS
- Alle Pittsburgh-Varianten scheitern auf `synth_monk3` (F1=0.398 – nur Default-Regel!) und `uci_car_evaluation` (F1=0.206)
- Auf `synth_highdim_p500_n120` überraschend gut (0.839 bei nur 3 Atomen!)
- **Empfehlung:** Problem bei Multiclass-Datasets untersuchen (monk3, car_eval). Pittsburgh funktioniert nur bei binären Problemen mit klaren Regeln.

### 🔧 LogicGP
- `wrapper_logicgp_flcw` hat **6× längere Laufzeit** (15s vs. 2.4s) als `wrapper_logicgp`/`wrapper_logicgp_rlcw_macro` bei leicht schlechterem F1
- `wrapper_logicgp_rlcw_macro` ist die effizienteste Variante (Ø F1=0.683, 4.6 Atome, 2.2s)
- **Empfehlung:** FLCW nur als Option behalten, RLCW_macro als Default

### 🔧 ExSTraCS Shrinking
- `wrapper_exstracs_pruned` hat den besten F1 der ExSTraCS-Familie (0.879), aber ~1690 Atome
- `wrapper_exstracs_compact` verliert zu viel F1 (3 Datasets mit >15% Verlust-Warnungen)
- **Empfehlung:** Nur `wrapper_exstracs` (Basis) und `wrapper_exstracs_pruned` behalten. Compact streichen.

### 🔧 RuleKit Transformation
- `wrapper_rulekit` (mit ScoredRuleSet-Transformation) hat höheren F1 als `rulekit_native` (0.815 vs. 0.803) – überraschend
- Aber auf `mux_11`: rulekit_native=1.0 vs. wrapper_rulekit=0.9375
- **Empfehlung:** RuleKit-Transformation prüfen – wo kommt der F1-Gewinn her?

## 3. Welche Schätzer sind deutlich besser?

### 🏆 Klare Top-Tier (über alle Datasets konsistent gut)

| Rang | Schätzer | Ø F1 | Ø Atome | Ø Fit | Stärke |
|---|---|---|---|---|---|
| 1 | **`wrapper_exstracs_pruned`** | 0.879 | 1690 | 151s | Bester F1, aber riesige Modelle & langsam |
| 2 | **`wrapper_exstracs`** | 0.859 | 2322 | 47s | Zweitbester F1, sehr große Modelle |
| 3 | **`rulekit_native`** | 0.803 | 56 | 3.3s | Sehr guter F1/Größe-Trade-off |
| 4 | **`wrapper_hs_pruned`** | 0.785 | 35 | 34s | Guter F1 bei kompaktem Modell |
| 5 | **`wrapper_rulekit`** | 0.815 | 103 | 0.7s | Schnell, guter F1 |

### 🎯 Bester Trade-off (F1 vs. Modellgröße)
- **`wrapper_logicgp_rlcw_macro`**: F1=0.683, nur 4.6 Atome, 2.2s – extrem kompakte Modelle
- **`wrapper_pittsburgh_diverse`**: F1=0.592, nur 1.7 Atome – allerdings auf Multiclass sehr schwach
- **`wrapper_cart_pruned`**: F1=0.715, 25 Atome – bewährter Baseline

### 📊 Zusammenfassung
1. **ExSTraCS** und **RuleKit** sind die klar stärksten Backends für F1, aber mit sehr unterschiedlichen Modellgrößen
2. **HS pruned** bietet den besten Kompromiss zwischen F1 und Kompaktheit bei den Wrapper-Schätzern
3. **LogicGP (RLCW)** liefert die kompaktesten interpretablen Modelle (2-5 Atome)
4. **NLN ist durchgehend unbrauchbar** und sollte entfernt werden
5. **Pittsburgh** funktioniert nur auf binären Problemen
6. Die **GP-Varianten** (ohne Pittsburgh/LogicGP) sind generell die schwächsten Schätzer

### Empfohlene Aktionen zur Entscheidung:

**Option A: Aggressives Streichen (14 Schätzer → 17 übrig)**
Entfernen: `gp`, `gp_diverse`, `gp_fast`, `nln_native`, `wrapper_nln`, `wrapper_nln_fast`, `wrapper_nln_strong`, `wrapper_cart` (unpruned), `wrapper_hs` (unpruned), `wrapper_rulekit_native` (Duplikat), `wrapper_rulekit_native_fast`, `wrapper_exstracs_compact`, `wrapper_rulefit_compact`, `wrapper_pittsburgh_fast`

**Option B: Moderates Streichen (8 exakte Duplikate + NLN)**
Entfernen: `gp_diverse`, `wrapper_nln`, `wrapper_rulekit_native`, `wrapper_cart` (unpruned), `nln_native`, `wrapper_nln_fast`, `wrapper_nln_strong`, `wrapper_hs` (unpruned)

**Option C: Nur exakte Duplikate entfernen (4 Schätzer)**
Entfernen: `gp_diverse`, `wrapper_nln`, `wrapper_rulekit_native`, `wrapper_cart` (unpruned = ersetzt durch pruned)
