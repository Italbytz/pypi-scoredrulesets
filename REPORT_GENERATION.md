# Report-Generierung aus Benchmark-Ergebnissen

Nach einem fertigen Benchmark-Lauf können die Reports unabhängig und wiederholt erstellt werden, ohne die Benchmarks erneut zu laufen.

## Motivation

- **Neue Report-Layouts**: Änderung der Report-Formatierung nach einem Benchmark
- **Alternative Aggregationen**: Reports mit anderen statistischen Methoden (z.B. andere error bars)
- **Dokumentation updaten**: Reports neu generieren nach Dokumentations-Updates
- **Verschiedene Report-Formate**: Zusätzliche Visualisierungen oder Ableitungen

## Verwendung

### CLI-Modus (vordefinierte Benchmarks)

```bash
# Normal-Lite-Benchmark
python3 examples/benchmarks/generate_reports.py normal-lite

# Standard-Benchmark
python3 examples/benchmarks/generate_reports.py standard

# Full-Benchmark
python3 examples/benchmarks/generate_reports.py full
```

### Custom-Modus (beliebige Benchmark-Verzeichnisse)

```bash
python3 examples/benchmarks/generate_reports.py \
  --input-dir benchmarks/normal_lite \
  --output-dir benchmarks/normal_lite \
  --title "Custom Report Title" \
  --design "Custom design description"
```

### Via Makefile

```bash
# Regeneriere alle Reports
make reports-normal-lite
make reports-standard
make reports-full

# Oder nach einem vollständigen Benchmark-Lauf:
make benchmark-normal-lite
make reports-normal-lite  # Reports neu erstellen
```

## Eingabe-Format

Das Skript erwartet eine `benchmark_results.json` im Input-Verzeichnis mit folgendem Format:

```json
[
  {
    "dataset": "synth_dnf_3x2",
    "estimator": "wrapper_cart_pruned",
    "repeat": 0,
    "f1_macro": 0.95,
    "f1_micro": 0.94,
    "accuracy": 0.92,
    "recall_macro": 0.91,
    "precision_macro": 0.93,
    "training_time": 0.12,
    "prediction_time": 0.008,
    "n_rules": 5
  },
  ...
]
```

## Ausgabe-Dateien

Das Skript generiert folgende Reports und Visualisierungen:

### Textuelle Reports
- `benchmark_report.md` - Markdown-Report mit Leaderboard
- `benchmark_report.html` - HTML-Report (Browser-freundlich)

### Datenexporte
- `benchmark_results.json` - (kopiert/validiert)
- `benchmark_results_aggregated.json` - Aggregierte Statistiken
- `benchmark_results_aggregated.csv` - CSV-Export aggregierter Daten

### Visualisierungen
- `benchmark_results.png / .pdf` - Performance-Plots pro Dataset
- `benchmark_results_heatmap.png / .pdf` - Heatmap aller Kombinationen
- `benchmark_results_heatmap_combined.png / .pdf` - Normalisierte Heatmap
- `benchmark_results_combined_dot.png / .pdf` - Dot-Plot Vergleich
- `benchmark_results_pareto.png / .pdf` - Pareto-Front Visualisierung

## Beispiele

### Workflow nach Benchmark-Lauf
```bash
# 1. Normal-Lite-Benchmark starten
make benchmark-normal-lite

# 2. Nach Fertigstellung: Reports regenerieren
make reports-normal-lite

# 3. Reports in `benchmarks/normal_lite/` prüfen
open benchmarks/normal_lite/benchmark_report.html
```

### Reports mit angepasstem Titel
```bash
python3 examples/benchmarks/generate_reports.py \
  --input-dir benchmarks/normal_lite \
  --title "Normal-Lite Benchmark – Q1 2026 Results" \
  --design "10 top discriminative datasets, full estimator catalog (without MUX), reduced repeats"
```

### Batch-Regenerierung
```bash
# Alle vorhandenen Benchmark-Ergebnisse erneuern
for mode in normal-lite standard full; do
  python3 examples/benchmarks/generate_reports.py $mode
done
```

## Technische Details

- **Aggregation**: Verwendet Standardabweichung als error bar
- **Leaderboard**: Top-15 beste Estimators nach F1-Macro sortiert
- **Pareto-Front**: Visualisierung der nicht-dominierten Estimator-Eigenschaften
- **Heatmap**: Standardisierte Performance pro Estimator × Dataset

## Troubleshooting

**Error: "benchmark_results.json nicht gefunden"**
```bash
# Stelle sicher, dass der Benchmark fertig ist und JSON exportiert wurde
ls -la benchmarks/normal_lite/benchmark_results.json
```

**Reports werden nicht aktualisiert**
```bash
# Lösche alte Reports und regeneriere
rm benchmarks/normal_lite/benchmark_report.*
python3 examples/benchmarks/generate_reports.py normal-lite
```

## Performance

- Typische Laufzeit: 5-15 Sekunden pro Benchmark-Set
- Abhängig von: Anzahl Datasets × Estimators × Aggregationsaufwand
- I/O: Liest JSON (typisch 1-5 MB), schreibt Reports (~2-3 MB PNG/PDF + Text)
