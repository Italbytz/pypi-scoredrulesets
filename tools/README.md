# Tools

Dieser Ordner enthaelt **nicht-pipelinekritische** Hilfswerkzeuge.

## Struktur

- `tools/diagnostics/`: One-off-Diagnosen, Debug-Experimente, lokale Ursachenanalyse
- `tools/analysis/`: Vergleichs- und Analyse-Skripte fuer Evaluationsfragen

Diese Skripte sind bewusst getrennt von `examples/` und `scripts/run/`, damit
klar ist, was stabiler Entry-Point ist und was explorativ/temporär ist.

## Wichtiger Hinweis

- Keine Makefile-Kernziele sollten auf `tools/` zeigen.
- API/CLI-Kompatibilitaet von `tools/` ist nicht garantiert.
