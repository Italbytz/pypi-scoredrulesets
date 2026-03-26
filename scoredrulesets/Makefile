.PHONY: help benchmark benchmark-resume benchmark-recover benchmark-clean benchmark-status \
       benchmark-standard benchmark-standard-resume benchmark-standard-clean benchmark-standard-status \
	benchmark-normal benchmark-normal-resume benchmark-normal-clean benchmark-normal-status \
	benchmark-normal-lite benchmark-normal-lite-resume benchmark-normal-lite-clean benchmark-normal-lite-status \
	reports-normal reports-normal-lite reports-standard reports-full

CHECKPOINT ?= benchmarks/checkpoint.jsonl
CHECKPOINT_STANDARD ?= benchmarks/checkpoint_standard.jsonl
CHECKPOINT_NORMAL_LITE ?= benchmarks/checkpoint_normal_lite.jsonl
TIMEOUT    ?= 300
REPEATS    ?= 3
MAX_ATTEMPTS ?= 5

help:
	@echo "Benchmark-Targets:"
	@echo "  make benchmark               - Einmaliger Full-Report (mit Checkpoint/Resume)"
	@echo "  make benchmark-resume        - Startet/setzt Benchmark fort bis alle Laeufe fertig"
	@echo "                                 (max $(MAX_ATTEMPTS) Versuche, Checkpoint: $(CHECKPOINT))"
	@echo "  make benchmark-clean         - Loescht Checkpoint und startet von vorne"
	@echo "  make benchmark-status        - Zeigt Checkpoint-Status (fertige Laeufe)"
	@echo ""
	@echo "Standard-Benchmark-Targets (alle Schaetzer, 10 Datasets):"
	@echo "  make benchmark-standard          - Einmaliger Standard-Report (mit Checkpoint/Resume)"
	@echo "  make benchmark-normal            - Alias fuer benchmark-standard"
	@echo "  make benchmark-standard-resume   - Startet/setzt Standard-Benchmark fort"
	@echo "                                     (max $(MAX_ATTEMPTS) Versuche, Checkpoint: $(CHECKPOINT_STANDARD))"
	@echo "  make benchmark-standard-clean    - Loescht Standard-Checkpoint und startet von vorne"
	@echo "  make benchmark-standard-status   - Zeigt Standard-Checkpoint-Status (fertige Laeufe)"
	@echo ""
	@echo "Normal-Lite-Benchmark-Targets (alle Schaetzer, 10 Datasets, reduzierte Repeats):"
	@echo "  make benchmark-normal-lite        - Einmaliger Normal-Lite-Report"
	@echo "  make benchmark-normal-lite-resume - Startet/setzt Normal-Lite-Benchmark fort"
	@echo "                                       (Checkpoint: $(CHECKPOINT_NORMAL_LITE))"
	@echo "  make benchmark-normal-lite-clean  - Loescht Normal-Lite-Checkpoint und startet von vorne"
	@echo "  make benchmark-normal-lite-status - Zeigt Normal-Lite-Checkpoint-Status"
	@echo ""
	@echo "Report-Generierung (aus existierenden Ergebnissen):"
	@echo "  make reports-normal          - Alias fuer reports-standard"
	@echo "  make reports-normal-lite     - Regeneriere Reports fuer Normal-Lite-Benchmark"
	@echo "  make reports-standard        - Regeneriere Reports fuer Standard-Benchmark"
	@echo "  make reports-full            - Regeneriere Reports fuer Full-Benchmark"
	@echo ""
	@echo "Optionen (als Variablen):"
	@echo "  CHECKPOINT=path.jsonl  CHECKPOINT_STANDARD=path.jsonl  CHECKPOINT_NORMAL_LITE=path.jsonl"
	@echo "  TIMEOUT=300  REPEATS=3  MAX_ATTEMPTS=5"
	@echo "  ARGS='--datasets cart_hard --estimators wrapper_cart,gp'"

benchmark:
	bash scripts/run/run_full_benchmark_with_recovery.sh \
		--checkpoint $(CHECKPOINT) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-resume:
	bash scripts/run/run_full_benchmark_with_recovery.sh \
		--max-attempts $(MAX_ATTEMPTS) \
		--checkpoint $(CHECKPOINT) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-recover: benchmark

benchmark-clean:
	@echo "Loesche Checkpoint: $(CHECKPOINT)"
	rm -f $(CHECKPOINT)
	@echo "Fertig. Naechster 'make benchmark' startet von vorne."

benchmark-status:
	@if [ -f "$(CHECKPOINT)" ]; then \
		echo "Checkpoint: $(CHECKPOINT)"; \
		echo "Zeilen (fertige Laeufe): $$(wc -l < $(CHECKPOINT))"; \
		echo "Dateigroesse: $$(du -h $(CHECKPOINT) | cut -f1)"; \
		echo "Letzte 3 Laeufe:"; \
		tail -3 $(CHECKPOINT) | python -m json.tool --compact 2>/dev/null || tail -3 $(CHECKPOINT); \
	else \
		echo "Kein Checkpoint vorhanden: $(CHECKPOINT)"; \
	fi

# ---------------------------------------------------------------------------
# Standard-Benchmark (alle Schaetzer, 10 ausgewaehlte Datasets)
# ---------------------------------------------------------------------------

benchmark-standard:
	bash scripts/run/run_standard_benchmark_with_recovery.sh \
		--checkpoint $(CHECKPOINT_STANDARD) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-standard-resume:
	bash scripts/run/run_standard_benchmark_with_recovery.sh \
		--max-attempts $(MAX_ATTEMPTS) \
		--checkpoint $(CHECKPOINT_STANDARD) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-standard-clean:
	@echo "Loesche Standard-Checkpoint: $(CHECKPOINT_STANDARD)"
	rm -f $(CHECKPOINT_STANDARD)
	@echo "Fertig. Naechster 'make benchmark-standard' startet von vorne."

benchmark-standard-status:
	@if [ -f "$(CHECKPOINT_STANDARD)" ]; then \
		echo "Standard-Checkpoint: $(CHECKPOINT_STANDARD)"; \
		echo "Zeilen (fertige Laeufe): $$(wc -l < $(CHECKPOINT_STANDARD))"; \
		echo "Dateigroesse: $$(du -h $(CHECKPOINT_STANDARD) | cut -f1)"; \
		echo "Letzte 3 Laeufe:"; \
		tail -3 $(CHECKPOINT_STANDARD) | python -m json.tool --compact 2>/dev/null || tail -3 $(CHECKPOINT_STANDARD); \
	else \
		echo "Kein Standard-Checkpoint vorhanden: $(CHECKPOINT_STANDARD)"; \
	fi

benchmark-normal: benchmark-standard
benchmark-normal-resume: benchmark-standard-resume
benchmark-normal-clean: benchmark-standard-clean
benchmark-normal-status: benchmark-standard-status

# ---------------------------------------------------------------------------
# Normal-Lite-Benchmark (alle Schaetzer, 10 Datasets, weniger Repeats)
# ---------------------------------------------------------------------------

benchmark-normal-lite:
	bash scripts/run/run_normal_lite_benchmark_with_recovery.sh \
		--checkpoint $(CHECKPOINT_NORMAL_LITE) \
		--timeout $(TIMEOUT) \
		--repeats 2 \
		$(ARGS)

benchmark-normal-lite-resume:
	bash scripts/run/run_normal_lite_benchmark_with_recovery.sh \
		--checkpoint $(CHECKPOINT_NORMAL_LITE) \
		--timeout $(TIMEOUT) \
		--repeats 2 \
		$(ARGS)

benchmark-normal-lite-clean:
	@echo "Loesche Normal-Lite-Checkpoint: $(CHECKPOINT_NORMAL_LITE)"
	rm -f $(CHECKPOINT_NORMAL_LITE)
	@echo "Fertig. Naechster 'make benchmark-normal-lite' startet von vorne."

benchmark-normal-lite-status:
	@if [ -f "$(CHECKPOINT_NORMAL_LITE)" ]; then \
		echo "Normal-Lite-Checkpoint: $(CHECKPOINT_NORMAL_LITE)"; \
		echo "Zeilen (fertige Laeufe): $$(wc -l < $(CHECKPOINT_NORMAL_LITE))"; \
		echo "Dateigroesse: $$(du -h $(CHECKPOINT_NORMAL_LITE) | cut -f1)"; \
		echo "Letzte 3 Laeufe:"; \
		tail -3 $(CHECKPOINT_NORMAL_LITE) | python -m json.tool --compact 2>/dev/null || tail -3 $(CHECKPOINT_NORMAL_LITE); \
	else \
		echo "Kein Normal-Lite-Checkpoint vorhanden: $(CHECKPOINT_NORMAL_LITE)"; \
	fi

# ---------------------------------------------------------------------------
# Report-Generierung (aus existierenden Benchmark-Ergebnissen)
# ---------------------------------------------------------------------------
# Nach einem fertigen Benchmark können Reports unabhängig neu erstellt werden.
# Nützlich für: neue Layouts, andere Aggregationsmethoden, Dokumentation update.

reports-normal-lite:
	@echo "Regeneriere Reports fuer Normal-Lite-Benchmark..."
	python3 -u examples/benchmarks/generate_reports.py normal-lite

reports-normal: reports-standard

reports-standard:
	@echo "Regeneriere Reports fuer Standard-Benchmark..."
	python3 -u examples/benchmarks/generate_reports.py standard

reports-full:
	@echo "Regeneriere Reports fuer Full-Benchmark..."
	python3 -u examples/benchmarks/generate_reports.py full

