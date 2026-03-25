.PHONY: help benchmark benchmark-resume benchmark-recover benchmark-clean benchmark-status recover-latest \
       benchmark-standard benchmark-standard-resume benchmark-standard-clean benchmark-standard-status \
       benchmark-slim benchmark-slim-resume benchmark-slim-clean benchmark-slim-status \
       benchmark-logicgp benchmark-logicgp-clean benchmark-logicgp-status

CHECKPOINT ?= benchmarks/checkpoint.jsonl
CHECKPOINT_STANDARD ?= benchmarks/checkpoint_standard.jsonl
CHECKPOINT_SLIM ?= benchmarks/checkpoint_slim.jsonl
CHECKPOINT_LOGICGP ?= benchmarks/checkpoint_logicgp.jsonl
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
	@echo "  make recover-latest          - Recovery aus letztem Log"
	@echo ""
	@echo "Standard-Benchmark-Targets (alle Schaetzer, 10 Datasets):"
	@echo "  make benchmark-standard          - Einmaliger Standard-Report (mit Checkpoint/Resume)"
	@echo "  make benchmark-standard-resume   - Startet/setzt Standard-Benchmark fort"
	@echo "                                     (max $(MAX_ATTEMPTS) Versuche, Checkpoint: $(CHECKPOINT_STANDARD))"
	@echo "  make benchmark-standard-clean    - Loescht Standard-Checkpoint und startet von vorne"
	@echo "  make benchmark-standard-status   - Zeigt Standard-Checkpoint-Status (fertige Laeufe)"
	@echo ""
	@echo "Slim-Benchmark-Targets (9 Schaetzer, 10 Datasets):"
	@echo "  make benchmark-slim          - Einmaliger Slim-Report (mit Checkpoint/Resume)"
	@echo "  make benchmark-slim-resume   - Startet/setzt Slim-Benchmark fort"
	@echo "                                 (max $(MAX_ATTEMPTS) Versuche, Checkpoint: $(CHECKPOINT_SLIM))"
	@echo "  make benchmark-slim-clean    - Loescht Slim-Checkpoint und startet von vorne"
	@echo "  make benchmark-slim-status   - Zeigt Slim-Checkpoint-Status (fertige Laeufe)"
	@echo ""
	@echo "LogicGP-Varianten-Benchmark (12 Varianten, 9 Datasets):"
	@echo "  make benchmark-logicgp       - LogicGP-Varianten-Benchmark mit Paarvergleich"
	@echo "  make benchmark-logicgp-clean - Loescht LogicGP-Checkpoint und startet von vorne"
	@echo "  make benchmark-logicgp-status- Zeigt LogicGP-Checkpoint-Status"
	@echo ""
	@echo "Optionen (als Variablen):"
	@echo "  CHECKPOINT=path.jsonl  CHECKPOINT_STANDARD=path.jsonl  CHECKPOINT_SLIM=path.jsonl"
	@echo "  TIMEOUT=300  REPEATS=3  MAX_ATTEMPTS=5"
	@echo "  ARGS='--datasets cart_hard --estimators wrapper_cart,gp'"

benchmark:
	./scripts/run_full_benchmark_with_recovery.sh \
		--checkpoint $(CHECKPOINT) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-resume:
	./scripts/run_full_benchmark_with_recovery.sh \
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

recover-latest:
	./scripts/recover_latest_benchmark.sh $(ARGS)

# ---------------------------------------------------------------------------
# Standard-Benchmark (alle Schaetzer, 10 ausgewaehlte Datasets)
# ---------------------------------------------------------------------------

benchmark-standard:
	./scripts/run_standard_benchmark_with_recovery.sh \
		--checkpoint $(CHECKPOINT_STANDARD) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-standard-resume:
	./scripts/run_standard_benchmark_with_recovery.sh \
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

# ---------------------------------------------------------------------------
# Slim-Benchmark (9 Schaetzer, 10 Datasets – maximale Typ-Diskriminierung)
# ---------------------------------------------------------------------------

benchmark-slim:
	./scripts/run_slim_benchmark_with_recovery.sh \
		--checkpoint $(CHECKPOINT_SLIM) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-slim-resume:
	./scripts/run_slim_benchmark_with_recovery.sh \
		--max-attempts $(MAX_ATTEMPTS) \
		--checkpoint $(CHECKPOINT_SLIM) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-slim-clean:
	@echo "Loesche Slim-Checkpoint: $(CHECKPOINT_SLIM)"
	rm -f $(CHECKPOINT_SLIM)
	@echo "Fertig. Naechster 'make benchmark-slim' startet von vorne."

benchmark-slim-status:
	@if [ -f "$(CHECKPOINT_SLIM)" ]; then \
		echo "Slim-Checkpoint: $(CHECKPOINT_SLIM)"; \
		echo "Zeilen (fertige Laeufe): $$(wc -l < $(CHECKPOINT_SLIM))"; \
		echo "Dateigroesse: $$(du -h $(CHECKPOINT_SLIM) | cut -f1)"; \
		echo "Letzte 3 Laeufe:"; \
		tail -3 $(CHECKPOINT_SLIM) | python -m json.tool --compact 2>/dev/null || tail -3 $(CHECKPOINT_SLIM); \
	else \
		echo "Kein Slim-Checkpoint vorhanden: $(CHECKPOINT_SLIM)"; \
	fi

# ---------------------------------------------------------------------------
# LogicGP-Varianten-Benchmark (12 Varianten × 9 Datasets – Achsenvergleich)
# ---------------------------------------------------------------------------

benchmark-logicgp:
	python -u examples/benchmark_logicgp_variants.py \
		--checkpoint $(CHECKPOINT_LOGICGP) \
		--timeout $(TIMEOUT) \
		--repeats $(REPEATS) \
		$(ARGS)

benchmark-logicgp-clean:
	@echo "Loesche LogicGP-Checkpoint: $(CHECKPOINT_LOGICGP)"
	rm -f $(CHECKPOINT_LOGICGP)
	@echo "Fertig. Naechster 'make benchmark-logicgp' startet von vorne."

benchmark-logicgp-status:
	@if [ -f "$(CHECKPOINT_LOGICGP)" ]; then \
		echo "LogicGP-Checkpoint: $(CHECKPOINT_LOGICGP)"; \
		echo "Zeilen (fertige Laeufe): $$(wc -l < $(CHECKPOINT_LOGICGP))"; \
		echo "Dateigroesse: $$(du -h $(CHECKPOINT_LOGICGP) | cut -f1)"; \
		echo "Letzte 3 Laeufe:"; \
		tail -3 $(CHECKPOINT_LOGICGP) | python -m json.tool --compact 2>/dev/null || tail -3 $(CHECKPOINT_LOGICGP); \
	else \
		echo "Kein LogicGP-Checkpoint vorhanden: $(CHECKPOINT_LOGICGP)"; \
	fi

