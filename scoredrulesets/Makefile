.PHONY: help benchmark benchmark-resume benchmark-recover benchmark-clean benchmark-status recover-latest

CHECKPOINT ?= benchmarks/checkpoint.jsonl
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
	@echo "Optionen (als Variablen):"
	@echo "  CHECKPOINT=path.jsonl  TIMEOUT=300  REPEATS=3  MAX_ATTEMPTS=5"
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
