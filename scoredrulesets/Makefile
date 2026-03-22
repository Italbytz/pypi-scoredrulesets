.PHONY: help benchmark benchmark-recover recover-latest

help:
	@echo "Available targets:"
	@echo "  make benchmark [ARGS='...']         - Run full benchmark with auto-recovery"
	@echo "  make benchmark-recover [ARGS='...'] - Alias for benchmark"
	@echo "  make recover-latest [ARGS='...']    - Recover artifacts from latest benchmark log"

benchmark:
	./scripts/run_full_benchmark_with_recovery.sh $(ARGS)

benchmark-recover: benchmark

recover-latest:
	./scripts/recover_latest_benchmark.sh $(ARGS)

