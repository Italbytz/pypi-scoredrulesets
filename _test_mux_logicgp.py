"""Test: logicGP auf Multiplexer-Datensaetzen (Ziel: F1=1.0)."""
import warnings
warnings.filterwarnings("ignore")

from scoredrulesets.benchmarking import BenchmarkConfig, run_benchmarks

config = BenchmarkConfig(
    dataset_names=["mux_6", "mux_11"],
    estimator_names=["wrapper_logicgp_mux", "wrapper_cart_mux"],
    repeats=1,
    show_progress=True,
)

results = run_benchmarks(config)
print()
print("=" * 60)
for r in results:
    if r.status == "ok":
        marker = "PERFEKT" if abs(r.f1_macro - 1.0) < 1e-9 else f"F1={r.f1_macro:.4f}"
        print(f"  {r.estimator:30s} | {r.dataset:8s} | {marker} | rules={r.n_rules} | {r.fit_seconds:.2f}s")
    else:
        print(f"  {r.estimator:30s} | {r.dataset:8s} | ERROR: {r.error}")

