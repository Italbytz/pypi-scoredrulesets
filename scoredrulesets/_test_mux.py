"""Schnelltest: Multiplexer-Datensaetze und No-Split-Modus."""
from scoredrulesets.benchmarking.datasets import (
    generate_multiplexer_dataset,
    load_multiplexer_datasets,
)
import numpy as np

# 1) Generatorkorrektheit pruefen
for n_addr in [2, 3, 4]:
    X, y = generate_multiplexer_dataset(n_addr, max_samples=5000 if n_addr == 4 else None)
    n_feat = n_addr + (1 << n_addr)
    errors = 0
    for i in range(X.shape[0]):
        addr = sum(X[i, a] << a for a in range(n_addr))
        expected = X[i, n_addr + addr]
        if y[i] != expected:
            errors += 1
    print(f"{n_addr} addr bits: {X.shape[0]:>6} rows, {X.shape[1]:>2} cols, errors={errors}, balance={np.bincount(y).tolist()}")

# 2) load_multiplexer_datasets
bundles = load_multiplexer_datasets()
for name, b in bundles.items():
    print(f"  {name}: shape={b.X.shape}, no_split={b.no_split}")

# 3) No-Split im Runner testen
from scoredrulesets.benchmarking import BenchmarkConfig, run_benchmarks
config = BenchmarkConfig(
    dataset_names=["mux_6"],
    estimator_names=["wrapper_cart_mux"],
    repeats=1,
    show_progress=False,
)
results = run_benchmarks(config)
for r in results:
    print(f"  {r.estimator} on {r.dataset}: F1={r.f1_macro:.4f}, n_train={r.n_train}, n_test={r.n_test}")
    assert r.n_train == r.n_test, "No-split: n_train should equal n_test"
    assert abs(r.f1_macro - 1.0) < 1e-9, f"CART should achieve F1=1.0 on mux_6, got {r.f1_macro}"

print("\nAlle Tests bestanden!")
