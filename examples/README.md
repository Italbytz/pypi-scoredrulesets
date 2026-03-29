# Examples

Die Beispiele sind jetzt nach Zweck getrennt:

- `examples/estimators/`: Endnutzer-Demos fuer einzelne Schaetzer/Backends
- `examples/benchmarks/`: Benchmark- und Reporting-Entrypoints

## Estimator-Demos

Direkter RuleLCS-Schaetzer:

```bash
python3 examples/estimators/example_ruleplcs_backend.py
```

RuleLCS ueber Wrapper:

```bash
python3 examples/estimators/example_ruleplcs_wrapper.py
```

Weitere fachliche Demos:

- `examples/estimators/example_rulenln_backend.py`
- `examples/estimators/example_rule_shrinking.py`
- `examples/estimators/example_exstracs_shrinking.py`
- `examples/estimators/example_rulekit_exstracs.py`

## Benchmark-Entrypoints

Full:

```bash
python3 examples/benchmarks/benchmark_full_report.py
```

Standard (aka Normal):

```bash
python3 examples/benchmarks/benchmark_standard.py
```

Normal-Lite:

```bash
python3 examples/benchmarks/benchmark_normal_lite.py
```

Reports aus vorhandenen Benchmark-Ergebnissen regenerieren:

```bash
python3 examples/benchmarks/generate_reports.py normal-lite
python3 examples/benchmarks/generate_reports.py standard
python3 examples/benchmarks/generate_reports.py full
```

## Bevorzugter Einstieg

Fuer den Alltag sind die Makefile-Targets die stabilste Schnittstelle:

```bash
make benchmark
make benchmark-standard
make benchmark-normal-lite
make reports-standard
make reports-normal-lite
```
