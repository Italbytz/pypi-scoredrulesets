# Examples

This folder contains runnable demos for the Pittsburgh-style backend and wrapper integration.

## Pittsburgh Direct Estimator

File: `examples/example_pittsburgh_backend.py`

What it shows:
- direct training with `PittsburghRuleSetClassifier`
- metadata and ruleset table output
- mini comparison against `ScoredRuleSetClassifier(backend="cart")` and `GeneticScoredRuleSetClassifier`

Run:

```bash
python examples/example_pittsburgh_backend.py
```

Run with profile and seed:

```bash
python examples/example_pittsburgh_backend.py --profile fast --random-state 42
python examples/example_pittsburgh_backend.py --profile strong --random-state 0
python examples/example_pittsburgh_backend.py --profile diverse --random-state 7
```

Available profiles:
- `default`
- `fast`
- `strong`
- `diverse`

When to use which profile:
- `fast`: quick smoke checks, CI sanity runs, and rapid iteration while editing code
- `default`: balanced starting point for everyday local experiments
- `strong`: larger search budget when model quality is more important than runtime
- `diverse`: broader rule-set exploration when you want alternative model structures

## Pittsburgh Wrapper Backend

File: `examples/example_pittsburgh_wrapper.py`

What it shows:
- training through `ScoredRuleSetClassifier(backend="pittsburgh")`
- metadata and ruleset table output
- mini wrapper comparison against `wrapper_cart_d2` and `wrapper_cart_d4`

Run:

```bash
python examples/example_pittsburgh_wrapper.py
```

Run with profile and seed:

```bash
python examples/example_pittsburgh_wrapper.py --profile fast --random-state 42
python examples/example_pittsburgh_wrapper.py --profile strong --random-state 0
python examples/example_pittsburgh_wrapper.py --profile diverse --random-state 7
```

Available profiles:
- `default`
- `fast`
- `strong`
- `diverse`

When to use which profile:
- `fast`: quick smoke checks and wrapper-path sanity checks
- `default`: balanced wrapper baseline for local runs
- `strong`: stronger search setup for higher-quality candidate models
- `diverse`: emphasizes broader rule-set exploration in wrapper mode

## Michigan Wrapper Backend

File: `examples/example_michigan_wrapper.py`

What it shows:
- training through `ScoredRuleSetClassifier(backend="michigan")`
- metadata and ruleset table output
- mini wrapper comparison against `wrapper_cart_d2` and `wrapper_cart_d4`

Run:

```bash
python examples/example_michigan_wrapper.py
```

Run with profile and seed:

```bash
python examples/example_michigan_wrapper.py --profile fast --random-state 42
python examples/example_michigan_wrapper.py --profile strong --random-state 0
```

Available profiles:
- `default`
- `fast`
- `strong`
- `compact`

When to use which profile:
- `fast`: quickest Michigan sanity checks and CI-smoke style runs
- `default`: balanced Michigan baseline for normal local experiments
- `strong`: larger Michigan population/epochs when model quality matters more than runtime
- `compact`: stronger Michigan training with explicit final rule cap for shorter, more readable models

## Michigan Direct Estimator

File: `examples/example_michigan_backend.py`

What it shows:
- direct training with `MichiganRuleSetClassifier`
- metadata and ruleset table output
- mini comparison against `ScoredRuleSetClassifier(backend="cart")` and `GeneticScoredRuleSetClassifier`

Run:

```bash
python examples/example_michigan_backend.py
```

Run with profile and seed:

```bash
python examples/example_michigan_backend.py --profile fast --random-state 42
python examples/example_michigan_backend.py --profile strong --random-state 0
```

Available profiles:
- `default`
- `fast`
- `strong`
- `compact`

When to use which profile:
- `fast`: quickest direct-estimator smoke checks
- `default`: balanced direct Michigan baseline for local experiments
- `strong`: larger direct Michigan search budget for quality-focused runs
- `compact`: quality-oriented direct training with explicit final rule cap

## Recover heatmaps from the latest benchmark log

Files:
- `examples/recover_heatmap_from_log.py`
- `examples/recover_latest_benchmark.py`

What it shows:
- recovers partial benchmark results from a console log
- writes recovered CSV/JSON plus aggregated CSV/JSON
- writes Markdown and HTML leaderboard reports for the recovered subset
- renders the standard heatmap and the combined multi-metric heatmap
- can automatically use the newest log from `benchmarks/logs/`

Run the recovery explicitly on the newest full-report log:

```bash
python examples/recover_heatmap_from_log.py --latest-log
```

Run the one-command convenience wrapper:

```bash
python examples/recover_latest_benchmark.py
```

Optional custom output directory:

```bash
python examples/recover_latest_benchmark.py --output-dir benchmarks/recovered-from-log/manual
```

Expected output location by default:
- `benchmarks/recovered-from-log/<log-stem>/`

## Quick smoke check

```bash
python -m pytest tests/test_pittsburgh_estimator.py tests/test_michigan_estimator.py -k "example_run_demo_smoke or wrapper_example_run_demo_smoke" -q
```

## Full benchmark with automatic recovery

Run the full benchmark with automatic recovery on failure:

```bash
python examples/benchmark_full_report_with_recovery.py
```

Optional custom log and recovery directory:

```bash
python examples/benchmark_full_report_with_recovery.py \
  --log-file benchmarks/logs/manual_full_run.log \
  --recovery-output-dir benchmarks/recovered-from-log/manual_full_run
```

Behavior:
- runs `examples/benchmark_full_report.py`
- writes a console log file
- if the run crashes, automatically recovers CSV/JSON, Markdown/HTML reports, plus both heatmaps from that log
