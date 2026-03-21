from scoredrulesets.benchmarking import (
    AggregatedBenchmarkResult,
    compare_benchmark_runs,
    format_benchmark_comparison_report_html,
    format_benchmark_comparison_report_markdown,
    format_benchmark_meta_report_html,
    format_benchmark_meta_report_markdown,
)


def _row(dataset: str, estimator: str, f1: float, fit_s: float, n_rules: float, n_atoms: float) -> AggregatedBenchmarkResult:
    return AggregatedBenchmarkResult(
        dataset=dataset,
        estimator=estimator,
        n_repeats=3,
        status="ok",
        f1_macro_mean=f1,
        f1_macro_error=0.01,
        fit_seconds_mean=fit_s,
        fit_seconds_error=0.001,
        predict_seconds_mean=0.0001,
        predict_seconds_error=0.0,
        n_rules_mean=n_rules,
        n_rules_error=0.0,
        n_atoms_mean=n_atoms,
        n_atoms_error=0.0,
        ruleset_json_bytes_mean=1000.0,
        ruleset_json_bytes_error=10.0,
    )


def test_compare_benchmark_runs_computes_dataset_deltas():
    core = [
        _row("d1", "wrapper_cart", 0.90, 0.001, 8, 20),
        _row("d1", "native", 0.85, 0.010, 7, 8),
        _row("d2", "wrapper_cart", 0.80, 0.002, 10, 30),
    ]
    compare = [
        _row("d1", "wrapper_hs", 0.92, 0.005, 10, 35),
        _row("d1", "wrapper_cart", 0.89, 0.001, 8, 20),
        _row("d2", "wrapper_hs", 0.78, 0.004, 11, 33),
        _row("d2", "wrapper_cart", 0.81, 0.002, 10, 30),
    ]

    rows = compare_benchmark_runs(core, compare)
    assert len(rows) == 2

    d1 = next(row for row in rows if row.dataset == "d1")
    assert d1.compare_best_estimator == "wrapper_hs"
    assert round(d1.compare_minus_core_best_f1, 4) == 0.02
    assert round(d1.focus_minus_baseline_f1, 4) == 0.03

    d2 = next(row for row in rows if row.dataset == "d2")
    assert d2.compare_best_estimator == "wrapper_cart"
    assert round(d2.compare_minus_core_best_f1, 4) == 0.01
    assert round(d2.focus_minus_baseline_f1, 4) == -0.03


def test_format_benchmark_comparison_report_markdown_contains_sections():
    rows = [
        type("R", (), {
            "dataset": "d1",
            "core_best_estimator": "wrapper_cart",
            "core_best_f1": 0.90,
            "compare_best_estimator": "wrapper_hs",
            "compare_best_f1": 0.92,
            "compare_minus_core_best_f1": 0.02,
            "focus_estimator": "wrapper_hs",
            "focus_f1": 0.92,
            "baseline_estimator": "wrapper_cart",
            "baseline_f1": 0.89,
            "focus_minus_baseline_f1": 0.03,
            "focus_minus_baseline_fit_seconds": 0.004,
            "focus_minus_baseline_rules": 2.0,
        })()
    ]

    report = format_benchmark_comparison_report_markdown(
        rows,
        core_label="core_run",
        compare_label="hs_run",
        core_artifacts={"aggregated_json": "core.json"},
        compare_artifacts={"aggregated_json": "hs.json"},
    )
    assert "# Core vs HS Benchmark Comparison" in report
    assert "## Executive Summary" in report
    assert "## Inputs" in report
    assert "## Dataset Comparison" in report
    assert "## Interpretation" in report
    assert "| dataset | core_best | core_f1 | compare_best |" in report
    assert "core_run" in report
    assert "hs_run" in report


def test_format_benchmark_comparison_report_html_contains_sections():
    rows = [
        type("R", (), {
            "dataset": "d1",
            "core_best_estimator": "wrapper_cart",
            "core_best_f1": 0.90,
            "compare_best_estimator": "wrapper_hs",
            "compare_best_f1": 0.92,
            "compare_minus_core_best_f1": 0.02,
            "focus_estimator": "wrapper_hs",
            "focus_f1": 0.92,
            "baseline_estimator": "wrapper_cart",
            "baseline_f1": 0.89,
            "focus_minus_baseline_f1": 0.03,
            "focus_minus_baseline_fit_seconds": 0.004,
            "focus_minus_baseline_rules": 2.0,
        })()
    ]
    report = format_benchmark_comparison_report_html(
        rows,
        core_label="core_run",
        compare_label="hs_run",
        core_artifacts={"aggregated_json": "core.json"},
        compare_artifacts={"aggregated_json": "hs.json"},
    )
    assert "<!doctype html>" in report
    assert "<h2>Executive Summary</h2>" in report
    assert "<h2>Inputs</h2>" in report
    assert "<h2>Dataset Comparison</h2>" in report
    assert "<h2>Interpretation</h2>" in report
    assert "focus_estimator" in report
    assert "<table>" in report


def test_format_benchmark_meta_report_markdown_contains_snapshots_and_takeaways():
    core = [
        _row("d1", "wrapper_cart", 0.90, 0.001, 8, 20),
        _row("d1", "native", 0.85, 0.010, 7, 8),
    ]
    compare = [
        _row("d1", "wrapper_hs", 0.92, 0.005, 10, 35),
        _row("d1", "wrapper_cart", 0.89, 0.001, 8, 20),
    ]
    rows = compare_benchmark_runs(core, compare)

    report = format_benchmark_meta_report_markdown(
        core,
        compare,
        rows,
        core_label="core_run",
        compare_label="hs_run",
        core_artifacts={"aggregated_json": "core.json"},
        compare_artifacts={"aggregated_json": "hs.json"},
        comparison_artifacts={"comparison": "compare.md"},
    )

    assert "# Combined Benchmark Meta Report" in report
    assert "## Executive Summary" in report
    assert "## Snapshot: core_run" in report
    assert "## Snapshot: hs_run" in report
    assert "## Direct Comparison" in report
    assert "## Per-Dataset Takeaways" in report
    assert "best_in_core_run" in report
    assert "best_in_hs_run" in report
    assert "focus_estimator" in report


def test_format_benchmark_meta_report_html_contains_snapshots_and_takeaways():
    core = [
        _row("d1", "wrapper_cart", 0.90, 0.001, 8, 20),
        _row("d1", "native", 0.85, 0.010, 7, 8),
    ]
    compare = [
        _row("d1", "wrapper_hs", 0.92, 0.005, 10, 35),
        _row("d1", "wrapper_cart", 0.89, 0.001, 8, 20),
    ]
    rows = compare_benchmark_runs(core, compare)

    report = format_benchmark_meta_report_html(
        core,
        compare,
        rows,
        core_label="core_run",
        compare_label="hs_run",
        core_artifacts={"aggregated_json": "core.json"},
        compare_artifacts={"aggregated_json": "hs.json"},
        comparison_artifacts={"focus_estimator": "wrapper_hs", "baseline_estimator": "wrapper_cart"},
    )
    assert "<!doctype html>" in report
    assert "<h2>Executive Summary</h2>" in report
    assert "<h2>Snapshot: core_run</h2>" in report
    assert "<h2>Snapshot: hs_run</h2>" in report
    assert "<h2>Direct Comparison</h2>" in report
    assert "<h2>Per-Dataset Takeaways</h2>" in report
    assert "focus_estimator" in report
    assert "best_in_core_run" in report


