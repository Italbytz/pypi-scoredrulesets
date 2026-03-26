import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from scoredrulesets.benchmarking import (
    BenchmarkConfig,
    aggregate_benchmark_results,
    build_benchmark_leaderboard,
    format_benchmark_leaderboard_html,
    format_benchmark_leaderboard_markdown,
    format_benchmark_report_html,
    format_benchmark_report_markdown,
    format_benchmark_leaderboard_table,
    plot_benchmark_heatmap,
    plot_benchmark_heatmap_combined,
    plot_benchmark_results,
    run_benchmarks,
)
import scoredrulesets.benchmarking.plotting as plotting_module
from scoredrulesets.benchmarking.datasets import DatasetBundle
from scoredrulesets.benchmarking.runner import (
    AggregatedBenchmarkResult,
    BenchmarkResult,
    _evaluate_transformation_gap,
    _resolve_test_size,
)


def test_benchmarking_runs_for_core_estimators():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs", "wrapper_logicgp"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    assert len(results) == 3
    assert all(r.dataset == "uci_iris" for r in results)
    assert all(r.status == "ok" for r in results)
    assert all(r.f1_macro is not None for r in results)
    assert all(r.n_rules is not None and r.n_rules > 0 for r in results)


def test_benchmarking_hs_is_ok_or_skipped():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_hs"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)
    assert len(results) == 1

    result = results[0]
    if importlib.util.find_spec("imodels") is None:
        assert result.status == "skipped"
        assert result.skip_reason == "missing_optional_dependency"
    else:
        assert result.status in {"ok", "error"}


def test_benchmarking_unknown_dataset_raises():
    config = BenchmarkConfig(dataset_names=["does_not_exist"], estimator_names=["wrapper_cart"])
    try:
        run_benchmarks(config)
    except ValueError as exc:
        assert "Unknown dataset names" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown dataset")


def test_benchmark_plot_is_written_as_png_and_pdf(tmp_path: Path):
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs", "wrapper_logicgp"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    base = tmp_path / "benchmark_plot"
    png_path, pdf_path = plot_benchmark_results(results, base, size_metric="n_rules")

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.suffix == ".png"
    assert pdf_path.suffix == ".pdf"
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_benchmarking_aggregation_over_repeats():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)
    aggregated = aggregate_benchmark_results(results, error_bar="std")

    assert len(aggregated) == 2
    assert all(item.dataset == "uci_iris" for item in aggregated)
    assert all(item.n_repeats == 2 for item in aggregated)
    assert all(item.f1_macro_mean is not None for item in aggregated)
    assert all(item.f1_macro_error is not None for item in aggregated)


def test_benchmarking_smoke_compare_rulelcs_native_gp():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_rulelcs", "wrapper_cart", "wrapper_logicgp"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    assert len(results) == 3
    by_estimator = {result.estimator: result for result in results}
    assert set(by_estimator) == {"wrapper_rulelcs", "wrapper_cart", "wrapper_logicgp"}
    assert all(result.status == "ok" for result in results)
    assert all(result.f1_macro is not None for result in results)
    assert all(result.n_rules is not None and result.n_rules > 0 for result in results)


def test_benchmarking_smoke_compare_rulelcs_profiles():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_rulelcs", "wrapper_rulelcs_strong"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    assert len(results) == 2
    by_estimator = {result.estimator: result for result in results}
    assert set(by_estimator) == {
        "wrapper_rulelcs",
        "wrapper_rulelcs_strong",
    }
    assert all(result.status == "ok" for result in results)
    assert all(result.f1_macro is not None for result in results)
    assert all(result.n_rules is not None and result.n_rules > 0 for result in results)


def test_benchmarking_progress_output_is_emitted(capsys):
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart"],
        repeats=1,
        random_state=0,
        show_progress=True,
    )

    results = run_benchmarks(config)
    captured = capsys.readouterr()

    assert len(results) == 1
    assert "[progress] Benchmark started:" in captured.out
    assert "[progress 1/1] START" in captured.out
    assert "[progress 1/1] DONE status=OK" in captured.out


def test_benchmark_plot_aggregated_is_written_as_png_and_pdf(tmp_path: Path):
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)

    base = tmp_path / "benchmark_plot_aggregated"
    png_path, pdf_path = plot_benchmark_results(
        results,
        base,
        size_metric="n_rules",
        aggregate_repeats=True,
        error_bar="std",
    )

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_benchmark_plot_multiple_datasets_is_written_as_png_and_pdf(tmp_path: Path):
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris", "sklearn_wine"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)

    base = tmp_path / "benchmark_plot_multi_dataset"
    png_path, pdf_path = plot_benchmark_results(
        results,
        base,
        size_metric="n_rules",
        aggregate_repeats=True,
        error_bar="std",
    )

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_benchmark_heatmap_is_written_as_png_and_pdf(tmp_path: Path):
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris", "sklearn_wine"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)

    base = tmp_path / "benchmark_heatmap"
    png_path, pdf_path = plot_benchmark_heatmap(results, base, error_bar="std")

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_combined_heatmap_cell_uses_shared_colour_scale():
    fig, ax = plt.subplots()
    cmap = plt.colormaps["Greens"]
    f1_norm = Normalize(vmin=0.0, vmax=1.0)
    atom_norm = LogNorm(vmin=10.0, vmax=1000.0)
    fit_norm = LogNorm(vmin=0.1, vmax=10.0)
    entry = AggregatedBenchmarkResult(
        dataset="synthetic",
        estimator="wrapper_rulelcs",
        n_repeats=1,
        status="ok",
        f1_macro_mean=0.75,
        f1_macro_error=0.02,
        fit_seconds_mean=1.0,
        fit_seconds_error=0.1,
        predict_seconds_mean=0.05,
        predict_seconds_error=0.01,
        n_rules_mean=8.0,
        n_rules_error=0.0,
        n_atoms_mean=100.0,
        n_atoms_error=5.0,
        ruleset_json_bytes_mean=256.0,
        ruleset_json_bytes_error=0.0,
    )

    plotting_module._draw_combined_cell(ax, 0, 0, entry, cmap, f1_norm, atom_norm, fit_norm)

    assert len(ax.patches) == 3
    np.testing.assert_allclose(ax.patches[0].get_facecolor(), cmap(0.75))
    np.testing.assert_allclose(ax.patches[1].get_facecolor(), cmap(1.0 - float(atom_norm(100.0))))
    np.testing.assert_allclose(ax.patches[2].get_facecolor(), cmap(1.0 - float(fit_norm(1.0))))
    plt.close(fig)


def test_combined_heatmap_adds_single_unlabelled_legend_bar(tmp_path: Path, monkeypatch):
    legend_calls: list[dict[str, object]] = []
    original_add_legend_bar = plotting_module._add_legend_bar

    def _capture_add_legend_bar(*args, **kwargs):
        legend_calls.append(dict(kwargs))
        return original_add_legend_bar(*args, **kwargs)

    monkeypatch.setattr(plotting_module, "_add_legend_bar", _capture_add_legend_bar)

    results = [
        BenchmarkResult(
            dataset="synthetic_a",
            estimator="wrapper_rulelcs",
            repeat=0,
            status="ok",
            skip_reason=None,
            error=None,
            f1_macro=0.81,
            fit_seconds=0.4,
            predict_seconds=0.05,
            n_rules=6,
            n_atoms=24,
            ruleset_json_bytes=320,
            n_train=80,
            n_test=20,
        ),
        BenchmarkResult(
            dataset="synthetic_a",
            estimator="wrapper_cart",
            repeat=0,
            status="ok",
            skip_reason=None,
            error=None,
            f1_macro=0.76,
            fit_seconds=1.2,
            predict_seconds=0.05,
            n_rules=9,
            n_atoms=60,
            ruleset_json_bytes=480,
            n_train=80,
            n_test=20,
        ),
    ]

    base = tmp_path / "benchmark_heatmap_combined"
    png_path, pdf_path = plot_benchmark_heatmap_combined(results, base, error_bar="std")

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
    assert len(legend_calls) == 1
    assert legend_calls[0]["show_ticks"] is False
    assert legend_calls[0].get("label") is None


def test_benchmark_leaderboard_sorting_and_markdown_output():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs", "wrapper_logicgp"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated, primary_metric="f1_macro_mean")

    assert len(leaderboard) == 3
    assert leaderboard[0].f1_macro_mean >= leaderboard[1].f1_macro_mean

    markdown = format_benchmark_leaderboard_markdown(leaderboard)
    table = format_benchmark_leaderboard_table(leaderboard)
    assert "| rank | dataset | estimator |" in markdown
    assert "+-" in table
    assert leaderboard[0].estimator in markdown


def test_benchmark_report_markdown_contains_sections_and_artifacts():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated, primary_metric="f1_macro_mean")

    report = format_benchmark_report_markdown(
        leaderboard,
        title="Unit Test Report",
        config={"repeats": 2, "error_bar": "std"},
        artifact_paths={
            "plot_png": "benchmark.png",
            "plot_pdf": "benchmark.pdf",
            "heatmap_png": "benchmark_heatmap.png",
            "heatmap_pdf": "benchmark_heatmap.pdf",
        },
        notes=["synthetic unit test"],
    )

    assert "# Unit Test Report" in report
    assert "## Summary" in report
    assert "## Configuration" in report
    assert "## Artifacts" in report
    assert "## Plot Preview" in report
    assert "## Heatmap Preview" not in report
    assert "## Notes" in report
    assert "## Top per Dataset" in report
    assert "### uci_iris" in report
    assert "## Leaderboard" in report
    assert "## Dataset: uci_iris" in report
    assert "[benchmark.png](benchmark.png)" in report
    assert "![Benchmark plot](benchmark.png)" in report
    assert "_Heatmap add-on: compact overview of aggregated F1 values and fit times per dataset/estimator._" in report
    assert "![Benchmark heatmap](benchmark_heatmap.png)" in report
    assert report.index("![Benchmark plot](benchmark.png)") < report.index("![Benchmark heatmap](benchmark_heatmap.png)")
    assert "**top_1_model**" in report
    assert "**best_model**" in report
    assert "**smallest_model**" in report
    assert "**fastest_model**" in report
    assert "| rank | dataset | estimator |" in report
    assert report.index("## Top per Dataset") < report.index("## Leaderboard")


def test_benchmark_report_html_contains_sections_and_preview():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated, primary_metric="f1_macro_mean")

    html = format_benchmark_report_html(
        leaderboard,
        title="Unit Test Report",
        config={"repeats": 2, "error_bar": "std"},
        artifact_paths={
            "plot_png": "benchmark.png",
            "plot_pdf": "benchmark.pdf",
            "heatmap_png": "benchmark_heatmap.png",
            "heatmap_pdf": "benchmark_heatmap.pdf",
        },
        notes=["synthetic unit test"],
    )

    assert "<!doctype html>" in html
    assert "<h1>Unit Test Report</h1>" in html
    assert "<h2>Summary</h2>" in html
    assert "<h2>Plot Preview</h2>" in html
    assert "<h2>Heatmap Preview</h2>" not in html
    assert "<img src='benchmark.png' alt='Benchmark plot'>" in html
    assert "Heatmap add-on: compact overview of aggregated F1 values and fit times per dataset/estimator." in html
    assert "<img src='benchmark_heatmap.png' alt='Benchmark heatmap'>" in html
    assert html.index("<img src='benchmark.png' alt='Benchmark plot'>") < html.index("<img src='benchmark_heatmap.png' alt='Benchmark heatmap'>")
    assert "<h2>Top per Dataset</h2>" in html
    assert "<h2>Leaderboard</h2>" in html
    assert "<h2>Dataset: uci_iris</h2>" in html
    assert "<table>" in html


def test_benchmark_leaderboard_html_contains_table():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "wrapper_rulelcs"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)
    aggregated = aggregate_benchmark_results(results, error_bar="std")
    leaderboard = build_benchmark_leaderboard(aggregated, primary_metric="f1_macro_mean")

    html = format_benchmark_leaderboard_html(leaderboard)
    assert "<table>" in html
    assert "<th>dataset</th>" in html
    assert leaderboard[0].dataset in html


def test_paper_split_policy_thresholds():
    small = DatasetBundle(
        name="small",
        X=np.zeros((150, 1)),
        y=np.zeros(150),
        source="test",
    )
    medium = DatasetBundle(
        name="medium",
        X=np.zeros((1200, 1)),
        y=np.zeros(1200),
        source="test",
    )
    large = DatasetBundle(
        name="large",
        X=np.zeros((6000, 1)),
        y=np.zeros(6000),
        source="test",
    )

    cfg = BenchmarkConfig(use_paper_split_policy=True)
    assert _resolve_test_size(small, cfg) == 0.30
    assert _resolve_test_size(medium, cfg) == 0.25
    assert _resolve_test_size(large, cfg) == 0.20


def test_transformation_gap_standard_estimators_still_abort_on_large_gap():
    action, message = _evaluate_transformation_gap(
        estimator_name="wrapper_exstracs",
        is_lossy=False,
        f1_native=0.91,
        f1_transformed=0.16,
        dataset_name="sklearn_iris",
    )
    assert action == "abort"
    assert message is not None
    assert "zerstört" in message


def test_transformation_gap_exstracs_shrink_warns_on_moderate_gap():
    action, message = _evaluate_transformation_gap(
        estimator_name="wrapper_exstracs_pruned",
        is_lossy=False,
        f1_native=0.91,
        f1_transformed=0.73,
        dataset_name="sklearn_iris",
    )
    assert action == "warn"
    assert message is not None
    assert "Deutliche Warnung" in message


def test_transformation_gap_exstracs_shrink_still_aborts_on_very_large_gap():
    action, message = _evaluate_transformation_gap(
        estimator_name="wrapper_exstracs_pruned",
        is_lossy=False,
        f1_native=0.91,
        f1_transformed=0.16,
        dataset_name="sklearn_iris",
    )
    assert action == "abort"
    assert message is not None
    assert "stark zerstört" in message


def test_aggregate_benchmark_results_persists_warning_metadata():
    results = [
        BenchmarkResult(
            dataset="sklearn_iris",
            estimator="wrapper_exstracs_pruned",
            repeat=0,
            status="ok",
            skip_reason=None,
            error=None,
            f1_macro=0.80,
            fit_seconds=0.1,
            predict_seconds=0.01,
            n_rules=10,
            n_atoms=20,
            ruleset_json_bytes=123,
            n_train=100,
            n_test=50,
            validation_action="warn",
            validation_message="example warning",
        ),
        BenchmarkResult(
            dataset="sklearn_iris",
            estimator="wrapper_exstracs_pruned",
            repeat=1,
            status="ok",
            skip_reason=None,
            error=None,
            f1_macro=0.82,
            fit_seconds=0.2,
            predict_seconds=0.01,
            n_rules=10,
            n_atoms=20,
            ruleset_json_bytes=123,
            n_train=100,
            n_test=50,
            validation_action=None,
            validation_message=None,
        ),
    ]

    aggregated = aggregate_benchmark_results(results)
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row.validation_warning_count == 1
    assert row.validation_warning_example == "example warning"


def test_report_outputs_include_warning_information():
    rows = [
        AggregatedBenchmarkResult(
            dataset="sklearn_iris",
            estimator="wrapper_exstracs_pruned",
            n_repeats=3,
            status="ok",
            f1_macro_mean=0.80,
            f1_macro_error=0.01,
            fit_seconds_mean=0.2,
            fit_seconds_error=0.01,
            predict_seconds_mean=0.01,
            predict_seconds_error=0.0,
            n_rules_mean=10,
            n_rules_error=0.0,
            n_atoms_mean=20,
            n_atoms_error=0.0,
            ruleset_json_bytes_mean=123,
            ruleset_json_bytes_error=0.0,
            validation_warning_count=2,
            validation_warning_example="example warning",
        )
    ]

    markdown = format_benchmark_report_markdown(rows, title="Warn Report")
    html = format_benchmark_report_html(rows, title="Warn Report")

    assert "## Warnings" in markdown
    assert "warning_runs" in markdown
    assert "⚠x2" in markdown
    assert "example warning" in markdown

    assert "<h2>Warnings</h2>" in html
    assert "warning_runs" in html
    assert "⚠x2" in html
    assert "example warning" in html


def test_no_split_disables_validation_fraction():
    """Runner must set validation_fraction=0.0 for no_split datasets."""
    from scoredrulesets.benchmarking.runner import _disable_validation_fraction
    from scoredrulesets.estimators.sklearn_wrapper import ScoredRuleSetClassifier

    # ScoredRuleSetClassifier wrapper – backend_params patched
    est = ScoredRuleSetClassifier(
        backend="rulelcs",
        backend_params={"validation_fraction": 0.25, "max_rules": 5},
    )
    _disable_validation_fraction(est)
    assert est.backend_params["validation_fraction"] == 0.0
    assert est.backend_params["max_rules"] == 5  # other params unchanged

    # Wrapper with backend that does NOT support validation_fraction
    est_cart = ScoredRuleSetClassifier(backend="cart", backend_params={"max_depth": 4})
    _disable_validation_fraction(est_cart)
    assert est_cart.backend_params == {"max_depth": 4}  # unchanged

    # Direct backend estimator (e.g. RuleLCSClassifier)
    from scoredrulesets import RuleLCSClassifier

    direct = RuleLCSClassifier(validation_fraction=0.25)
    _disable_validation_fraction(direct)
    assert direct.validation_fraction == 0.0

    # Wrapper with no backend_params initially
    est_no_bp = ScoredRuleSetClassifier(backend="rulegp")
    _disable_validation_fraction(est_no_bp)
    assert est_no_bp.backend_params["validation_fraction"] == 0.0
