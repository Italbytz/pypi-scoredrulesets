import importlib.util
from pathlib import Path

import numpy as np

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
    plot_benchmark_results,
    run_benchmarks,
)
from scoredrulesets.benchmarking.datasets import DatasetBundle
from scoredrulesets.benchmarking.runner import _resolve_test_size


def test_benchmarking_runs_for_core_estimators():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "native", "gp"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    assert len(results) == 3
    assert all(r.dataset == "sklearn_iris" for r in results)
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
    config = BenchmarkConfig(dataset_names=["does_not_exist"], estimator_names=["native"])
    try:
        run_benchmarks(config)
    except ValueError as exc:
        assert "Unknown dataset names" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown dataset")


def test_benchmark_plot_is_written_as_png_and_pdf(tmp_path: Path):
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "native", "gp"],
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
        estimator_names=["wrapper_cart", "native"],
        repeats=2,
        random_state=0,
    )
    results = run_benchmarks(config)
    aggregated = aggregate_benchmark_results(results, error_bar="std")

    assert len(aggregated) == 2
    assert all(item.dataset == "sklearn_iris" for item in aggregated)
    assert all(item.n_repeats == 2 for item in aggregated)
    assert all(item.f1_macro_mean is not None for item in aggregated)
    assert all(item.f1_macro_error is not None for item in aggregated)


def test_benchmarking_smoke_compare_pittsburgh_native_gp():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_pittsburgh_fast", "native", "gp"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    assert len(results) == 3
    by_estimator = {result.estimator: result for result in results}
    assert set(by_estimator) == {"wrapper_pittsburgh_fast", "native", "gp"}
    assert all(result.status == "ok" for result in results)
    assert all(result.f1_macro is not None for result in results)
    assert all(result.n_rules is not None and result.n_rules > 0 for result in results)


def test_benchmarking_smoke_compare_pittsburgh_profiles():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_pittsburgh_fast", "wrapper_pittsburgh_strong", "wrapper_pittsburgh_diverse"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    assert len(results) == 3
    by_estimator = {result.estimator: result for result in results}
    assert set(by_estimator) == {
        "wrapper_pittsburgh_fast",
        "wrapper_pittsburgh_strong",
        "wrapper_pittsburgh_diverse",
    }
    assert all(result.status == "ok" for result in results)
    assert all(result.f1_macro is not None for result in results)
    assert all(result.n_rules is not None and result.n_rules > 0 for result in results)


def test_benchmarking_smoke_compare_michigan_native_gp():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_michigan_compact", "native", "gp"],
        repeats=1,
        random_state=0,
    )
    results = run_benchmarks(config)

    assert len(results) == 3
    by_estimator = {result.estimator: result for result in results}
    assert set(by_estimator) == {"wrapper_michigan_compact", "native", "gp"}
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
        estimator_names=["wrapper_cart", "native"],
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
        estimator_names=["wrapper_cart", "native"],
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
        estimator_names=["wrapper_cart", "native"],
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


def test_benchmark_leaderboard_sorting_and_markdown_output():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "native", "gp"],
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
        estimator_names=["wrapper_cart", "native"],
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
    assert "### sklearn_iris" in report
    assert "## Leaderboard" in report
    assert "## Dataset: sklearn_iris" in report
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
        estimator_names=["wrapper_cart", "native"],
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
    assert "<h2>Dataset: sklearn_iris</h2>" in html
    assert "<table>" in html


def test_benchmark_leaderboard_html_contains_table():
    config = BenchmarkConfig(
        dataset_names=["sklearn_iris"],
        estimator_names=["wrapper_cart", "native"],
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


