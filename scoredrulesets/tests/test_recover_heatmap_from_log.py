from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "recover_heatmap_from_log.py"
    spec = importlib.util.spec_from_file_location("recover_heatmap_from_log", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_latest_log_file_picks_newest_mtime(tmp_path: Path):
    mod = _load_module()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    older = log_dir / "benchmark_full_report_older.log"
    newer = log_dir / "benchmark_full_report_newer.log"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")

    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))

    latest = mod.find_latest_log_file(log_dir)
    assert latest == newer



def test_resolve_log_input_prefers_explicit_file(tmp_path: Path):
    mod = _load_module()
    log_file = tmp_path / "run.log"
    log_file.write_text("x", encoding="utf-8")

    resolved = mod.resolve_log_input(
        log_file=log_file,
        latest_log=False,
        log_dir=tmp_path,
        pattern="*.log",
    )
    assert resolved == log_file



def test_default_output_dir_for_log_uses_stem():
    mod = _load_module()
    out = mod.default_output_dir_for_log(Path("benchmarks/logs/benchmark_full_report_20260322-123456.log"))
    assert out == Path("benchmarks/recovered-from-log/benchmark_full_report_20260322-123456")


def test_main_writes_reports_for_recovered_results(monkeypatch, tmp_path: Path):
    mod = _load_module()
    log_file = tmp_path / "run.log"
    output_dir = tmp_path / "recovered"
    log_file.write_text(
        "\n".join(
            [
                "[progress 1/2] START dataset=iris repeat=1/1 estimator=DemoEstimator train=100 test=50",
                "[BENCHMARK] DemoEstimator | iris | F1=0.7500 | Regeln=3 | Atome=5 | fit=1.2s",
                "[progress 2/2] START dataset=wine repeat=1/1 estimator=DemoEstimator train=120 test=58",
                "[BENCHMARK] DemoEstimator | wine | F1=0.8100 | Regeln=4 | Atome=7 | fit=0.8s",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_plot(results, output_base):
        png = output_base.with_suffix(".png")
        pdf = output_base.with_suffix(".pdf")
        png.write_text("png", encoding="utf-8")
        pdf.write_text("pdf", encoding="utf-8")
        return png, pdf

    monkeypatch.setattr(mod, "plot_benchmark_heatmap", _fake_plot)
    monkeypatch.setattr(mod, "plot_benchmark_heatmap_combined", _fake_plot)

    mod.main(["--log-file", str(log_file), "--output-dir", str(output_dir)])

    markdown_report = output_dir / "benchmark_leaderboard.md"
    html_report = output_dir / "benchmark_leaderboard.html"
    assert markdown_report.exists()
    assert html_report.exists()
    assert "Recovered ScoredRuleSets Benchmark Report" in markdown_report.read_text(encoding="utf-8")
    assert "DemoEstimator" in markdown_report.read_text(encoding="utf-8")
    assert "benchmark_results_recovered.csv" in html_report.read_text(encoding="utf-8")


