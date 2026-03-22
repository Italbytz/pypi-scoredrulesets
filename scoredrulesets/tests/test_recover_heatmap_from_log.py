from __future__ import annotations

import importlib.util
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

    older.touch()
    newer.touch()

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

