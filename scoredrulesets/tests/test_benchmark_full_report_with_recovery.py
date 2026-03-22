from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path


def _load_module():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    module_path = examples_dir / "benchmark_full_report_with_recovery.py"
    spec = importlib.util.spec_from_file_location("benchmark_full_report_with_recovery", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_success_path_returns_zero(monkeypatch, tmp_path: Path):
    mod = _load_module()
    log_file = tmp_path / "run.log"

    monkeypatch.setattr(mod.full_report, "_default_log_path", lambda: log_file)
    monkeypatch.setattr(mod.full_report, "_maybe_tee_to_file", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(mod.full_report, "main", lambda log_file=None: None)

    called = {"recovery": False}

    def _fake_recovery(*args, **kwargs):
        called["recovery"] = True
        return tmp_path / "recovered"

    monkeypatch.setattr(mod, "run_recovery_for_log", _fake_recovery)

    rc = mod.run_full_report_with_recovery()
    assert rc == 0
    assert called["recovery"] is False



def test_failure_path_triggers_recovery(monkeypatch, tmp_path: Path):
    mod = _load_module()
    log_file = tmp_path / "run.log"
    log_file.write_text("partial log", encoding="utf-8")

    monkeypatch.setattr(mod.full_report, "_default_log_path", lambda: log_file)
    monkeypatch.setattr(mod.full_report, "_maybe_tee_to_file", lambda path: contextlib.nullcontext())

    def _boom(log_file=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.full_report, "main", _boom)

    called: dict[str, object] = {"recovery": None}

    def _fake_recovery(path, output_dir=None):
        called["recovery"] = (path, output_dir)
        return tmp_path / "recovered"

    monkeypatch.setattr(mod, "run_recovery_for_log", _fake_recovery)

    rc = mod.run_full_report_with_recovery()
    assert rc == 1
    assert called["recovery"] == (log_file, None)


