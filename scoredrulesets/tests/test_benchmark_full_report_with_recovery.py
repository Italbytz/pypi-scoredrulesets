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
    monkeypatch.setattr(mod.full_report, "main", lambda log_file=None, **kw: None)

    rc = mod.run_with_resume(
        max_attempts=1,
        log_file=log_file,
        checkpoint_path=str(tmp_path / "ckpt.jsonl"),
    )
    assert rc == 0


def test_failure_path_retries_and_returns_two(monkeypatch, tmp_path: Path):
    mod = _load_module()
    log_file = tmp_path / "run.log"

    monkeypatch.setattr(mod.full_report, "_default_log_path", lambda: log_file)
    monkeypatch.setattr(mod.full_report, "_maybe_tee_to_file", lambda path: contextlib.nullcontext())

    call_count = {"n": 0}

    def _boom(log_file=None, **kw):
        call_count["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.full_report, "main", _boom)
    # Disable recovery so it doesn't interfere
    monkeypatch.setattr(mod, "_HAS_RECOVER", False)

    rc = mod.run_with_resume(
        max_attempts=2,
        log_file=log_file,
        checkpoint_path=str(tmp_path / "ckpt.jsonl"),
    )
    assert rc == 2
    assert call_count["n"] == 2  # sollte 2x versucht haben
