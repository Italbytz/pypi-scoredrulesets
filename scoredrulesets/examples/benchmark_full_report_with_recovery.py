from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import benchmark_full_report as full_report
import recover_heatmap_from_log as recover_log


def run_recovery_for_log(log_file: Path, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = recover_log.default_output_dir_for_log(log_file)
    recover_log.main(["--log-file", str(log_file), "--output-dir", str(output_dir)])
    return output_dir


def run_full_report_with_recovery(
    *,
    log_file: Path | None = None,
    recovery_output_dir: Path | None = None,
) -> int:
    resolved_log = log_file or full_report._default_log_path()

    with full_report._maybe_tee_to_file(resolved_log):
        try:
            full_report.main(log_file=resolved_log)
            return 0
        except Exception as exc:
            print("\n[RECOVERY] benchmark run failed; attempting heatmap recovery from current log...", flush=True)
            traceback.print_exc()
            try:
                recovered_dir = run_recovery_for_log(resolved_log, recovery_output_dir)
                print(f"[RECOVERY] recovered artifacts written to: {recovered_dir}", flush=True)
            except Exception:
                print("[RECOVERY] recovery also failed.", flush=True)
                traceback.print_exc()
            return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full benchmark report and automatically recover heatmaps from the log if the run fails."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Console log file path. Default: benchmarks/logs/benchmark_full_report_<timestamp>.log",
    )
    parser.add_argument(
        "--recovery-output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for recovered artifacts.",
    )
    args = parser.parse_args(argv)
    return run_full_report_with_recovery(
        log_file=args.log_file,
        recovery_output_dir=args.recovery_output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())

