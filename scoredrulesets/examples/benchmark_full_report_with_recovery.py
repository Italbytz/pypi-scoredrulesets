"""Benchmark-Wrapper mit automatischem Resume und Recovery.

Ruft ``benchmark_full_report.main()`` auf.  Bei Absturz oder Ctrl-C:
  1. Checkpoint ist bereits aktuell (jeder Lauf wird sofort geschrieben).
  2. Beim naechsten Aufruf setzt der Benchmark automatisch an der
     abgebrochenen Stelle fort.
  3. Falls trotzdem Reports fehlen, wird die Log-basierte Recovery versucht.

Aufruf:
    python examples/benchmark_full_report_with_recovery.py
    python examples/benchmark_full_report_with_recovery.py --max-attempts 5
    python examples/benchmark_full_report_with_recovery.py --checkpoint benchmarks/my_run.jsonl
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import benchmark_full_report as full_report

# Optionale Log-Recovery (falls vorhanden)
try:
    import recover_heatmap_from_log as recover_log

    _HAS_RECOVER = True
except ImportError:
    _HAS_RECOVER = False


def run_with_resume(
    *,
    max_attempts: int = 3,
    log_file: Path | None = None,
    checkpoint_path: str = "benchmarks/checkpoint.jsonl",
    extra_kwargs: dict | None = None,
) -> int:
    """Fuehrt den Benchmark mit automatischem Resume aus.

    Bei jedem Versuch werden nur noch fehlende Laeufe ausgefuehrt
    (dank Checkpoint).  Nach ``max_attempts`` Versuchen wird aufgehoert.

    Returns 0 bei Erfolg, 1 bei Teil-Erfolg, 2 bei komplettem Fehlschlag.
    """
    kwargs = dict(
        checkpoint_path=checkpoint_path,
        **(extra_kwargs or {}),
    )

    for attempt in range(1, max_attempts + 1):
        resolved_log = log_file or full_report._default_log_path()
        print(
            f"\n{'=' * 70}\n"
            f"[RESUME] Versuch {attempt}/{max_attempts}  "
            f"(Checkpoint: {checkpoint_path})\n"
            f"{'=' * 70}",
            flush=True,
        )

        with full_report._maybe_tee_to_file(resolved_log):
            try:
                full_report.main(log_file=resolved_log, **kwargs)
                print(f"\n[RESUME] Benchmark erfolgreich abgeschlossen (Versuch {attempt}).", flush=True)
                return 0
            except KeyboardInterrupt:
                print(
                    f"\n[RESUME] Abbruch durch Benutzer (Ctrl-C) in Versuch {attempt}. "
                    f"Checkpoint ist gespeichert – naechster Aufruf setzt fort.",
                    flush=True,
                )
                return 1
            except Exception as exc:
                print(
                    f"\n[RESUME] Fehler in Versuch {attempt}: {exc}",
                    flush=True,
                )
                traceback.print_exc()

                # Log-basierte Recovery versuchen
                if _HAS_RECOVER:
                    try:
                        out_dir = recover_log.default_output_dir_for_log(resolved_log)
                        recover_log.main(["--log-file", str(resolved_log), "--output-dir", str(out_dir)])
                        print(f"[RECOVERY] Artefakte aus Log wiederhergestellt: {out_dir}", flush=True)
                    except Exception:
                        pass

                if attempt < max_attempts:
                    print(
                        f"[RESUME] Starte naechsten Versuch ({attempt + 1}/{max_attempts})...",
                        flush=True,
                    )
                else:
                    print(
                        f"[RESUME] Alle {max_attempts} Versuche erschoepft. "
                        f"Checkpoint liegt vor: {checkpoint_path}",
                        flush=True,
                    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark mit automatischem Resume bei Abbruch/Crash.",
        epilog="Der Checkpoint wird nach jedem Einzellauf geschrieben. "
               "Wiederholter Aufruf setzt an der abgebrochenen Stelle fort.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximale Anzahl Wiederholungsversuche (default: 3).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Console log file path. Default: auto-generated.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="benchmarks/checkpoint.jsonl",
        help="JSONL-Checkpoint-Datei (default: benchmarks/checkpoint.jsonl).",
    )
    parser.add_argument(
        "--recovery-output-dir",
        type=Path,
        default=None,
        help="Output-Verzeichnis fuer Recovery-Artefakte.",
    )
    # Alle full_report-Argumente weiterreichen
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--estimators", type=str, default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--include-pmlb", action="store_true")
    args = parser.parse_args(argv)

    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    est_names = [x.strip() for x in args.estimators.split(",") if x.strip()] or None
    timeout = args.timeout if args.timeout > 0 else None

    extra_kwargs = dict(
        dataset_names=ds_names,
        estimator_names=est_names,
        repeats=args.repeats,
        timeout_seconds=timeout,
        skip_synthetic=args.skip_synthetic,
        include_pmlb=args.include_pmlb,
    )

    return run_with_resume(
        max_attempts=args.max_attempts,
        log_file=args.log_file,
        checkpoint_path=args.checkpoint,
        extra_kwargs=extra_kwargs,
    )


if __name__ == "__main__":
    raise SystemExit(main())

