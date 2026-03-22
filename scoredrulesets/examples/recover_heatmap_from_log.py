from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from scoredrulesets.benchmarking import (
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
    build_benchmark_leaderboard,
    format_benchmark_report_html,
    format_benchmark_report_markdown,
    plot_benchmark_heatmap,
    plot_benchmark_heatmap_combined,
)
from scoredrulesets.benchmarking.runner import BenchmarkResult

START_RE = re.compile(
    r"^\[progress\s+\d+/\d+\]\s+START\s+dataset=(?P<dataset>\S+)\s+repeat=(?P<repeat>\d+)/\d+\s+estimator=(?P<estimator>\S+)\s+train=(?P<train>\d+)\s+test=(?P<test>\d+)"
)
BENCH_RE = re.compile(
    r"^\[BENCHMARK\]\s+(?P<estimator>.*?)\s+\|\s+(?P<dataset>.*?)\s+\|\s+F1=(?P<f1>[0-9.]+)(?:\s+\(nativ=(?P<f1_native>[0-9.]+)\))?\s+\|\s+Regeln=(?P<rules>\d+)\s+\|\s+Atome=(?P<atoms>\d+)\s+\|\s+fit=(?P<fit>.+?)\s*$"
)


def _parse_duration_to_seconds(text: str) -> float:
    text = text.strip()
    if "m" in text and "s" in text:
        m = re.search(r"([0-9]+)m", text)
        s = re.search(r"([0-9]+(?:\.[0-9]+)?)s", text)
        minutes = int(m.group(1)) if m else 0
        seconds = float(s.group(1)) if s else 0.0
        return minutes * 60.0 + seconds
    s = re.search(r"([0-9]+(?:\.[0-9]+)?)s", text)
    if s:
        return float(s.group(1))
    return 0.0


def parse_benchmark_log(text: str) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    start_info: dict[tuple[str, str], tuple[int, int, int]] = {}
    fallback_repeat_counter: dict[tuple[str, str], int] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        start_match = START_RE.match(line)
        if start_match:
            ds = start_match.group("dataset")
            est = start_match.group("estimator")
            rep = int(start_match.group("repeat")) - 1
            n_train = int(start_match.group("train"))
            n_test = int(start_match.group("test"))
            start_info[(ds, est)] = (rep, n_train, n_test)
            continue

        bench_match = BENCH_RE.match(line)
        if not bench_match:
            continue

        ds = bench_match.group("dataset")
        est = bench_match.group("estimator")
        f1 = float(bench_match.group("f1"))
        fit_seconds = _parse_duration_to_seconds(bench_match.group("fit"))
        n_rules = int(bench_match.group("rules"))
        n_atoms = int(bench_match.group("atoms"))

        key = (ds, est)
        if key in start_info:
            rep, n_train, n_test = start_info[key]
        else:
            rep = fallback_repeat_counter.get(key, 0)
            fallback_repeat_counter[key] = rep + 1
            n_train = None
            n_test = None

        result = BenchmarkResult(
            dataset=ds,
            estimator=est,
            repeat=rep,
            status="ok",
            skip_reason=None,
            error=None,
            f1_macro=f1,
            fit_seconds=fit_seconds,
            predict_seconds=0.0,
            n_rules=n_rules,
            n_atoms=n_atoms,
            ruleset_json_bytes=None,
            n_train=n_train,
            n_test=n_test,
            validation_action=None,
            validation_message=None,
        )
        results.append(result)

    return results


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def find_latest_log_file(
    log_dir: Path,
    pattern: str = "benchmark_full_report_*.log",
) -> Path:
    candidates = [path for path in log_dir.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No log files matching '{pattern}' found in '{log_dir}'."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_log_input(
    *,
    log_file: Path | None,
    latest_log: bool,
    log_dir: Path,
    pattern: str,
) -> Path | None:
    if log_file is not None and latest_log:
        raise ValueError("Use either --log-file or --latest-log, not both.")
    if log_file is not None:
        return log_file
    if latest_log:
        return find_latest_log_file(log_dir=log_dir, pattern=pattern)
    return None


def default_output_dir_for_log(log_path: Path) -> Path:
    stem = log_path.stem
    return Path("benchmarks") / "recovered-from-log" / stem


def _write_recovery_reports(
    *,
    results: list[BenchmarkResult],
    leaderboard,
    output_dir: Path,
    artifact_paths: dict[str, str],
) -> tuple[Path, Path]:
    dataset_names = sorted({result.dataset for result in results})
    estimator_names = sorted({result.estimator for result in results})
    notes = [
        "This report was reconstructed from a partial benchmark console log.",
        "Missing runs are not recreated; summaries reflect only recovered [BENCHMARK] entries.",
    ]
    config = {
        "recovered_runs": len(results),
        "datasets": ",".join(dataset_names),
        "estimators": ",".join(estimator_names),
        "source": "console_log_recovery",
    }

    markdown_path = output_dir / "benchmark_leaderboard.md"
    html_path = output_dir / "benchmark_leaderboard.html"
    markdown_path.write_text(
        format_benchmark_report_markdown(
            leaderboard,
            title="Recovered ScoredRuleSets Benchmark Report",
            config=config,
            artifact_paths=artifact_paths,
            notes=notes,
        ),
        encoding="utf-8",
    )
    html_path.write_text(
        format_benchmark_report_html(
            leaderboard,
            title="Recovered ScoredRuleSets Benchmark Report",
            config=config,
            artifact_paths=artifact_paths,
            notes=notes,
        ),
        encoding="utf-8",
    )
    return markdown_path, html_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Recover benchmark heatmaps from partial console logs.")
    parser.add_argument("--log-file", type=Path, default=None, help="Path to text log file. If omitted, reads stdin.")
    parser.add_argument("--latest-log", action="store_true", help="Use the newest benchmark_full_report_*.log from --log-dir.")
    parser.add_argument("--log-dir", type=Path, default=Path("benchmarks/logs"), help="Directory searched by --latest-log.")
    parser.add_argument("--log-pattern", type=str, default="benchmark_full_report_*.log", help="Glob pattern used with --latest-log.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for recovered artifacts. Default: benchmarks/recovered-from-log/<log-stem> for file-based input, otherwise benchmarks/recovered-from-log")
    args = parser.parse_args(argv)

    resolved_log_file = resolve_log_input(
        log_file=args.log_file,
        latest_log=bool(args.latest_log),
        log_dir=args.log_dir,
        pattern=args.log_pattern,
    )

    if resolved_log_file is not None:
        text = resolved_log_file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    results = parse_benchmark_log(text)
    if not results:
        raise SystemExit("No [BENCHMARK] lines found in input log.")

    if args.output_dir is not None:
        out = args.output_dir
    elif resolved_log_file is not None:
        out = default_output_dir_for_log(resolved_log_file)
    else:
        out = Path("benchmarks/recovered-from-log")
    out.mkdir(parents=True, exist_ok=True)

    payload = [asdict(r) for r in results]
    (out / "benchmark_results_recovered.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(payload, out / "benchmark_results_recovered.csv")

    aggregated = aggregate_benchmark_results(results, error_bar="std")
    aggregated_payload = aggregated_results_as_dicts(aggregated)
    leaderboard = build_benchmark_leaderboard(aggregated)
    (out / "benchmark_results_recovered_aggregated.json").write_text(
        json.dumps(aggregated_payload, indent=2), encoding="utf-8"
    )
    _write_csv(aggregated_payload, out / "benchmark_results_recovered_aggregated.csv")

    heatmap_png, heatmap_pdf = plot_benchmark_heatmap(results, output_base=out / "benchmark_results_recovered_heatmap")
    combined_png, combined_pdf = plot_benchmark_heatmap_combined(
        results, output_base=out / "benchmark_results_recovered_heatmap_combined"
    )
    report_md, report_html = _write_recovery_reports(
        results=results,
        leaderboard=leaderboard,
        output_dir=out,
        artifact_paths={
            "raw_csv": "benchmark_results_recovered.csv",
            "raw_json": "benchmark_results_recovered.json",
            "aggregated_csv": "benchmark_results_recovered_aggregated.csv",
            "aggregated_json": "benchmark_results_recovered_aggregated.json",
            "heatmap_png": heatmap_png.name,
            "heatmap_pdf": heatmap_pdf.name,
            "combined_heatmap_png": combined_png.name,
            "combined_heatmap_pdf": combined_pdf.name,
        },
    )

    print(f"Recovered runs: {len(results)}")
    print(f"Datasets: {len({r.dataset for r in results})}")
    print(f"Estimators: {len({r.estimator for r in results})}")
    if resolved_log_file is not None:
        print(f"Recovered from log: {resolved_log_file}")
    print(f"Heatmap: {heatmap_png}")
    print(f"Heatmap PDF: {heatmap_pdf}")
    print(f"Combined heatmap: {combined_png}")
    print(f"Combined heatmap PDF: {combined_pdf}")
    print(f"Report (Markdown): {report_md}")
    print(f"Report (HTML): {report_html}")


if __name__ == "__main__":
    main()


