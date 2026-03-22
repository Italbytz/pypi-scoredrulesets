from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import asdict
from pathlib import Path

from scoredrulesets.benchmarking import (
    aggregate_benchmark_results,
    aggregated_results_as_dicts,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover benchmark heatmaps from partial console logs.")
    parser.add_argument("--log-file", type=Path, default=None, help="Path to text log file. If omitted, reads stdin.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/recovered-from-log"))
    args = parser.parse_args()

    if args.log_file is not None:
        text = args.log_file.read_text(encoding="utf-8")
    else:
        import sys

        text = sys.stdin.read()

    results = parse_benchmark_log(text)
    if not results:
        raise SystemExit("No [BENCHMARK] lines found in input log.")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    payload = [asdict(r) for r in results]
    (out / "benchmark_results_recovered.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(payload, out / "benchmark_results_recovered.csv")

    aggregated = aggregate_benchmark_results(results, error_bar="std")
    aggregated_payload = aggregated_results_as_dicts(aggregated)
    (out / "benchmark_results_recovered_aggregated.json").write_text(
        json.dumps(aggregated_payload, indent=2), encoding="utf-8"
    )
    _write_csv(aggregated_payload, out / "benchmark_results_recovered_aggregated.csv")

    heatmap_png, heatmap_pdf = plot_benchmark_heatmap(results, output_base=out / "benchmark_results_recovered_heatmap")
    combined_png, combined_pdf = plot_benchmark_heatmap_combined(
        results, output_base=out / "benchmark_results_recovered_heatmap_combined"
    )

    print(f"Recovered runs: {len(results)}")
    print(f"Datasets: {len({r.dataset for r in results})}")
    print(f"Estimators: {len({r.estimator for r in results})}")
    print(f"Heatmap: {heatmap_png}")
    print(f"Heatmap PDF: {heatmap_pdf}")
    print(f"Combined heatmap: {combined_png}")
    print(f"Combined heatmap PDF: {combined_pdf}")


if __name__ == "__main__":
    main()

