"""Regenerate only the paper figures (pareto + rank2d) from the existing checkpoint.

Usage (from pypi-scoredrulesets root):
    python examples/benchmarks/replot_paper_figures.py \
        --checkpoint benchmarks/checkpoint_paper.jsonl \
        --output-dir articles/2026/ecta
"""
import argparse
from pathlib import Path

from scoredrulesets.benchmarking.runner import load_checkpoint_results
from scoredrulesets.benchmarking.plotting import plot_pareto_front, plot_2d_rank_plot


def main(checkpoint_path: str | Path, output_dir: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, done_keys = load_checkpoint_results(checkpoint_path)
    ok_results = [r for r in results if r.status == "ok"]
    print(f"Loaded {len(results)} results ({len(ok_results)} ok) from {checkpoint_path}")

    pareto_png, pareto_pdf = plot_pareto_front(
        ok_results, output_base=output_dir / "benchmark_results_pareto"
    )
    print(f"Pareto front: {pareto_pdf}")

    rank2d_png, rank2d_pdf = plot_2d_rank_plot(
        ok_results, output_base=output_dir / "benchmark_results_rank2d"
    )
    print(f"Rank2D:       {rank2d_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate paper figures from checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="benchmarks/checkpoint_paper.jsonl",
        help="Path to the checkpoint JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="articles/2026/ecta",
        help="Output directory for the regenerated figures.",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir)
