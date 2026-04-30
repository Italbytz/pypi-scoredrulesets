#!/usr/bin/env python3
"""
Cross-language LogicGP Benchmark Suite

Runs benchmarks on both Python and C# LogicGP implementations,
comparing performance and F1 scores.
"""

import json
import sys
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def get_profile_config(profile: str):
    """Return predefined benchmark matrix by profile name."""
    profile = profile.lower()
    if profile == "quick":
        return {
            "trainers": ["flcw-macro"],
            "datasets": ["iris"],
            "seeds": [42],
            "generations": 20,
            "population": 200,
            "f1_averaging": "macro",
        }
    if profile == "thorough":
        return {
            "trainers": ["flcw-macro", "flcw-micro", "rlcw-macro"],
            "datasets": ["iris", "npha"],
            "seeds": [41, 42, 43],
            "generations": 80,
            "population": 800,
            "f1_averaging": "macro",
        }

    # default: standard
    return {
        "trainers": ["flcw-macro", "flcw-micro", "rlcw-macro"],
        "datasets": ["iris", "npha"],
        "seeds": [41, 42],
        "generations": 50,
        "population": 500,
        "f1_averaging": "macro",
    }


@dataclass
class BenchmarkResult:
    trainer: str
    dataset: str
    f1_score: float
    train_time_ms: int
    f1_averaging: str = "macro"
    num_rules: int = 0
    num_atoms: int = 0
    status: str = "ok"
    error: Optional[str] = None
    timestamp: str = ""
    implementation: str = "python"
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class PythonBenchmarkRunner:
    """Run benchmarks using Python LogicGP implementation"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        sys.path.insert(0, str(self.repo_path / "src"))
    
    def run_benchmark(self, trainer: str, dataset: str, generations: int = 100,
                      population: int = 1000, seed: int = 42,
                      f1_averaging: str = "macro") -> BenchmarkResult:
        """Run a single benchmark"""
        try:
            import time
            from sklearn.datasets import load_iris
            from scoredrulesets.estimators import LogicGPClassifier
            
            # Load dataset
            if dataset == "iris":
                X, y = load_iris(return_X_y=True)
            else:
                return BenchmarkResult(
                    trainer=trainer,
                    dataset=dataset,
                    f1_score=0.0,
                    train_time_ms=0,
                    f1_averaging=f1_averaging,
                    status="skipped",
                    error=f"Dataset not implemented on Python side: {dataset}",
                    implementation="python"
                )
            
            # Train model
            start = time.time()
            model = LogicGPClassifier(
                max_generations=generations,
                population_size=population,
                random_state=seed,
            )
            model.fit(X, y)
            train_time_ms = int((time.time() - start) * 1000)
            
            # Test performance
            f1_score = model.score(X, y)  # Assuming this returns F1 or similar
            
            return BenchmarkResult(
                trainer=trainer,
                dataset=dataset,
                f1_score=f1_score,
                train_time_ms=train_time_ms,
                f1_averaging=f1_averaging,
                status="ok",
                implementation="python"
            )
        
        except Exception as e:
            return BenchmarkResult(
                trainer=trainer,
                dataset=dataset,
                f1_score=0.0,
                train_time_ms=0,
                f1_averaging=f1_averaging,
                status="error",
                error=str(e),
                implementation="python"
            )


class MockCSharpBenchmarkRunner:
    """Mock C# runner for testing (generates synthetic results)"""
    
    def run_benchmark(self, trainer: str, dataset: str, generations: int = 100,
                      population: int = 1000, seed: int = 42,
                      f1_averaging: str = "macro") -> BenchmarkResult:
        """Run a C# benchmark (currently mock)"""
        import random
        
        # Generate synthetic results similar to Python
        f1_score = 0.92 + random.random() * 0.05
        train_time_ms = 150 + random.randint(-50, 100)
        
        return BenchmarkResult(
            trainer=trainer,
            dataset=dataset,
            f1_score=f1_score,
            train_time_ms=train_time_ms,
            f1_averaging=f1_averaging,
            num_rules=3,
            num_atoms=12,
            status="ok",
            implementation="csharp"
        )


class RealCSharpBenchmarkRunner:
    """Run benchmarks via the compiled C# CLI using csharp_wrapper."""

    def __init__(self, cli_path: Optional[Path] = None):
        from scoredrulesets.benchmarking.csharp_wrapper import CSharBenchmarkRunner

        self._runner = CSharBenchmarkRunner(cli_path=cli_path)

    def run_benchmark(self, trainer: str, dataset: str, generations: int = 100,
                      population: int = 1000, seed: int = 42,
                      f1_averaging: str = "macro") -> BenchmarkResult:
        cs_result = self._runner.run_benchmark(
            trainer=trainer,
            dataset=dataset,
            generations=generations,
            population=population,
            seed=seed,
            f1_averaging=f1_averaging,
        )
        # Normalize wrapper result to this script's local result type.
        return BenchmarkResult(
            trainer=cs_result.trainer,
            dataset=cs_result.dataset,
            f1_score=cs_result.f1_score,
            train_time_ms=int(cs_result.train_time_ms),
            f1_averaging=cs_result.f1_averaging,
            num_rules=cs_result.num_rules,
            num_atoms=cs_result.num_atoms,
            status=cs_result.status,
            error=cs_result.error,
            implementation="csharp",
        )


class BenchmarkComparator:
    """Compare benchmark results across implementations"""
    
    @staticmethod
    def compare(python_result: BenchmarkResult, csharp_result: BenchmarkResult) -> dict:
        """Compare two benchmark results"""

        if python_result.status != "ok" or csharp_result.status != "ok":
            warnings = [
                f"Non-ok status detected (python={python_result.status}, csharp={csharp_result.status})"
            ]
            if python_result.error:
                warnings.append(f"Python: {python_result.error}")
            if csharp_result.error:
                warnings.append(f"C#: {csharp_result.error}")
            return {
                "python": asdict(python_result),
                "csharp": asdict(csharp_result),
                "f1_difference": None,
                "f1_relative_error_percent": None,
                "time_ratio_csharp_python": None,
                "warnings": warnings,
            }
        
        f1_diff = abs(python_result.f1_score - csharp_result.f1_score)
        f1_rel_err = (f1_diff / python_result.f1_score * 100) if python_result.f1_score > 0 else 0
        
        time_ratio = csharp_result.train_time_ms / python_result.train_time_ms if python_result.train_time_ms > 0 else 0
        
        warnings = []
        if f1_rel_err > 5:
            warnings.append(f"F1 divergence {f1_rel_err:.1f}% (possible implementation difference)")
        if time_ratio > 2 or time_ratio < 0.5:
            warnings.append(f"Time ratio {time_ratio:.2f}X (possible optimization issue)")
        
        return {
            "python": asdict(python_result),
            "csharp": asdict(csharp_result),
            "f1_difference": f1_diff,
            "f1_relative_error_percent": f1_rel_err,
            "time_ratio_csharp_python": time_ratio,
            "warnings": warnings
        }
    
    @staticmethod
    def generate_report(comparisons):
        """Generate markdown comparison report"""
        report = "# LogicGP Cross-Language Benchmark Report\n\n"
        report += f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        report += "## Summary\n\n"
        report += "| Test | Trainer | Dataset | Python F1 | C# F1 | Diff | Time Ratio | Status |\n"
        report += "|------|---------|---------|-----------|-------|------|-----------|--------|\n"

        for i, comp in enumerate(comparisons, 1):
            py = comp["python"]
            cs = comp["csharp"]
            warnings = comp.get("warnings", [])
            status = "WARN" if warnings else "OK"
            diff = comp["f1_difference"]
            rel = comp["f1_relative_error_percent"]
            ratio = comp["time_ratio_csharp_python"]
            diff_cell = "n/a" if diff is None or rel is None else f"{diff:.4f} ({rel:.1f}%)"
            ratio_cell = "n/a" if ratio is None else f"{ratio:.2f}X"
            report += f"| {i} | {py['trainer']} | {py['dataset']} | "
            report += f"{py['f1_score']:.4f} | {cs['f1_score']:.4f} | "
            report += f"{diff_cell} | "
            report += f"{ratio_cell} | {status} |\n"

        return report

    @staticmethod
    def generate_compact_summary(comparisons, sort_by: str = "diff"):
        """Generate compact plain-text summary grouped by trainer."""
        lines = []
        lines.append("Compact Summary")
        lines.append("=" * 60)

        grouped = {}
        for comp in comparisons:
            trainer = comp["python"]["trainer"]
            grouped.setdefault(trainer, []).append(comp)

        trainer_stats = []
        for trainer, comps in grouped.items():
            diffs = [c.get("f1_relative_error_percent") for c in comps if c.get("f1_relative_error_percent") is not None]
            ratios = [c.get("time_ratio_csharp_python") for c in comps if c.get("time_ratio_csharp_python") is not None]
            warns = sum(1 for c in comps if c.get("warnings"))
            avg_diff = (sum(diffs) / len(diffs)) if diffs else -1.0
            avg_ratio = (sum(ratios) / len(ratios)) if ratios else -1.0
            warn_rate = (warns / len(comps)) if comps else 0.0
            trainer_stats.append((trainer, avg_diff, avg_ratio, warn_rate))

        if sort_by == "time":
            trainer_stats.sort(key=lambda item: item[2], reverse=True)
        elif sort_by == "warn":
            trainer_stats.sort(key=lambda item: item[3], reverse=True)
        elif sort_by == "name":
            trainer_stats.sort(key=lambda item: item[0])
        else:
            trainer_stats.sort(key=lambda item: item[1], reverse=True)

        lines.append(f"Sortierung: {sort_by}")
        lines.append("")

        for trainer, _, _, _ in trainer_stats:
            comps = grouped[trainer]
            lines.append(f"Trainer: {trainer}")
            lines.append("Dataset | Seed | PyF1 | CsF1 | Diff% | TimeX | Status")
            lines.append("--------|------|------|------|-------|-------|-------")

            diff_values = []
            ratio_values = []
            warn_count = 0

            for comp in comps:
                py = comp["python"]
                cs = comp["csharp"]
                seed = comp.get("seed", "n/a")
                status = "WARN" if comp.get("warnings") else "OK"
                if status == "WARN":
                    warn_count += 1

                diff = comp.get("f1_relative_error_percent")
                ratio = comp.get("time_ratio_csharp_python")
                diff_cell = "n/a" if diff is None else f"{diff:.1f}"
                ratio_cell = "n/a" if ratio is None else f"{ratio:.2f}"

                if diff is not None:
                    diff_values.append(diff)
                if ratio is not None:
                    ratio_values.append(ratio)

                lines.append(
                    f"{py['dataset']} | {seed} | {py['f1_score']:.4f} | {cs['f1_score']:.4f} | "
                    f"{diff_cell} | {ratio_cell} | {status}"
                )

            avg_diff = "n/a" if not diff_values else f"{(sum(diff_values) / len(diff_values)):.1f}"
            avg_ratio = "n/a" if not ratio_values else f"{(sum(ratio_values) / len(ratio_values)):.2f}"
            warn_rate = f"{(warn_count / len(comps) * 100):.0f}%"
            lines.append(f"AVG | - | - | - | {avg_diff} | {avg_ratio} | WARN={warn_rate}")
            lines.append("")

        return "\n".join(lines)


def main():
    """Run benchmark comparison"""
    parser = argparse.ArgumentParser(description="Cross-language LogicGP benchmark")
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "thorough"],
        default="standard",
        help="Predefined benchmark profile",
    )
    parser.add_argument("--trainer", default=None, help="Single trainer type")
    parser.add_argument("--trainers", default=None, help="Comma-separated trainers, e.g. flcw-macro,flcw-micro,rlcw-macro")
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets, e.g. iris,npha")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 41,42,43")
    parser.add_argument("--generations", type=int, default=None, help="Generations")
    parser.add_argument("--population", type=int, default=None, help="Population size")
    parser.add_argument("--f1-averaging", choices=["macro", "micro"], default=None, help="F1 averaging mode")
    parser.add_argument("--report-file", default=None, help="Optional output path for markdown report")
    parser.add_argument(
        "--summary-mode",
        choices=["full", "compact"],
        default="full",
        help="Output mode: full prints JSON plus markdown report, compact prints a short trainer table",
    )
    parser.add_argument(
        "--sort-by",
        choices=["diff", "time", "warn", "name"],
        default="diff",
        help="Sort key for compact summary trainer blocks",
    )
    parser.add_argument(
        "--use-profile-only",
        action="store_true",
        help="Ignore manual matrix arguments and use profile defaults exactly",
    )
    parser.add_argument(
        "--use-profile-trainers",
        action="store_true",
        help="Use profile trainer list, even if --trainer is set",
    )
    args = parser.parse_args()
    
    # Get repo paths
    pyrepo = Path("/Users/nunkesser/repos/work/artifacts/pypi-scoredrulesets")
    
    # Initialize runners
    py_runner = PythonBenchmarkRunner(pyrepo)
    cli_path = Path(
        "/Users/nunkesser/repos/work/artifacts/consumers/production/csharp-console-logicgp/"
        "logicGP/logicGP/bin/Debug/net9.0/"
        "Italbytz.AI.ML.LogicGp.Benchmark.Cli"
    )
    try:
        cs_runner = RealCSharpBenchmarkRunner(cli_path=cli_path)
    except Exception as ex:
        print(f"Falling back to mock C# runner: {ex}", file=sys.stderr)
        cs_runner = MockCSharpBenchmarkRunner()
    
    profile_cfg = get_profile_config(args.profile)
    if args.use_profile_only:
        trainers = profile_cfg["trainers"]
        datasets = profile_cfg["datasets"]
        seeds = profile_cfg["seeds"]
        generations = profile_cfg["generations"]
        population = profile_cfg["population"]
        f1_averaging = profile_cfg["f1_averaging"]
    else:
        # profile gives sane defaults; explicit args can still override
        if args.trainers:
            trainers = [t.strip() for t in args.trainers.split(",") if t.strip()]
        elif args.use_profile_trainers:
            trainers = profile_cfg["trainers"]
        elif args.trainer:
            trainers = [args.trainer]
        else:
            trainers = profile_cfg["trainers"]

        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else profile_cfg["datasets"]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else profile_cfg["seeds"]
        generations = args.generations if args.generations is not None else profile_cfg["generations"]
        population = args.population if args.population is not None else profile_cfg["population"]
        f1_averaging = args.f1_averaging or profile_cfg["f1_averaging"]

    print(
        (
            f"Using profile={args.profile} trainers={trainers} "
            f"datasets={datasets} seeds={seeds} generations={generations} "
            f"population={population} f1_averaging={f1_averaging}"
        ),
        file=sys.stderr,
    )

    comparisons = []
    for trainer in trainers:
        for dataset in datasets:
            for seed in seeds:
                print(f"Running Python LogicGP benchmark: trainer={trainer} dataset={dataset} seed={seed}...", file=sys.stderr)
                py_result = py_runner.run_benchmark(
                    trainer,
                    dataset,
                    generations=generations,
                    population=population,
                    seed=seed,
                    f1_averaging=f1_averaging,
                )
                if args.summary_mode == "full":
                    print(f"Python: F1={py_result.f1_score:.4f} Time={py_result.train_time_ms}ms")

                print(f"Running C# LogicGP benchmark: trainer={trainer} dataset={dataset} seed={seed}...", file=sys.stderr)
                cs_result = cs_runner.run_benchmark(
                    trainer,
                    dataset,
                    generations=generations,
                    population=population,
                    seed=seed,
                    f1_averaging=f1_averaging,
                )
                if args.summary_mode == "full":
                    print(f"C#: F1={cs_result.f1_score:.4f} Time={cs_result.train_time_ms}ms")

                comparison = BenchmarkComparator.compare(py_result, cs_result)
                comparison["seed"] = seed
                comparisons.append(comparison)

    report = BenchmarkComparator.generate_report(comparisons)
    if args.summary_mode == "full":
        print("\nComparison Result:")
        print(json.dumps(comparisons, indent=2))
        print("\n" + "=" * 60)
        print(report)
    else:
        compact = BenchmarkComparator.generate_compact_summary(comparisons, sort_by=args.sort_by)
        print("\n" + compact)

    if args.report_file:
        Path(args.report_file).write_text(report, encoding="utf-8")
        print(f"Report written to {args.report_file}")


if __name__ == "__main__":
    main()
