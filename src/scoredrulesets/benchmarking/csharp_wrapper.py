"""
Python wrapper to run LogicGP benchmarks against the C# implementation.

This module provides a high-level interface to:
1. Run benchmarks using the C# CLI tool
2. Run the same benchmarks using the Python implementation
3. Compare results and generate reports
"""

import subprocess
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import time

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    trainer: str
    dataset: str
    f1_score: float
    train_time_ms: float
    total_time_ms: float
    f1_averaging: str = "macro"
    num_rules: int = 0
    num_atoms: int = 0
    status: str = "ok"
    error: Optional[str] = None
    implementation: str = "unknown"  # "python" or "csharp"

class CSharBenchmarkRunner:
    """Runs benchmarks using the C# CLI tool."""
    
    def __init__(self, cli_path: Optional[Path] = None):
        """
        Initialize the C# benchmark runner.
        
        Args:
            cli_path: Path to the compiled CLI tool. If None, will search in common locations.
        """
        self.cli_path = cli_path or self._find_cli_tool()
        if not self.cli_path:
            raise FileNotFoundError("C# CLI tool not found. Please build first with 'dotnet build'")
    
    def _find_cli_tool(self) -> Optional[Path]:
        """Find the CLI tool in common output locations."""
        common_paths = [
            Path("/Users/nunkesser/repos/work/artifacts/nuget-adapters-algorithms-ea")
            / "Italbytz.Adapters.Algorithms.EA"
            / "Italbytz.Adapters.Algorithms.EA.Benchmark.Cli"
            / "bin"
            / "Release"
            / "net9.0"
            / "Italbytz.Adapters.Algorithms.EA.Benchmark.Cli",
            Path("/Users/nunkesser/repos/work/artifacts/nuget-adapters-algorithms-ea")
            / "Italbytz.Adapters.Algorithms.EA"
            / "Italbytz.Adapters.Algorithms.EA.Benchmark.Cli"
            / "bin"
            / "Debug"
            / "net9.0"
            / "Italbytz.Adapters.Algorithms.EA.Benchmark.Cli",
        ]
        
        for path in common_paths:
            if path.exists():
                return path
        return None
    
    def run_benchmark(
        self,
        trainer: str = "flcw-macro",
        dataset: str = "iris",
        generations: int = 100,
        population: int = 1000,
        seed: int = 42,
        f1_averaging: str = "macro",
    ) -> BenchmarkResult:
        """
        Run a benchmark using the C# implementation.
        
        Args:
            trainer: Trainer type (flcw-macro, flcw-micro, rlcw-macro, rlcw-micro)
            dataset: Dataset name (iris, npha)
            generations: Number of generations
            population: Population size
            seed: Random seed
            f1_averaging: F1 averaging mode (macro, micro)
            
        Returns:
            BenchmarkResult with metrics
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            cmd = [
                str(self.cli_path),
                "run",
                "--trainer", trainer,
                "--dataset", dataset,
                "--generations", str(generations),
                "--population", str(population),
                "--seed", str(seed),
                "--f1-averaging", f1_averaging,
                "--output", output_file,
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"CLI failed: {result.stderr}")
            
            # Parse result JSON
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            result = BenchmarkResult(
                trainer=data.get("trainer", trainer),
                dataset=data.get("dataset", dataset),
                f1_score=data.get("f1_score", 0.0),
                train_time_ms=data.get("train_time_ms", 0),
                total_time_ms=data.get("total_time_ms", 0),
                f1_averaging=data.get("f1_averaging", f1_averaging),
                num_rules=data.get("num_rules", 0),
                num_atoms=data.get("num_atoms", 0),
                status=data.get("status", "ok"),
                error=data.get("error"),
                implementation="csharp"
            )
            return result
        finally:
            Path(output_file).unlink(missing_ok=True)
    
    def run_benchmarks_batch(
        self,
        configs: List[Dict],
    ) -> List[BenchmarkResult]:
        """
        Run multiple benchmarks.
        
        Args:
            configs: List of benchmark configurations
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        for config in configs:
            try:
                result = self.run_benchmark(**config)
                results.append(result)
            except Exception as e:
                results.append(BenchmarkResult(
                    trainer=config.get("trainer", "unknown"),
                    dataset=config.get("dataset", "unknown"),
                    f1_score=0.0,
                    train_time_ms=0,
                    total_time_ms=0,
                    f1_averaging=config.get("f1_averaging", "macro"),
                    status="error",
                    error=str(e),
                    implementation="csharp"
                ))
        
        return results


class BenchmarkComparator:
    """Compares Python and C# benchmark results."""
    
    @staticmethod
    def compare_results(
        python_result: BenchmarkResult,
        csharp_result: BenchmarkResult,
    ) -> Dict:
        """
        Compare two benchmark results.
        
        Returns a dictionary with differences and analysis.
        """
        return {
            "python": asdict(python_result),
            "csharp": asdict(csharp_result),
            "comparison": {
                "f1_difference": round(abs(python_result.f1_score - csharp_result.f1_score), 4),
                "f1_relative_diff_pct": round(
                    (abs(python_result.f1_score - csharp_result.f1_score) / max(python_result.f1_score, csharp_result.f1_score, 0.0001)) * 100,
                    2
                ) if max(python_result.f1_score, csharp_result.f1_score) > 0 else 0,
                "time_ratio_cs_to_py": round(csharp_result.total_time_ms / max(python_result.total_time_ms, 1), 2),
                "warning": _check_result_validity(python_result, csharp_result)
            }
        }
    
    @staticmethod
    def generate_comparison_report(comparisons: List[Dict]) -> str:
        """
        Generate a human-readable comparison report.
        
        Args:
            comparisons: List of comparison dictionaries from compare_results()
            
        Returns:
            Formatted report string
        """
        report = "# LogicGP Benchmark Comparison Report\n\n"
        
        total_pairs = len(comparisons)
        high_diff_count = sum(1 for c in comparisons if c["comparison"]["f1_relative_diff_pct"] > 5.0)
        avg_time_ratio = sum(c["comparison"]["time_ratio_cs_to_py"] for c in comparisons) / max(total_pairs, 1)
        
        report += f"## Summary\n"
        report += f"- Total comparisons: {total_pairs}\n"
        report += f"- High F1 divergence (>5%): {high_diff_count}\n"
        report += f"- Average C# / Python time ratio: {avg_time_ratio:.2f}x\n\n"
        
        report += f"## Detailed Results\n\n"
        for i, comp in enumerate(comparisons, 1):
            py = comp["python"]
            cs = comp["csharp"]
            cmp = comp["comparison"]
            
            report += f"### Benchmark {i}: {py['trainer']} on {py['dataset']}\n"
            report += f"| Metric | Python | C# | Diff | \n"
            report += f"|--------|--------|-----|------|\n"
            report += f"| F1 Score | {py['f1_score']:.4f} | {cs['f1_score']:.4f} | {cmp['f1_relative_diff_pct']:.2f}% |\n"
            report += f"| Time (ms) | {py['total_time_ms']:.0f} | {cs['total_time_ms']:.0f} | {cmp['time_ratio_cs_to_py']:.2f}x |\n"
            
            if cmp["warning"]:
                report += f"⚠️ **Warning**: {cmp['warning']}\n"
            else:
                report += "✓ Results within expected tolerance\n"
            
            report += "\n"
        
        return report


def _check_result_validity(
    python_result: BenchmarkResult,
    csharp_result: BenchmarkResult
) -> Optional[str]:
    """Check for anomalies between Python and C# results."""
    
    # Check for significant F1 divergence
    f1_diff = abs(python_result.f1_score - csharp_result.f1_score)
    if f1_diff > 0.1:
        return (f"Large F1 divergence: {f1_diff:.4f}. "
                "This may indicate different discretization or randomization.")
    
    # Check for large time differences (may indicate optimization differences)
    if csharp_result.total_time_ms > python_result.total_time_ms * 10:
        return "C# implementation is significantly slower (>10x)"
    elif python_result.total_time_ms > csharp_result.total_time_ms * 10:
        return "Python implementation is significantly slower (>10x)"
    
    # Check for NaN/Inf
    if not (0 <= python_result.f1_score <= 1.0):
        return f"Invalid Python F1 score: {python_result.f1_score}"
    if not (0 <= csharp_result.f1_score <= 1.0):
        return f"Invalid C# F1 score: {csharp_result.f1_score}"
    
    return None


# Example usage function
def example_usage():
    """Example showing how to use this module."""
    
    # Initialize C# runner
    runner = CSharBenchmarkRunner()
    
    # Run benchmarks
    configs = [
        {"trainer": "flcw-macro", "dataset": "iris", "generations": 100},
        {"trainer": "rlcw-macro", "dataset": "iris", "generations": 100},
    ]
    
    csharp_results = runner.run_benchmarks_batch(configs)
    
    # Print results
    for result in csharp_results:
        print(f"{result.trainer} on {result.dataset}: F1={result.f1_score:.4f}")
        if result.error:
            print(f"  Error: {result.error}")


if __name__ == "__main__":
    example_usage()
