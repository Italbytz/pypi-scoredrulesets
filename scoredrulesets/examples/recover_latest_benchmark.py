from __future__ import annotations

import sys

from recover_heatmap_from_log import main as recover_main


if __name__ == "__main__":
    # One-command convenience entry point:
    #   python examples/recover_latest_benchmark.py
    # Optional extra args still work, e.g.:
    #   python examples/recover_latest_benchmark.py --output-dir benchmarks/recovered-from-log/manual
    recover_main(["--latest-log", *sys.argv[1:]])

