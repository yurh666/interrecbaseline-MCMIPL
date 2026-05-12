#!/usr/bin/env python3
"""Generate a report.md for an existing run directory.

Usage::

    python scripts/generate_report.py --run-dir experiments/runs/<run_id> --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging.report_generator import generate_run_report
from src.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    path = generate_run_report(args.run_dir, cfg)
    print(f"Report written to: {path}")


if __name__ == "__main__":
    main()
