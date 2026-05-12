#!/usr/bin/env python3
"""Single-turn choice diagnostic: checks if intent-level choices cover theta*.

Usage::

    python scripts/run_choice_diagnostic.py --config configs/experiment_choice_diagnostic.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config

# Reuses the main experiment runner with max_turns=1
from scripts.run_main_experiment import main as run_main

if __name__ == "__main__":
    sys.argv += ["--config", "configs/experiment_choice_diagnostic.yaml"]
    run_main()
