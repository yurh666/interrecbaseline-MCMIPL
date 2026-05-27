#!/usr/bin/env python3
"""Extract CRS-style metrics (SR10, AvgT) from MCMIPL train logs — no checkpoint needed."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

DONE_RE = re.compile(r"=== DONE: (\w+) seed=(\d+)")
BEST_RE = re.compile(
    r"best!!!!!!!!!SR5:([\d.]+), SR10:([\d.]+), SR15:([\d.]+), AvgT:([\d.]+)"
)
SAVED_RE = re.compile(r"RL policy model saved at .*epoch-(\d+)\.pkl")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    m = re.match(r"train_(\w+)_s(\d+)\.log", path.name)
    dataset = m.group(1) if m else path.stem
    seed = int(m.group(2)) if m else -1

    done = bool(DONE_RE.search(text))
    best = None
    for line in text.splitlines():
        if "best!!!!!!!!!" in line:
            bm = BEST_RE.search(line)
            if bm:
                best = {
                    "SR5": float(bm.group(1)),
                    "SR10": float(bm.group(2)),
                    "SR15": float(bm.group(3)),
                    "AvgT": float(bm.group(4)),
                }
    epochs_saved = sorted({int(x) for x in SAVED_RE.findall(text)})

    return {
        "log_file": path.name,
        "dataset": dataset,
        "seed": seed,
        "phase_b_done_in_log": done,
        "best_valid_crs": best,
        "checkpoint_epochs_in_log": epochs_saved,
        "has_rl_policy_saved": bool(epochs_saved),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(args.log_dir.glob("train_*.log")):
        rows.append(parse_log(p))

    payload = {
        "generated_at": datetime.now().isoformat(),
        "note": "CRS metrics from logs; InterRec ID protocol still requires checkpoint + eval export",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} ({len(rows)} logs)")


if __name__ == "__main__":
    main()
