#!/usr/bin/env python3
"""Take the first N elements of a JSON list (sessions.json) for smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--max", type=int, required=True, dest="max_n")
    args = ap.parse_args()

    raw = json.loads(args.in_path.read_text())
    if not isinstance(raw, list):
        raise SystemExit("Input JSON must be a list")
    sliced = raw[: args.max_n]
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(sliced, ensure_ascii=False, indent=2) + "\n")
    print(f"[slice_sessions] {args.in_path} -> {args.out_path}  n={len(sliced)}")


if __name__ == "__main__":
    main()
