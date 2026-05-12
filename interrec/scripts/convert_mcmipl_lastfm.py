#!/usr/bin/env python3
"""Convert MCMIPL lastfm_star processed data to InterRec CSV format.

Prefer `convert_mcmipl_to_interrec_csv.py` for the same convention on all four
MCMIPL datasets (lastfm_star, yelp_star, book, movie).

This script reads the already-processed MCMIPL LastFM data (review_dict JSON +
item_fea pickle) and produces the interactions.csv / items.csv expected by
preprocess_dataset.py.

Why use this instead of raw hetrec2011 TSV?
- The MCMIPL data is already available locally, no extra download needed.
- Sharing the same item split ensures fair comparison between InterRec and MCMIPL.

Limitation:
- item_fea only contains tag IDs (integers), NOT artist/track names.
  TF-IDF on tag IDs produces lower-quality semantic embeddings than real text.
  When the hetrec2011-lastfm-2k download is available, use preprocess_lastfm.py
  instead to get full artist/track name text.

Usage::

    python scripts/convert_mcmipl_lastfm.py \\
        --mcmipl-dir /home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL \\
        --dataset lastfm_star \\
        --out data/raw
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def load_review_dicts(data_dir: Path) -> dict[str, list[int]]:
    """Merge train + valid + test review dicts into one full history per user."""
    full: dict[str, list[int]] = {}
    for split in ["train", "valid", "test"]:
        p = data_dir / "UI_Interaction_data" / f"review_dict_{split}.json"
        if not p.exists():
            print(f"  [warn] {p} not found, skipping")
            continue
        data = json.loads(p.read_text())
        for uid, items in data.items():
            full.setdefault(str(uid), [])
            full[str(uid)].extend([int(x) for x in items])
    return full


def load_item_fea(data_dir: Path) -> dict[int, tuple]:
    p = data_dir / "Graph_generate_data" / "item_fea.pkl"
    with p.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mcmipl-dir", default="/home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL")
    p.add_argument("--dataset", default="lastfm_star")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    data_dir = Path(args.mcmipl_dir) / "data" / args.dataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] Loading review dicts from {data_dir}")
    histories = load_review_dicts(data_dir)
    print(f"[convert] {len(histories)} users")

    item_fea = load_item_fea(data_dir)
    print(f"[convert] {len(item_fea)} items with features")

    # Build interactions: one row per (user, item, position-as-timestamp)
    # We use list order as timestamp proxy (MCMIPL does not store real timestamps)
    rows = []
    all_items: set[int] = set()
    for uid, items in histories.items():
        seen: set[int] = set()
        for ts, iid in enumerate(items):
            if iid in seen:
                continue  # deduplicate within user
            seen.add(iid)
            all_items.add(iid)
            rows.append({
                "user_id": str(uid),
                "item_id": str(iid),
                "timestamp": ts,
                "play_count": 1,  # MCMIPL does not store counts
            })
    interactions = pd.DataFrame(rows)
    interactions.to_csv(out_dir / "interactions.csv", index=False)
    print(f"[convert] Saved interactions: {len(interactions)} rows -> {out_dir / 'interactions.csv'}")

    # Build items: item_id, tags (as comma-separated integers = tag IDs)
    items_rows = []
    for iid in sorted(all_items):
        tags = item_fea.get(iid, ())
        tag_str = " ".join(str(t) for t in tags)
        items_rows.append({
            "item_id": str(iid),
            "title": f"item_{iid}",
            "artist_name": "",
            "track_name": f"item_{iid}",
            "category": "",
            "tags": tag_str,
            "description": f"item {iid} tags: {tag_str}",
        })
    items_df = pd.DataFrame(items_rows)
    items_df.to_csv(out_dir / "items.csv", index=False)
    print(f"[convert] Saved items: {len(items_df)} rows -> {out_dir / 'items.csv'}")
    print()
    print("NOTE: item text only contains numeric tag IDs (not artist/track names).")
    print("For full-quality semantic embeddings, use preprocess_lastfm.py with")
    print("the hetrec2011-lastfm-2k dataset which contains actual text metadata.")


if __name__ == "__main__":
    main()
