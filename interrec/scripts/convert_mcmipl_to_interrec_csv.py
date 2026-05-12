#!/usr/bin/env python3
"""Export MCMIPL official processed splits to InterRec `raw` CSVs (interactions + items).

Use this before `scripts/preprocess_dataset.py` when you want **the same ordered
interaction history** as MCMIPL (list order as timestamp proxy — MCMIPL has no
real event time in these JSONs). That matches the pipeline used by
`convert_mcmipl_lastfm.py`, extended to yelp_star / book / movie.

This does **not** regenerate MCMIPL graph pickles or TransE; for a full baseline
rebuild you still run MCMIPL `graph_init.py` / embeddings on their side.

Usage::

    python scripts/convert_mcmipl_to_interrec_csv.py \\
        --mcmipl-dir /path/to/MCMIPL \\
        --dataset yelp_star \\
        --out data/raw/yelp_mcmipl
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

VALID_DATASETS = ("lastfm_star", "yelp_star", "book", "movie")


def load_review_dicts(data_dir: Path) -> dict[str, list[int]]:
    full: dict[str, list[int]] = {}
    for split in ("train", "valid", "test"):
        p = data_dir / "UI_Interaction_data" / f"review_dict_{split}.json"
        if not p.exists():
            print(f"  [warn] {p} not found, skipping")
            continue
        data = json.loads(p.read_text())
        for uid, items in data.items():
            full.setdefault(str(uid), [])
            full[str(uid)].extend(int(x) for x in items)
    return full


def load_lastfm_item_fea(data_dir: Path) -> dict[int, tuple]:
    p = data_dir / "Graph_generate_data" / "item_fea.pkl"
    with p.open("rb") as f:
        raw = pickle.load(f)
    return {int(k): v for k, v in raw.items()}


def load_yelp_item_meta(data_dir: Path) -> dict[int, list[int]]:
    p = data_dir / "Graph_generate_data" / "item_dict-original_tag.json"
    data = json.loads(p.read_text())
    out: dict[int, list[int]] = {}
    for sid, blob in data.items():
        feats = blob.get("feature_index")
        if feats is None:
            feats = []
        out[int(sid)] = [int(x) for x in feats]
    return out


def load_book_movie_item_features(data_dir: Path) -> dict[int, tuple]:
    p = data_dir / "fea_item" / "item_feature.pkl"
    with p.open("rb") as f:
        raw = pickle.load(f)
    return {int(k): tuple(v) for k, v in raw.items()}


def item_features_for_dataset(data_dir: Path, dataset: str) -> dict[int, tuple]:
    if dataset == "lastfm_star":
        return load_lastfm_item_fea(data_dir)
    if dataset == "yelp_star":
        d = load_yelp_item_meta(data_dir)
        return {k: tuple(v) for k, v in d.items()}
    if dataset in ("book", "movie"):
        return load_book_movie_item_features(data_dir)
    raise ValueError(dataset)


def main() -> None:
    p = argparse.ArgumentParser(description="MCMIPL → InterRec raw CSV bridge")
    p.add_argument(
        "--mcmipl-dir",
        type=Path,
        default=Path("/home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL"),
    )
    p.add_argument(
        "--dataset",
        choices=VALID_DATASETS,
        required=True,
        help="Subfolder under MCMIPL/data/",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for CSVs")
    args = p.parse_args()

    data_dir = args.mcmipl_dir / "data" / args.dataset
    if not data_dir.is_dir():
        raise SystemExit(f"Missing MCMIPL data dir: {data_dir}")

    args.out.mkdir(parents=True, exist_ok=True)

    histories = load_review_dicts(data_dir)
    print(f"[convert] {args.dataset}: {len(histories)} users (merged train/valid/test dicts)")

    try:
        item_fea = item_features_for_dataset(data_dir, args.dataset)
    except FileNotFoundError as e:
        raise SystemExit(f"Item feature file missing for {args.dataset}: {e}") from e
    print(f"[convert] {len(item_fea)} item metadata records")

    rows = []
    all_items: set[int] = set()
    for uid, items in histories.items():
        seen: set[int] = set()
        for ts, iid in enumerate(items):
            if iid in seen:
                continue
            seen.add(iid)
            all_items.add(iid)
            rows.append(
                {
                    "user_id": str(uid),
                    "item_id": str(iid),
                    "timestamp": ts,
                    "play_count": 1,
                }
            )
    interactions = pd.DataFrame(rows)
    interactions.to_csv(args.out / "interactions.csv", index=False)
    print(f"[convert] interactions.csv  rows={len(interactions)}  -> {args.out / 'interactions.csv'}")

    items_rows = []
    for iid in sorted(all_items):
        tags = item_fea.get(iid, ())
        tag_str = " ".join(str(t) for t in tags)
        items_rows.append(
            {
                "item_id": str(iid),
                "title": f"item_{iid}",
                "artist_name": "",
                "track_name": f"item_{iid}",
                "category": "",
                "tags": tag_str,
                "description": f"item {iid} features: {tag_str}",
            }
        )
    items_df = pd.DataFrame(items_rows)
    items_df.to_csv(args.out / "items.csv", index=False)
    print(f"[convert] items.csv  rows={len(items_df)}  -> {args.out / 'items.csv'}")
    print()
    print("Next: point configs/default.yaml (or a dataset-specific yaml) at these paths")
    print("and run:  python scripts/preprocess_dataset.py --config <your.yaml>")


if __name__ == "__main__":
    main()
