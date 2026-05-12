#!/usr/bin/env python3
"""Build item embeddings from the processed items.csv.

Usage
-----
From the interrec/ project root::

    python scripts/build_item_embeddings.py --config configs/default.yaml

Outputs to data/processed/:
    item_embeddings.npy
    item_id_to_index.json
    item_index.pkl

Supported modes (set via embedding.mode in config):
    tfidf_svd         — offline, no GPU required (default)
    sentence_transformer — requires sentence-transformers package
    bge               — alias for sentence_transformer with a BGE model
    mock              — random vectors, for testing only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.embedding.item_encoder import ItemEncoder
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build item embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        default=None,
        help="Override embedding.mode from config (tfidf_svd | bge | mock)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    emb_cfg = cfg.get("embedding", {})
    mode = args.mode or emb_cfg.get("mode", "tfidf_svd")
    dim = int(emb_cfg.get("dim", 128))
    normalize = bool(emb_cfg.get("normalize", True))
    model_name = emb_cfg.get("bge_model") if mode in {"bge", "sentence_transformer"} else None
    processed_dir = Path(cfg["dataset"].get("processed_dir", "data/processed"))
    items_path = processed_dir / "items.csv"

    if not items_path.exists():
        print(f"[embed] ERROR: {items_path} not found. Run preprocess_dataset.py first.")
        sys.exit(1)

    items = pd.read_csv(items_path).fillna("")
    print(f"[embed] Encoding {len(items)} items with mode={mode}, dim={dim}")

    encoder = ItemEncoder(mode=mode, dim=dim, normalize=normalize, model_name=model_name)
    try:
        encoded = encoder.fit_transform(items)
    except Exception as exc:
        if emb_cfg.get("use_mock_if_failed", True):
            print(f"[embed] WARNING: encoding failed ({exc}), falling back to mock.")
            encoder = ItemEncoder(mode="mock", dim=dim, normalize=normalize)
            encoded = encoder.fit_transform(items)
        else:
            raise

    encoder.save(encoded, processed_dir)
    print(f"[embed] Saved to {processed_dir}")
    print(f"  embeddings shape: {encoded.embeddings.shape}")
    print(f"  mode: {encoded.mode}")


if __name__ == "__main__":
    main()
