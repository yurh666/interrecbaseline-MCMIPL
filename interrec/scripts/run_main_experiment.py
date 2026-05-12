#!/usr/bin/env python3
"""Run the InterRec main experiment or BM25 baseline.

Usage
-----
From the interrec/ project root::

    # InterRec (mock LLM, tfidf_svd embeddings)
    python scripts/run_main_experiment.py --config configs/default.yaml

    # BM25 baseline only
    python scripts/run_main_experiment.py --config configs/default.yaml --method bm25

    # Override split and max_users
    python scripts/run_main_experiment.py --split test --max-users 100

The script will NOT run if:
  - data/processed/interactions.csv does not exist  → run preprocess_dataset.py
  - data/processed/item_embeddings.npy does not exist → run build_item_embeddings.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import InterRecDataset
from src.embedding.bm25_index import BM25Index
from src.embedding.index import EmbeddingIndex
from src.logging.report_generator import generate_run_report
from src.logging.run_logger import RunLogger
from src.simulation.experiment_runner import run_bm25_baseline, run_interrec_experiment
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.time import make_run_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--method", default="interrec", choices=["interrec", "bm25"])
    p.add_argument("--split", default=None, help="Override simulation.split (train/valid/test)")
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.split:
        cfg.setdefault("simulation", {})["split"] = args.split
    if args.max_users is not None:
        cfg.setdefault("simulation", {})["max_users"] = args.max_users
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    method = args.method
    run_id = make_run_id(dataset_name, method, seed)
    run_dir = Path("experiments/runs") / run_id
    print(f"[run] run_id = {run_id}")

    logger = RunLogger(run_id, base_dir="experiments/runs")
    logger.write_config(cfg)
    logger.write_environment()
    logger.write_git_info(project_dir=Path(__file__).resolve().parent.parent)

    processed_dir = Path(cfg["dataset"].get("processed_dir", "data/processed"))
    for required in ["interactions.csv", "items.csv", "sessions.json"]:
        if not (processed_dir / required).exists():
            print(f"[run] ERROR: {processed_dir / required} not found.")
            print("       Run: python scripts/preprocess_dataset.py --config <config>")
            sys.exit(1)

    emb_path = processed_dir / "item_embeddings.npy"
    if not emb_path.exists():
        print(f"[run] ERROR: {emb_path} not found.")
        print("       Run: python scripts/build_item_embeddings.py --config <config>")
        sys.exit(1)

    print("[run] Loading dataset...")
    dataset = InterRecDataset.load(processed_dir)
    print(f"      {len(dataset.sessions)} sessions, {len(dataset.items)} items")

    print("[run] Loading embedding index...")
    index = EmbeddingIndex.load(processed_dir)

    print("[run] Building BM25 index...")
    bm25 = BM25Index.build(dataset.items)

    if method == "bm25":
        print("[run] Running BM25 baseline...")
        agg = run_bm25_baseline(cfg, dataset, index, logger, bm25)
    else:
        print("[run] Running InterRec...")
        agg = run_interrec_experiment(cfg, run_dir, dataset, index, logger, bm25)
        if cfg.get("logging", {}).get("generate_report", False):
            report_path = generate_run_report(run_dir, cfg)
            print(f"[run] Report: {report_path}")

    save_config(cfg, run_dir / "config.yaml")
    print("[run] Aggregate metrics:")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"[run] Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
