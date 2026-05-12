"""Smoke test: synthetic 50-user dataset through the full pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 0. build tiny synthetic data ─────────────────────────────────────────────
np.random.seed(42)
n_users, n_items = 50, 200
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

rows = []
for uid in range(n_users):
    for seq_pos in range(30):
        iid = np.random.randint(0, n_items)
        rows.append({
            "user_id": uid,
            "item_id": iid,
            "timestamp": seq_pos,
            "play_count": int(np.random.randint(1, 10)),
        })
pd.DataFrame(rows).to_csv("data/raw/interactions.csv", index=False)

genres = ["rock", "pop", "jazz", "classical", "hiphop"]
items = [
    {
        "item_id": iid,
        "title": f"Track_{iid}",
        "artist_name": f"Artist_{iid % 20}",
        "track_name": f"Track_{iid}",
        "category": genres[iid % len(genres)],
        "tags": f"tag_{iid % 10}",
        "description": f"A great {genres[iid % len(genres)]} song.",
    }
    for iid in range(n_items)
]
pd.DataFrame(items).to_csv("data/raw/items.csv", index=False)
print("✓ synthetic data written")

# ── 1. preprocess ────────────────────────────────────────────────────────────
from src.utils.config import load_config

cfg = load_config("configs/default.yaml")
cfg["dataset"]["raw_interactions_path"] = "data/raw/interactions.csv"
cfg["dataset"]["raw_items_path"] = "data/raw/items.csv"
cfg["dataset"]["min_user_interactions"] = 10
cfg["dataset"]["min_item_interactions"] = 2
cfg["simulation"]["max_users"] = 5
cfg["simulation"]["max_turns"] = 3
cfg["ask"]["mc_samples"] = 16

from src.data.preprocess import preprocess_dataset

result = preprocess_dataset(cfg)
print(f"✓ preprocess: {result['n_users']} users, {result['n_items']} items")

# ── 2. build embeddings (tfidf_svd) ─────────────────────────────────────────
items_df = pd.read_csv("data/processed/items.csv").fillna("")
from src.embedding.item_encoder import ItemEncoder

enc = ItemEncoder(mode="tfidf_svd", dim=32, normalize=True)
encoded = enc.fit_transform(items_df)
enc.save(encoded, "data/processed")
print(f"✓ embeddings: {encoded.embeddings.shape}")

# ── 3. full pipeline ─────────────────────────────────────────────────────────
from src.data.dataset import InterRecDataset
from src.embedding.bm25_index import BM25Index
from src.embedding.index import EmbeddingIndex
from src.logging.run_logger import RunLogger
from src.simulation.experiment_runner import run_bm25_baseline, run_interrec_experiment
from src.utils.seed import set_seed
from src.utils.time import make_run_id

set_seed(42)
dataset = InterRecDataset.load("data/processed")
index = EmbeddingIndex.load("data/processed")
bm25 = BM25Index.build(dataset.items)

run_id = make_run_id("smoke", "interrec", 42)
logger = RunLogger(run_id, "experiments/runs")
logger.write_config(cfg)

agg = run_interrec_experiment(
    cfg, Path("experiments/runs") / run_id, dataset, index, logger, bm25
)
print("✓ InterRec run, metrics:", {k: round(v, 4) for k, v in agg.items() if isinstance(v, float)})

run_id2 = make_run_id("smoke", "bm25", 42)
logger2 = RunLogger(run_id2, "experiments/runs")
logger2.write_config(cfg)
agg2 = run_bm25_baseline(cfg, dataset, index, logger2, bm25)
print("✓ BM25 baseline, metrics:", {k: round(v, 4) for k, v in agg2.items() if isinstance(v, float)})

print("\nAll smoke tests passed!")
