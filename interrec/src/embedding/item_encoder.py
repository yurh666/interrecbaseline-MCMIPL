from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.utils.io import save_numpy, save_pickle, write_json
from src.utils.math import l2_normalize


def build_item_text(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    title = get("title", "") or get("track_name", "") or get("artist_name", "") or get("item_id", "")
    return "\n".join(
        [
            f"title: {title}",
            f"artist: {get('artist_name', '')}",
            f"category: {get('category', '') or get('genre', '')}",
            f"tags: {get('tags', '')}",
            f"description: {get('description', '')}",
        ]
    )


@dataclass
class EncodedItems:
    embeddings: np.ndarray
    item_ids: list[str]
    item_id_to_index: dict[str, int]
    mode: str


class ItemEncoder:
    def __init__(self, mode: str = "tfidf_svd", dim: int = 128, normalize: bool = True, model_name: str | None = None) -> None:
        self.mode = mode
        self.dim = dim
        self.normalize = normalize
        self.model_name = model_name
        self.model: Any = None
        self.implementation_mode = mode

    def fit_transform(self, items: pd.DataFrame) -> EncodedItems:
        texts = [build_item_text(row) for _, row in items.fillna("").iterrows()]
        item_ids = items["item_id"].astype(str).tolist()
        if self.mode == "tfidf_svd":
            embeddings = self._tfidf_svd(texts)
        elif self.mode in {"sentence_transformer", "bge"}:
            embeddings = self._sentence_transformer(texts)
        elif self.mode == "mock":
            embeddings = self._mock(texts)
        else:
            raise ValueError(f"Unsupported embedding mode: {self.mode}")
        if self.normalize:
            embeddings = l2_normalize(embeddings, axis=1)
        return EncodedItems(
            embeddings=embeddings.astype(np.float32),
            item_ids=item_ids,
            item_id_to_index={item_id: idx for idx, item_id in enumerate(item_ids)},
            mode=self.implementation_mode,
        )

    def _tfidf_svd(self, texts: list[str]) -> np.ndarray:
        n_samples = len(texts)
        max_components = max(1, min(self.dim, n_samples - 1))
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)),
                ("svd", TruncatedSVD(n_components=max_components, random_state=42)),
            ]
        )
        emb = pipeline.fit_transform(texts)
        self.model = pipeline
        if emb.shape[1] < self.dim:
            emb = np.pad(emb, ((0, 0), (0, self.dim - emb.shape[1])))
        return emb[:, : self.dim]

    def _sentence_transformer(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name or "sentence-transformers/all-MiniLM-L6-v2")
        self.model = model
        return model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    def _mock(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        self.implementation_mode = "mock"
        return rng.normal(size=(len(texts), self.dim))

    def save(self, encoded: EncodedItems, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_numpy(encoded.embeddings, out / "item_embeddings.npy")
        write_json(encoded.item_id_to_index, out / "item_id_to_index.json")
        save_pickle(
            {
                "item_ids": encoded.item_ids,
                "item_id_to_index": encoded.item_id_to_index,
                "mode": encoded.mode,
                "model": self.model,
            },
            out / "item_index.pkl",
        )
