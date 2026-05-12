from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.embedding.item_encoder import build_item_text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", str(text).lower())


@dataclass
class BM25Index:
    item_ids: list[str]
    corpus_tokens: list[list[str]]
    engine: object

    @classmethod
    def build(cls, items: pd.DataFrame) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        texts = [build_item_text(row) for _, row in items.fillna("").iterrows()]
        tokens = [tokenize(t) for t in texts]
        return cls(items["item_id"].astype(str).tolist(), tokens, BM25Okapi(tokens))

    def search(self, query: str, top_k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        scores = self.engine.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        exclude = exclude or set()
        out = []
        for idx in order:
            item_id = self.item_ids[idx]
            if item_id in exclude:
                continue
            out.append((item_id, float(scores[idx])))
            if len(out) >= top_k:
                break
        return out
