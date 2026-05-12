from __future__ import annotations

import numpy as np

from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex
from src.recommendation.scorer import belief_scores


class BeliefRecommender:
    def __init__(self, index: EmbeddingIndex, lambda_explore: float = 0.1) -> None:
        self.index = index
        self.lambda_explore = lambda_explore

    def recommend(self, belief: BeliefState, top_k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        scores = belief_scores(belief, self.index.embeddings, self.lambda_explore)
        order = np.argsort(-scores)
        exclude = exclude or set()
        out: list[tuple[str, float]] = []
        for idx in order:
            item_id = self.index.item_ids[int(idx)]
            if item_id in exclude:
                continue
            out.append((item_id, float(scores[int(idx)])))
            if len(out) >= top_k:
                break
        return out
