from __future__ import annotations

import math


def hit_rate(recommended: list[str], relevant: set[str], k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ranking_metrics(recommended: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    return {
        f"HitRate@{k}": hit_rate(recommended, relevant, k),
        f"NDCG@{k}": ndcg_at_k(recommended, relevant, k),
        f"MRR@{k}": mrr_at_k(recommended, relevant, k),
    }
