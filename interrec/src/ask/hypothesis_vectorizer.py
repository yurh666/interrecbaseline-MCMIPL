from __future__ import annotations

from typing import Any

import numpy as np

from src.embedding.index import EmbeddingIndex
from src.utils.math import l2_normalize


def vectorize_hypothesis(
    hypothesis: dict[str, Any],
    index: EmbeddingIndex,
    representative_top_k: int = 5,
    item_texts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Retrieve representative items for a hypothesis and average their embeddings.

    Implements: v_h = (1/K) * sum_{j=1..K} e_{i_j}
    where {i_j} are the items whose text best matches the feature_signature.

    Parameters
    ----------
    item_texts : dict mapping item_id -> concatenated text (artist_name + tags + ...).
                 Used for keyword fallback when BM25 is not available.
                 Without this, fallback is essentially random for numeric IDs.
    """
    feature_tokens = hypothesis.get("feature_signature", [])
    query = " ".join(str(t) for t in feature_tokens)

    matched: list[tuple[str, float]] = _keyword_match(
        query, index, top_k=representative_top_k, item_texts=item_texts
    )

    if not matched:
        dim = index.embeddings.shape[1]
        return {
            **hypothesis,
            "representative_items": [],
            "vector": np.zeros(dim, dtype=float),
            "vector_norm": 0.0,
        }

    rep_ids = [item_id for item_id, _ in matched]
    vectors = index.vectors_for(rep_ids)
    avg = np.mean(vectors, axis=0)
    normalized = l2_normalize(avg.reshape(1, -1))[0]
    return {
        **hypothesis,
        "representative_items": rep_ids,
        "vector": normalized,
        "vector_norm": float(np.linalg.norm(normalized)),
    }


def vectorize_hypotheses(
    hypotheses: list[dict[str, Any]],
    index: EmbeddingIndex,
    representative_top_k: int = 5,
    bm25_index: Any = None,
    item_texts: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Vectorize all hypotheses.

    Priority:
    1. BM25 search on item text (preferred – retrieves items by semantic content)
    2. Keyword overlap against item_texts dict (fallback when BM25 not available)

    item_texts : dict mapping item_id -> full text string built from item metadata.
    """
    out = []
    for h in hypotheses:
        if bm25_index is not None:
            query = " ".join(str(t) for t in h.get("feature_signature", []))
            matched = bm25_index.search(query, top_k=representative_top_k)
            rep_ids = [item_id for item_id, _ in matched]
            if rep_ids:
                vectors = index.vectors_for(rep_ids)
                avg = l2_normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
                out.append({
                    **h,
                    "representative_items": rep_ids,
                    "vector": avg,
                    "vector_norm": float(np.linalg.norm(avg)),
                })
                continue
        out.append(vectorize_hypothesis(h, index, representative_top_k, item_texts=item_texts))
    return out


def _keyword_match(
    query: str,
    index: EmbeddingIndex,
    top_k: int,
    item_texts: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Keyword overlap fallback.

    Matches query tokens against item_texts (artist_name + tags + description).
    If item_texts is not provided, falls back to item_id string match (weak).
    """
    tokens = set(query.lower().split())
    scores = []
    for item_id in index.item_ids:
        if item_texts:
            text = item_texts.get(str(item_id), str(item_id))
        else:
            # Last-resort: match against item_id string (unreliable for numeric IDs)
            text = str(item_id).lower().replace("::", " ")
        item_tokens = set(text.lower().split())
        overlap = len(tokens & item_tokens)
        scores.append((item_id, float(overlap)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
