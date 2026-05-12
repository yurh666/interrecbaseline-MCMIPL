from __future__ import annotations

from typing import Any


def _item_label(item_id: str, item_lookup: dict[str, dict[str, Any]] | None) -> str:
    """Return a human-readable label for an item.

    Falls back to the raw item_id string if no metadata is available.
    Uses artist_name > title > item_id, plus first 2 tags if present.
    """
    if not item_lookup:
        return str(item_id)
    meta = item_lookup.get(str(item_id), {})
    name = meta.get("artist_name") or meta.get("title") or str(item_id)
    tags = str(meta.get("tags", "")).strip()
    tag_snippet = f" ({', '.join(tags.split()[:3])})" if tags else ""
    return f"{name}{tag_snippet}"


def direction_translator_prompt(
    directions: list[dict[str, Any]],
    item_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build the LLM prompt to translate K uncertainty directions into semantic descriptions.

    Parameters
    ----------
    directions : output of build_anchor_directions (contains positive_anchors / negative_anchors as item_ids)
    item_lookup : dict mapping item_id -> item metadata row (with artist_name, tags, etc.)
                  If provided, sends real item names to the LLM instead of numeric IDs.
    """
    def fmt_anchors(ids: list[str]) -> str:
        return "; ".join(_item_label(iid, item_lookup) for iid in ids)

    formatted = "\n".join(
        f"Direction {d['direction_id']} (uncertainty={d.get('lambda', 0.0):.3f}):\n"
        f"  Positive anchors (items the user probably likes more): {fmt_anchors(d['positive_anchors'])}\n"
        f"  Negative anchors (items the user probably likes less): {fmt_anchors(d['negative_anchors'])}"
        for d in directions
    )
    system = (
        "You are an expert recommender system assistant. "
        "Your task is to interpret item-embedding directions as human-understandable preference dimensions."
    )
    user = (
        f"Below are {len(directions)} uncertainty directions found in the user's preference embedding space. "
        "Each direction is described by its top 5 'positive anchor' items "
        "(items projected most strongly in this direction relative to the user's current mean preference) "
        "and 5 'negative anchor' items (projected most negatively).\n\n"
        f"{formatted}\n\n"
        "For each direction, output a JSON list with objects:\n"
        '{"direction_id": int, "direction_name": str, "positive_side": str, "negative_side": str, "is_meaningful": bool}\n'
        "- direction_name: a concise 2-5 word label for this preference axis\n"
        "- positive_side: what the positive-anchor items represent (1 sentence)\n"
        "- negative_side: what the negative-anchor items represent (1 sentence)\n"
        "- is_meaningful: false if the two ends look too similar or the direction is not interpretable\n"
        "Output ONLY the JSON array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def hypothesis_generator_prompt(
    directions: list[dict[str, Any]],
    observed_history_summary: str,
    M: int = 5,
) -> list[dict[str, str]]:
    """Build the LLM prompt to generate M intent hypotheses.

    directions must be sorted by lambda descending (highest uncertainty first).
    observed_history_summary should contain actual item names, not IDs.
    """
    # directions are already sorted by lambda descending from eigendecompose
    direction_text = "\n".join(
        f"Direction {d['direction_id']} (uncertainty={d.get('lambda', 0.0):.3f}): {d.get('direction_name', '')}\n"
        f"  Positive side: {d.get('positive_side', '')}\n"
        f"  Negative side: {d.get('negative_side', '')}"
        for d in directions
        if d.get("is_meaningful", True)
    )
    system = (
        "You are a recommendation agent that generates candidate intent hypotheses "
        "based on a user's uncertainty profile and interaction history."
    )
    user = (
        f"The user's recent interaction history:\n{observed_history_summary}\n\n"
        f"The system is uncertain about the user's preference in these directions "
        f"(sorted by uncertainty, highest first):\n{direction_text}\n\n"
        f"Generate exactly {M} distinct intent hypotheses. Each hypothesis should:\n"
        "- Address one or more of the uncertainty directions listed above\n"
        "- Be a plausible, specific user preference intent given their history\n"
        "- Include feature_signature: 2-5 keywords (genres, moods, styles) useful for "
        "retrieving representative items from a music catalog\n\n"
        'Output a JSON array of objects: {"hypothesis_id": str, "text_description": str, '
        '"feature_signature": [str], "rationale": str}.\n'
        "Output ONLY the JSON array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def option_writer_prompt(
    hypotheses: list[dict[str, Any]],
    representative_items: dict[str, list[str]],
) -> list[dict[str, str]]:
    options_text = "\n".join(
        f"Option {h['hypothesis_id']}: {h['text_description']}\n"
        f"  Representative items: {representative_items.get(h['hypothesis_id'], [])[:3]}"
        for h in hypotheses
    )
    system = "You are a neutral survey question writer for a recommender system."
    user = (
        "Rewrite the following intent options as concise, neutral multiple-choice answers (10-20 words each). "
        "Avoid any subjective superlatives. Keep sentence structure parallel.\n\n"
        f"{options_text}\n\n"
        'Output a JSON array: [{"option_id": str, "option_text": str}]. '
        "Do NOT include the 'none' option — it will be appended automatically. "
        "Output ONLY the JSON array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
