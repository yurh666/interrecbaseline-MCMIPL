from __future__ import annotations

from typing import Any

from src.agents.llm_client import LLMClient
from src.agents.prompt_templates import hypothesis_generator_prompt


class HypothesisGenerator:
    """Generates M candidate intent hypotheses given translated uncertainty directions.

    In mock / rule_based mode, creates one hypothesis per meaningful direction
    without calling an LLM.
    item_lookup (optional) maps item_id -> metadata dict with artist_name, tags, etc.
    When provided and real LLM is used, the prompt sends actual item names instead of IDs.
    """

    def __init__(
        self,
        llm: LLMClient,
        item_lookup: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.llm = llm
        self.item_lookup = item_lookup
        self.implementation_mode = "mock" if llm.mode == "mock" else "real_llm"

    def generate(
        self,
        directions: list[dict[str, Any]],
        observed_items: list[str],
        M: int = 5,
    ) -> list[dict[str, Any]]:
        if self.llm.mode == "mock":
            return self._rule_based(directions, observed_items, M)
        # Build human-readable history summary using actual item names
        from src.agents.prompt_templates import _item_label
        summary = ", ".join(_item_label(x, self.item_lookup) for x in observed_items[-10:])
        msgs = hypothesis_generator_prompt(directions, summary, M)
        parsed = self.llm.structured_json(msgs)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[:M]
        return self._rule_based(directions, observed_items, M)

    # Stop words to skip when extracting feature_signature from mock side descriptions
    _STOP = {"items", "like:", "like", "the", "a", "an", "and", "or", "of", "in", "with",
              "user", "may", "prefer", "vs", "versus", "side", "positive", "negative"}

    def _rule_based(
        self,
        directions: list[dict[str, Any]],
        observed_items: list[str],
        M: int,
    ) -> list[dict[str, Any]]:
        meaningful = [d for d in directions if d.get("is_meaningful", True)]
        hypotheses: list[dict[str, Any]] = []
        for i, d in enumerate(meaningful[:M]):
            hid = f"h{i + 1}"
            pos_side = d.get("positive_side", f"direction_{d['direction_id']} positive")
            neg_side = d.get("negative_side", f"direction_{d['direction_id']} negative")
            hypotheses.append(
                {
                    "hypothesis_id": hid,
                    "text_description": f"User may prefer: {pos_side}",
                    "feature_signature": self._extract_keywords(pos_side),
                    "rationale": f"Mock rule-based from direction {d['direction_id']}",
                    "direction_id": d["direction_id"],
                }
            )
            if len(hypotheses) < M:
                hypotheses.append(
                    {
                        "hypothesis_id": f"h{i + 1}b",
                        "text_description": f"User may prefer: {neg_side}",
                        "feature_signature": self._extract_keywords(neg_side),
                        "rationale": f"Mock rule-based from direction {d['direction_id']} negative",
                        "direction_id": d["direction_id"],
                    }
                )
            if len(hypotheses) >= M:
                break
        return hypotheses[:M]

    def _extract_keywords(self, text: str, max_k: int = 4) -> list[str]:
        """Extract meaningful keywords from a side description string."""
        import re
        tokens = re.sub(r"[(),:;]", " ", text).split()
        keywords = [
            t for t in tokens
            if t.lower() not in self._STOP and len(t) >= 3
        ]
        return keywords[:max_k] if keywords else tokens[:max_k]
