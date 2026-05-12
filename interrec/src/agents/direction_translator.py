from __future__ import annotations

from typing import Any

from src.agents.llm_client import LLMClient
from src.agents.prompt_templates import direction_translator_prompt


class DirectionTranslator:
    """Converts anchor-item pairs (positive/negative) into semantic direction descriptions.

    In mock / rule_based mode, returns a templated description without calling an LLM.
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

    def translate(self, directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.llm.mode == "mock":
            return self._rule_based(directions)
        msgs = direction_translator_prompt(directions, item_lookup=self.item_lookup)
        parsed = self.llm.structured_json(msgs)
        if isinstance(parsed, list):
            return self._merge_with_input(directions, parsed)
        return self._rule_based(directions)

    def _rule_based(self, directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from src.agents.prompt_templates import _item_label
        out = []
        for d in directions:
            pos = d.get("positive_anchors", [])[:2]
            neg = d.get("negative_anchors", [])[:2]
            pos_desc = ", ".join(_item_label(x, self.item_lookup) for x in pos)
            neg_desc = ", ".join(_item_label(x, self.item_lookup) for x in neg)
            out.append(
                {
                    "direction_id": d["direction_id"],
                    "direction_name": f"direction_{d['direction_id']}",
                    "positive_side": f"items like: {pos_desc}",
                    "negative_side": f"items like: {neg_desc}",
                    "is_meaningful": True,
                    "lambda": d.get("lambda", 0.0),
                    "positive_anchors": d.get("positive_anchors", []),
                    "negative_anchors": d.get("negative_anchors", []),
                }
            )
        return out

    def _merge_with_input(
        self, directions: list[dict[str, Any]], parsed: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id = {p["direction_id"]: p for p in parsed}
        out = []
        for d in directions:
            did = d["direction_id"]
            merged = dict(d)
            if did in by_id:
                merged.update(by_id[did])
            else:
                merged.setdefault("direction_name", f"direction_{did}")
                merged.setdefault("positive_side", "")
                merged.setdefault("negative_side", "")
                merged.setdefault("is_meaningful", True)
            out.append(merged)
        return out
