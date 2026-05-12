from __future__ import annotations

from typing import Any

from src.agents.llm_client import LLMClient
from src.agents.prompt_templates import option_writer_prompt


class OptionWriter:
    """Converts selected hypotheses into user-facing multiple-choice option text."""

    def __init__(self, llm: LLMClient, include_none: bool = True) -> None:
        self.llm = llm
        self.include_none = include_none
        self.implementation_mode = "template" if llm.mode == "mock" else "real_llm"

    def write(
        self,
        hypotheses: list[dict[str, Any]],
        representative_items: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        if self.llm.mode == "mock":
            options = self._template_options(hypotheses)
        else:
            msgs = option_writer_prompt(hypotheses, representative_items)
            parsed = self.llm.structured_json(msgs)
            if isinstance(parsed, list) and len(parsed) == len(hypotheses):
                options = parsed
            else:
                options = self._template_options(hypotheses)

        if self.include_none:
            options = list(options) + [
                {"option_id": "none", "option_text": "None of the above fits well."}
            ]
        return options

    def _template_options(self, hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"option_id": h["hypothesis_id"], "option_text": h["text_description"]}
            for h in hypotheses
        ]
