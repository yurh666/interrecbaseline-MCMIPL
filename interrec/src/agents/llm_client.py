from __future__ import annotations

import json
import time
from typing import Any


class LLMClient:
    """Thin wrapper around an LLM provider with mock fallback.

    When mode='mock', every call returns a canned JSON response without
    hitting any network endpoint. This is the default for the current
    milestone and is recorded as implementation_mode='mock_llm'.
    """

    def __init__(
        self,
        mode: str = "mock",
        provider: str = "none",
        log_prompts: bool = True,
        log_responses: bool = True,
        max_retry: int = 3,
    ) -> None:
        self.mode = mode
        self.provider = provider
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.max_retry = max_retry
        self.implementation_mode = "mock_llm" if mode == "mock" else f"real_llm_{provider}"

        self._client: Any = None
        if mode != "mock":
            self._init_real_client()

    def _init_real_client(self) -> None:
        if self.provider == "openai":
            import openai  # type: ignore
            self._client = openai.OpenAI()
        elif self.provider == "anthropic":
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat(self, messages: list[dict[str, str]], model: str = "gpt-4o-mini", **kwargs: Any) -> str:
        if self.mode == "mock":
            return self._mock_response(messages)
        for attempt in range(self.max_retry):
            try:
                return self._real_chat(messages, model, **kwargs)
            except Exception as exc:
                if attempt == self.max_retry - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _real_chat(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
            return resp.choices[0].message.content or ""
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[m for m in messages if m["role"] != "system"],
                system=next((m["content"] for m in messages if m["role"] == "system"), ""),
                **kwargs,
            )
            return resp.content[0].text
        raise NotImplementedError

    def _mock_response(self, messages: list[dict[str, str]]) -> str:
        last = messages[-1]["content"] if messages else ""
        if "direction" in last.lower():
            return json.dumps(
                [
                    {
                        "direction_id": 0,
                        "direction_name": "mock_direction_0",
                        "positive_side": "upbeat energetic tracks",
                        "negative_side": "slow calm music",
                        "is_meaningful": True,
                    }
                ]
            )
        if "hypothesis" in last.lower():
            return json.dumps(
                [
                    {
                        "hypothesis_id": "h1",
                        "text_description": "User enjoys upbeat pop music for workouts.",
                        "feature_signature": ["energetic", "pop", "workout"],
                        "rationale": "Mock: direction 0 positive side.",
                    },
                    {
                        "hypothesis_id": "h2",
                        "text_description": "User prefers relaxing indie music for studying.",
                        "feature_signature": ["indie", "calm", "study"],
                        "rationale": "Mock: direction 0 negative side.",
                    },
                ]
            )
        return json.dumps({"result": "mock", "content": "No specific mock rule matched."})

    def structured_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        raw = self.chat(messages, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\[.*\]|\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {}
