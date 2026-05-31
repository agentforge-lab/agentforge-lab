"""
LLM client — unified interface for Ollama (local) and Anthropic API.
Uses LiteLLM to route between backends transparently.
Default: ollama/qwen2.5-coder:1.5b — works with zero API key or cost.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.local")


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    duration_ms: int
    cost_usd: float = 0.0
    tool_calls: list[dict] | None = None   # populated when model calls tools


# Anthropic pricing per 1M tokens
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"in": 0.80,  "out": 4.00},
    "claude-sonnet-4-6":         {"in": 3.00,  "out": 15.00},
}


class LLMClient:
    """
    Single entry point for all LLM calls in AgentForge.

    Priority order:
      1. Local Ollama model (free, offline, default)
      2. Anthropic API (if key present and force_api=True or no local model)
    """

    OLLAMA_BASE = "http://localhost:11434"

    def __init__(
        self,
        local_model: str = "qwen2.5-coder:1.5b",
        api_model: str = "claude-haiku-4-5-20251001",
        prefer_local: bool = True,
    ):
        self.local_model = local_model
        self.api_model = api_model
        self.prefer_local = prefer_local
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key) and self._api_key.startswith("sk-")

    def _resolve_model(self, force_api: bool) -> tuple[str, str | None]:
        """
        Return (litellm_model_string, api_base_or_None).
        LiteLLM auto-detects provider from model string prefix:
          claude-*  → Anthropic, gpt-* → OpenAI, gemini/* → Google,
          groq/*    → Groq,     ollama/* → Ollama local
        """
        if self.prefer_local and self.local_model:
            return f"ollama/{self.local_model}", self.OLLAMA_BASE
        if self.api_model:
            # Pass raw model string — LiteLLM handles all provider routing
            return self.api_model, None
        if self.local_model:
            return f"ollama/{self.local_model}", self.OLLAMA_BASE
        raise RuntimeError(
            "No LLM configured. Options:\n"
            "  Local:  ollama serve && ollama pull qwen2.5-coder:7b\n"
            "  Free:   agentforge keys set gemini <key>  (aistudio.google.com)\n"
            "  Paid:   agentforge keys set anthropic <key>"
        )

    def complete(self, system: str, user: str, **kwargs) -> LLMResponse:
        """Shorthand for a single system + user turn."""
        return self.chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            **kwargs,
        )

    def chat(
        self,
        messages: list[dict],
        force_api: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        _purpose: str = "",
    ) -> LLMResponse:
        """
        Send a chat completion, optionally with tools for function/tool calling.
        When tools are provided and the model calls one, response.tool_calls is populated.
        """
        from litellm import completion
        from src.api.events import emit, E

        model, api_base = self._resolve_model(force_api)

        user_msg   = next((m["content"] for m in reversed(messages) if m["role"] == "user" and isinstance(m.get("content"), str)), "")
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        prompt_preview = (user_msg or "")[:200].replace("\n", " ")
        purpose_label  = _purpose or (system_msg or "")[:60].split("\n")[0]

        emit(E.LLM_CALL_STARTED,
             model=model.split("/")[-1],
             purpose=purpose_label,
             prompt_preview=prompt_preview)

        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if api_base:
            kwargs["api_base"] = api_base
            kwargs["options"] = {"num_ctx": 16384}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        t0 = time.monotonic()
        try:
            response = completion(**kwargs)
        except Exception as e:
            emit(E.LLM_CALL_FAILED, model=model.split("/")[-1], error=str(e)[:200])
            raise

        duration_ms = int((time.monotonic() - t0) * 1000)

        choice  = response.choices[0]
        content = choice.message.content or ""
        usage   = response.usage
        tokens_in  = usage.prompt_tokens     if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        # Extract tool calls if the model used them
        tool_calls_out: list[dict] | None = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_calls_out = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        cost = 0.0
        if not api_base and model in _PRICING:
            p = _PRICING[model]
            cost = (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]

        emit(E.LLM_CALL_COMPLETED,
             model=model.split("/")[-1],
             purpose=purpose_label,
             tokens_in=tokens_in,
             tokens_out=tokens_out,
             duration_ms=duration_ms,
             cost_usd=round(cost, 6),
             response_preview=content[:200].replace("\n", " "))

        return LLMResponse(
            content=content,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            cost_usd=cost,
            tool_calls=tool_calls_out,
        )

    @classmethod
    def from_hardware_profile(cls, profile_path: Path = Path(".agentforge/hardware_profile.md")) -> "LLMClient":
        """Build a client pre-configured from the detected hardware profile."""
        local_model = "qwen2.5-coder:1.5b"  # default
        api_model = "claude-haiku-4-5-20251001"

        if profile_path.exists():
            text = profile_path.read_text()
            for line in text.splitlines():
                if "ollama pull" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("pull")
                        candidate = parts[idx + 1]
                        if candidate and not candidate.startswith("http"):
                            local_model = candidate
                    except (ValueError, IndexError):
                        pass
                    break

        return cls(local_model=local_model, api_model=api_model)
