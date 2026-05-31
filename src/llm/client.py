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
        """Return (litellm_model_string, api_base_or_None)."""
        if force_api and self.has_api_key:
            return self.api_model, None
        if self.prefer_local and self.local_model:
            return f"ollama/{self.local_model}", self.OLLAMA_BASE
        if self.has_api_key:
            return self.api_model, None
        # Last resort: try Ollama even if prefer_local is False
        if self.local_model:
            return f"ollama/{self.local_model}", self.OLLAMA_BASE
        raise RuntimeError(
            "No LLM available: set ANTHROPIC_API_KEY in .env.local "
            "or start Ollama with `ollama serve`"
        )

    def chat(
        self,
        messages: list[dict],
        force_api: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        _purpose: str = "",        # human-readable label for event log
    ) -> LLMResponse:
        """Send a chat completion. Raises on connection failure."""
        from litellm import completion  # imported here to avoid slow startup elsewhere
        from src.api.events import emit, E

        model, api_base = self._resolve_model(force_api)

        # Build a short prompt preview from the user message
        user_msg   = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        prompt_preview   = user_msg[:200].replace("\n", " ")
        purpose_label    = _purpose or system_msg[:60].split("\n")[0]

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
            kwargs["options"] = {"num_ctx": 16384}  # Ollama default is 2048 — far too small for retry context

        t0 = time.monotonic()
        try:
            response = completion(**kwargs)
        except Exception as e:
            emit(E.LLM_CALL_FAILED, model=model.split("/")[-1], error=str(e)[:200])
            raise

        duration_ms = int((time.monotonic() - t0) * 1000)

        content = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in  = usage.prompt_tokens     if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

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
        )

    def complete(self, system: str, user: str, **kwargs) -> LLMResponse:
        """Shorthand for a single system + user turn."""
        return self.chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            **kwargs,
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
