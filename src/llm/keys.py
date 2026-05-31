"""
API key management for all supported providers.

Keys are stored in ~/.agentforge/keys.json and loaded into env vars
so LiteLLM can find them automatically when routing model calls.

Supported providers and the free options:
  anthropic  → ANTHROPIC_API_KEY  (claude-* models, paid)
  openai     → OPENAI_API_KEY     (gpt-* models, paid)
  gemini     → GEMINI_API_KEY     (gemini/* models — FREE tier: 1500 req/day)
               Get free key: https://aistudio.google.com → "Get API key"
  groq       → GROQ_API_KEY       (groq/* models — FREE tier, very fast)
               Get free key: https://console.groq.com
  openrouter → OPENROUTER_API_KEY (openrouter/* models, some free)
               Get key: https://openrouter.ai
  together   → TOGETHER_API_KEY   (together/* models, has free credits)
               Get key: https://api.together.xyz
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_KEYS_FILE = Path.home() / ".agentforge" / "keys.json"

# Maps provider name → environment variable LiteLLM reads
ENV_MAP: dict[str, str] = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "groq":       "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together":   "TOGETHER_API_KEY",
}

# Model string prefixes → provider (for auto-detection)
MODEL_PREFIX_MAP: dict[str, str] = {
    "claude-":      "anthropic",
    "anthropic/":   "anthropic",
    "gpt-":         "openai",
    "openai/":      "openai",
    "o1":           "openai",
    "o3":           "openai",
    "gemini/":      "gemini",
    "google/":      "gemini",
    "groq/":        "groq",
    "openrouter/":  "openrouter",
    "together/":    "together",
}

# Free models with notes — shown in `agentforge keys free`
FREE_MODELS: dict[str, str] = {
    "gemini/gemini-1.5-flash":    "Google Gemini Flash — FREE (1500 req/day, tool use supported)",
    "gemini/gemini-2.0-flash-exp":"Google Gemini 2.0 Flash Exp — FREE (experimental, latest)",
    "groq/llama-3.3-70b-versatile":"Groq Llama 3.3 70B — FREE tier, strong reasoning",
    "groq/llama-3.1-8b-instant":  "Groq Llama 3.1 8B Instant — FREE tier, very fast",
    "groq/mixtral-8x7b-32768":    "Groq Mixtral 8x7B — FREE tier, good for code",
}


def load_keys_to_env() -> None:
    """
    Load all saved API keys into environment variables.
    Call this at startup so LiteLLM can route to any provider.
    Also respects existing env vars (doesn't overwrite if already set).
    """
    if not _KEYS_FILE.exists():
        return
    try:
        keys: dict = json.loads(_KEYS_FILE.read_text())
    except Exception:
        return
    for provider, key in keys.items():
        if env_var := ENV_MAP.get(provider):
            os.environ.setdefault(env_var, key)


def save_key(provider: str, key: str) -> None:
    """Save an API key for a provider."""
    provider = provider.lower().strip()
    if provider not in ENV_MAP:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(ENV_MAP)}")
    keys = _load_raw()
    keys[provider] = key.strip()
    _KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEYS_FILE.write_text(json.dumps(keys, indent=2) + "\n")
    # Apply immediately to current process
    os.environ[ENV_MAP[provider]] = key.strip()


def remove_key(provider: str) -> bool:
    """Remove a saved API key. Returns True if it existed."""
    keys = _load_raw()
    if provider not in keys:
        return False
    del keys[provider]
    _KEYS_FILE.write_text(json.dumps(keys, indent=2) + "\n")
    return True


def list_keys() -> dict[str, str]:
    """Return saved keys with values masked (show only first/last 4 chars)."""
    raw = _load_raw()
    masked = {}
    for provider, key in raw.items():
        if len(key) > 8:
            masked[provider] = f"{key[:4]}...{key[-4:]}"
        else:
            masked[provider] = "****"
    return masked


def detect_provider(model_str: str) -> str | None:
    """Detect which provider a model string belongs to."""
    for prefix, provider in MODEL_PREFIX_MAP.items():
        if model_str.startswith(prefix):
            return provider
    return None   # local/Ollama


def has_key(provider: str) -> bool:
    """Check if a key is available (either saved or already in env)."""
    env_var = ENV_MAP.get(provider)
    if not env_var:
        return False
    return bool(os.environ.get(env_var))


def _load_raw() -> dict:
    if not _KEYS_FILE.exists():
        return {}
    try:
        return json.loads(_KEYS_FILE.read_text())
    except Exception:
        return {}
