"""
Per-agent model configuration.

Config is loaded in two layers (project overrides global):
  1. ~/.agentforge/model_config.json  — user-level, applies to all projects
  2. .agentforge/model_config.json    — project-level override

Store model names without the ollama/ prefix for local models.
Use claude-* names directly for Anthropic API models.

Examples:
  "qwen2.5-coder:7b"           → Ollama local
  "qwen2.5-coder:1.5b"         → Ollama local (small)
  "claude-haiku-4-5-20251001"  → Anthropic API (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

_GLOBAL_CONFIG = Path.home() / ".agentforge" / "model_config.json"
_PROJECT_CONFIG = Path(".agentforge") / "model_config.json"

_DEFAULT_MODEL = "qwen2.5-coder:1.5b"


@dataclass
class AgentModelConfig:
    planner:    str = _DEFAULT_MODEL
    developer:  str = _DEFAULT_MODEL
    tester:     str = _DEFAULT_MODEL
    agent_loop: str = ""              # model for agentic ReAct mode; falls back to developer
    default:    str = _DEFAULT_MODEL

    def for_agent(self, agent: str) -> str:
        """Return the model for agent, falling back to default."""
        if agent == "agent_loop":
            return self.agent_loop or self.developer or self.default
        return getattr(self, agent, self.default) or self.default

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self) -> str:
        from src.llm.keys import detect_provider
        lines = ["Per-agent model configuration:"]
        for agent in ("planner", "developer", "tester", "agent_loop"):
            model = self.for_agent(agent)
            provider = detect_provider(model)
            if provider is None:
                tag = " (local Ollama)"
            else:
                tag = f" ({provider} API)"
            if agent == "agent_loop" and not self.agent_loop:
                tag += "  ← inherits from developer"
            lines.append(f"  {agent:<12}: {model}{tag}")
        return "\n".join(lines)


def _is_api_model(model: str) -> bool:
    from src.llm.keys import detect_provider
    return detect_provider(model) is not None


def load_model_config(project_dir: Path | None = None) -> AgentModelConfig:
    """
    Load config: global defaults first, then project overrides on top.
    Returns built-in defaults if neither file exists.
    """
    config = AgentModelConfig()

    for path in [_GLOBAL_CONFIG, _project_config_path(project_dir)]:
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                _apply(config, data)
            except Exception:
                pass

    return config


def save_model_config(config: AgentModelConfig, global_scope: bool = True) -> Path:
    """Persist config to global (~/.agentforge/) or project (.agentforge/) location."""
    path = _GLOBAL_CONFIG if global_scope else _PROJECT_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    return path


def make_llm_client(model_str: str):
    """
    Build an LLMClient for any model string.

    Supported formats (LiteLLM routes automatically):
      "qwen2.5-coder:7b"            → local Ollama (legacy, auto-prefixed)
      "ollama/qwen2.5-coder:7b"     → local Ollama (explicit)
      "claude-sonnet-4-6"           → Anthropic API
      "gpt-4o"                      → OpenAI API
      "gemini/gemini-1.5-flash"     → Google Gemini (free tier)
      "groq/llama-3.1-8b-instant"   → Groq (free tier)
      "openrouter/..."              → OpenRouter
      "together/..."                → Together.ai
    """
    from src.llm.client import LLMClient
    from src.llm.keys import detect_provider, load_keys_to_env

    load_keys_to_env()   # ensure all saved keys are in env before any API call

    provider = detect_provider(model_str)
    if provider is None:
        # Local Ollama — strip prefix if present
        clean = model_str.replace("ollama/", "").strip()
        return LLMClient(local_model=clean, api_model="", prefer_local=True)

    # API model — pass raw string to LiteLLM, it handles routing
    return LLMClient(local_model="", api_model=model_str, prefer_local=False)


# ── Ollama detection ───────────────────────────────────────────────────────

def detect_ollama_models() -> list[str]:
    """Query Ollama for installed models. Returns [] if Ollama isn't running."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def suggest_config(available: list[str]) -> AgentModelConfig:
    """
    Given installed Ollama model names, suggest per-agent assignments.
    Strategy: best code model → developer + planner; smallest → tester.
    """
    if not available:
        return AgentModelConfig()

    # Score by parameter count hint in model name
    def _size(name: str) -> float:
        n = name.lower()
        for tok, score in [
            ("671b", 671), ("70b", 70), ("32b", 32), ("22b", 22), ("14b", 14),
            ("13b", 13), ("8b", 8), ("7b", 7), ("3b", 3), ("1.5b", 1.5), ("0.5b", 0.5),
        ]:
            if tok in n:
                return score
        return 2.0   # unknown — treat as small

    # Prefer coding-specialised models for developer/tester
    def _is_code_model(name: str) -> bool:
        n = name.lower()
        return any(kw in n for kw in ("coder", "code", "codellama", "deepseek-coder", "starcoder"))

    ranked = sorted(available, key=_size, reverse=True)
    code_ranked = sorted([m for m in available if _is_code_model(m)], key=_size, reverse=True)

    best_code = code_ranked[0] if code_ranked else ranked[0]
    best_any  = ranked[0]
    smallest  = sorted(available, key=_size)[0]

    return AgentModelConfig(
        planner=best_any,         # planner benefits from strongest reasoning
        developer=best_code,      # developer needs best code model
        tester=smallest,          # tester is less critical; smallest saves VRAM
        default=best_code,
    )


# ── Internal ───────────────────────────────────────────────────────────────

def _project_config_path(project_dir: Path | None) -> Path:
    if project_dir:
        return project_dir / ".agentforge" / "model_config.json"
    return _PROJECT_CONFIG


def _apply(config: AgentModelConfig, data: dict) -> None:
    for field in ("planner", "developer", "tester", "agent_loop", "default"):
        if data.get(field):
            setattr(config, field, data[field])
