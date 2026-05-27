"""
Memory Manager — reads and writes .agentforge/ memory files.
Keeps token budget under 8K by loading only what's needed per session.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class MemoryManager:
    """
    Manages the .agentforge/ memory directory.
    Provides structured read/write for all memory file types.
    """

    def __init__(self, base_dir: Path = Path(".agentforge")):
        self.base = base_dir
        self.base.mkdir(parents=True, exist_ok=True)

    # ── Project context ────────────────────────────────────────────────────

    def load_project_context(self) -> str:
        path = self.base / "project_context.md"
        return path.read_text() if path.exists() else ""

    def save_project_context(self, content: str) -> None:
        (self.base / "project_context.md").write_text(content)

    # ── Decisions log ──────────────────────────────────────────────────────

    def log_decision(
        self,
        decision: str,
        reason: str,
        alternatives: list[str] | None = None,
        agent: str = "Claude",
        overridden: bool = False,
        override_reason: str = "",
    ) -> None:
        path = self.base / "decisions_log.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        alts = "\n".join(f"  - {a}" for a in (alternatives or []))
        entry = f"""
## [{timestamp}] — {agent}
Decision: {decision}
Reason: {reason}
Alternatives considered:
{alts or "  None recorded"}
Overridden by human: {"yes" if overridden else "no"}
Override reason: {override_reason or "—"}
"""
        with path.open("a") as f:
            f.write(entry)

    def load_decisions_tail(self, n: int = 20) -> str:
        """Load last n decision entries to stay within token budget."""
        path = self.base / "decisions_log.md"
        if not path.exists():
            return ""
        entries = path.read_text().split("\n## ")
        tail = entries[-n:] if len(entries) > n else entries
        return "\n## ".join(tail)

    # ── Session summaries ──────────────────────────────────────────────────

    def write_session_summary(self, day: str, content: str) -> None:
        path = self.base / "session_summaries" / f"{day}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def load_recent_summaries(self, n: int = 3) -> str:
        summaries_dir = self.base / "session_summaries"
        if not summaries_dir.exists():
            return ""
        files = sorted(summaries_dir.glob("*.md"))[-n:]
        return "\n\n---\n\n".join(f.read_text() for f in files)

    # ── Cost ledger ────────────────────────────────────────────────────────

    def log_cost(
        self,
        agent: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        path = self.base / "cost_ledger.md"
        date = datetime.now().strftime("%Y-%m-%d")
        row = f"| {agent:<14} | {model:<18} | {tokens_in:>9,} | {tokens_out:>10,} | ${cost_usd:.4f} |"

        # Append to today's session block (or create it)
        existing = path.read_text() if path.exists() else ""
        session_header = f"## Session {date}"
        if session_header not in existing:
            header = f"\n{session_header}\n| Agent | Model | Tokens In | Tokens Out | Cost |\n|---|---|---|---|---|\n"
            existing += header
        existing += row + "\n"
        path.write_text(existing)

    # ── Env checklist ──────────────────────────────────────────────────────

    def update_env_var(self, variable: str, service: str, status: str) -> None:
        path = self.base / "env_checklist.md"
        existing = path.read_text() if path.exists() else self._env_checklist_header()
        # Simple append — full table management in Phase 3 (Explainer Agent)
        if variable not in existing:
            existing += f"| {variable:<25} | {service:<15} | {status:<13} | —  |\n"
            path.write_text(existing)

    @staticmethod
    def _env_checklist_header() -> str:
        return (
            "# Environment Variables\n"
            f"Generated: {datetime.now().isoformat()} | Status: 0 configured\n\n"
            "| Variable | Service | Status | Guide |\n"
            "|---|---|---|---|\n"
        )

    # ── Session context loader (8K token budget) ───────────────────────────

    def load_session_context(self) -> dict[str, str]:
        """Load all context needed at session start, within token budget."""
        return {
            "hardware_profile": (self.base / "hardware_profile.md").read_text()
                if (self.base / "hardware_profile.md").exists() else "",
            "project_context": self.load_project_context(),
            "recent_summaries": self.load_recent_summaries(n=3),
            "decisions_tail": self.load_decisions_tail(n=20),
            "env_checklist": (self.base / "env_checklist.md").read_text()
                if (self.base / "env_checklist.md").exists() else "",
        }
