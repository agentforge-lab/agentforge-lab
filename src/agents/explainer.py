"""
Explainer Agent — reads existing source files and produces a dense codebase
summary for the Developer Agent to use as context when modifying a project.

Only fires when _collect_existing_files finds files in the working directory.
For fresh greenfield runs it is never called.
"""

from __future__ import annotations

from src.llm.client import LLMClient
from src.llm.prompts import EXPLAINER_SYSTEM


class ExplainerAgent:
    """
    Summarises an existing codebase in plain English.
    Output is injected into the developer's project_context so it understands
    the project structure before writing new or modified code.
    """

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def explain(self, files: dict[str, str], goal: str = "") -> str:
        """
        files: {rel_path: file_content} — the existing source files
        goal:  the task the developer will perform (focuses the summary)
        Returns a plain-text codebase summary, or "" on any failure.
        """
        if not files:
            return ""

        user = self._build_prompt(files, goal)
        try:
            response = self.llm.complete(
                EXPLAINER_SYSTEM,
                user,
                _purpose="explainer: codebase summary",
            )
            return response.content.strip()
        except Exception:
            return ""  # summary is optional — never block the pipeline

    def _build_prompt(self, files: dict[str, str], goal: str) -> str:
        parts: list[str] = []
        if goal:
            parts.append(f"Task the developer will perform: {goal}")
        for path, content in files.items():
            truncated = content[:4000]
            suffix = "\n# ... (truncated)" if len(content) > 4000 else ""
            parts.append(f"File: {path}\n```\n{truncated}{suffix}\n```")
        return "\n\n".join(parts)
