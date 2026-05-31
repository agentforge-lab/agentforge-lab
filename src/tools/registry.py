"""
Tool registry — defines every tool the agent can call and the schema the LLM sees.

Each tool is an OpenAI-compatible function definition. LiteLLM converts this
format automatically for both Anthropic and Ollama backends.

Adding a new tool:
  1. Implement it in the appropriate module (file_ops, execution, git_ops, control)
  2. Add its schema to TOOL_SCHEMAS below
  3. Add its callable to TOOL_IMPLEMENTATIONS in loop.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Returned by every tool implementation. success=False surfaces as an error to the model."""
    success: bool
    output: str               # shown to the model as the tool result
    data: dict[str, Any] = field(default_factory=dict)   # structured data for the UI


# ── Schemas (what the LLM sees) ────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from project root, e.g. 'app.py' or 'src/auth.py'"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file with new content. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Relative path to write to"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all source files in the project (excludes tests, venv, cache).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a string or pattern across all project source files. Returns file paths and matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for (case-insensitive)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the full pytest test suite. Returns pass/fail counts and any failure tracebacks.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_security_scan",
            "description": "Run Bandit security scan on all Python source files. Returns any HIGH or CRITICAL findings.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all modified files and create a git commit on a new feature branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message summarising what was changed and why"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the task is complete. Call this when all tests pass, security is clean, and the code is committed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One-sentence summary of what was accomplished"},
                },
                "required": ["summary"],
            },
        },
    },
]
