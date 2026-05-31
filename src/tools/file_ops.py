"""
File operation tools — read, write, list, search.
All paths are relative to working_dir and validated to prevent traversal.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.tools.registry import ToolResult

_SKIP_DIRS = {".venv", "__pycache__", ".agentforge", ".git", "node_modules", "dist", "build"}
_MAX_READ_BYTES = 32_000   # ~8k tokens — enough for any single file


def _safe_path(working_dir: Path, rel_path: str) -> Path | None:
    """Resolve and validate path stays inside working_dir. Returns None if unsafe."""
    try:
        resolved = (working_dir / rel_path).resolve()
        resolved.relative_to(working_dir.resolve())   # raises if outside
        return resolved
    except (ValueError, Exception):
        return None


def read_file(working_dir: Path, path: str) -> ToolResult:
    abs_path = _safe_path(working_dir, path)
    if abs_path is None:
        return ToolResult(success=False, output=f"Unsafe path rejected: '{path}'")
    if not abs_path.exists():
        return ToolResult(success=False, output=f"File not found: {path}")
    if not abs_path.is_file():
        return ToolResult(success=False, output=f"Not a file: {path}")

    content = abs_path.read_text(encoding="utf-8", errors="replace")
    if len(content) > _MAX_READ_BYTES:
        content = content[:_MAX_READ_BYTES] + f"\n\n# ... (truncated at {_MAX_READ_BYTES} chars)"

    lines = content.count("\n") + 1
    return ToolResult(
        success=True,
        output=content,
        data={"path": path, "lines": lines, "size": len(content)},
    )


def write_file(working_dir: Path, path: str, content: str) -> ToolResult:
    abs_path = _safe_path(working_dir, path)
    if abs_path is None:
        return ToolResult(success=False, output=f"Unsafe path rejected: '{path}'")

    # Block writing into protected directories
    for part in abs_path.relative_to(working_dir.resolve()).parts:
        if part in _SKIP_DIRS:
            return ToolResult(success=False, output=f"Writing into '{part}/' is not allowed")

    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")

    lines = content.count("\n") + 1
    return ToolResult(
        success=True,
        output=f"Wrote {lines} lines to {path}",
        data={"path": path, "lines": lines},
    )


def list_files(working_dir: Path) -> ToolResult:
    files: list[str] = []
    for p in sorted(working_dir.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(working_dir)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        files.append(str(rel))

    if not files:
        return ToolResult(success=True, output="No files found in project directory")

    return ToolResult(
        success=True,
        output="\n".join(files),
        data={"files": files, "count": len(files)},
    )


def search_code(working_dir: Path, query: str) -> ToolResult:
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches: list[str] = []

    for p in sorted(working_dir.rglob("*.py")):
        rel = p.relative_to(working_dir)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{rel}:{i}: {line.strip()}")
        except Exception:
            continue

    if not matches:
        return ToolResult(success=True, output=f"No matches found for '{query}'")

    output = "\n".join(matches[:50])   # cap at 50 results
    if len(matches) > 50:
        output += f"\n... ({len(matches) - 50} more matches)"
    return ToolResult(success=True, output=output, data={"matches": len(matches)})
