"""
Git tool — stage all changes and commit to a feature branch.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

from src.tools.registry import ToolResult


def git_commit(working_dir: Path, message: str, goal: str = "") -> ToolResult:
    """Stage all modified/new files and commit on a feature branch."""

    def _run(args: list[str]) -> tuple[int, str]:
        r = subprocess.run(args, capture_output=True, text=True, cwd=str(working_dir))
        return r.returncode, (r.stdout + r.stderr).strip()

    # Init repo if needed
    if not (working_dir / ".git").exists():
        code, out = _run(["git", "init"])
        if code != 0:
            return ToolResult(success=False, output=f"git init failed: {out}")
        _run(["git", "config", "user.email", "agent@agentforge.local"])
        _run(["git", "config", "user.name",  "AgentForge"])

    # Create/switch to feature branch
    slug = re.sub(r"[^a-z0-9]+", "-", (goal or message).lower())[:30].strip("-")
    date  = datetime.now().strftime("%Y%m%d")
    branch = f"agent/{slug}-{date}"

    code, out = _run(["git", "checkout", "-b", branch])
    if code != 0:
        # Branch may already exist
        _run(["git", "checkout", branch])

    # Stage everything
    code, out = _run(["git", "add", "-A"])
    if code != 0:
        return ToolResult(success=False, output=f"git add failed: {out}")

    # Check if there's anything to commit
    code, status = _run(["git", "status", "--porcelain"])
    if not status:
        return ToolResult(success=True, output="Nothing to commit — working tree is clean", data={"branch": branch})

    # Commit
    code, out = _run(["git", "commit", "-m", message])
    if code != 0:
        return ToolResult(success=False, output=f"git commit failed: {out}")

    # Get SHA
    _, sha = _run(["git", "rev-parse", "--short", "HEAD"])

    return ToolResult(
        success=True,
        output=f"Committed {sha} on branch '{branch}'",
        data={"branch": branch, "commit_sha": sha, "message": message},
    )
