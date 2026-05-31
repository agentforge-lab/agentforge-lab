"""
Execution tools — run tests and security scans.
Reuses the existing TesterAgent and SecurityAgent implementations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.tools.registry import ToolResult

_SKIP_DIRS = {".venv", "__pycache__", ".agentforge", ".git", "node_modules"}


def run_tests(working_dir: Path) -> ToolResult:
    """Run pytest against the working directory. Returns full output."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", str(working_dir), "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(working_dir),
        )
        output = (result.stdout + result.stderr).strip()
        passed = "passed" in output.lower()

        # Extract summary line (last line with 'passed' or 'failed')
        summary = ""
        for line in reversed(output.splitlines()):
            if "passed" in line or "failed" in line or "error" in line.lower():
                summary = line.strip()
                break

        return ToolResult(
            success=result.returncode == 0,
            output=output[:6000] if len(output) > 6000 else output,
            data={"passed": passed, "summary": summary, "returncode": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="Tests timed out after 120 seconds")
    except FileNotFoundError:
        return ToolResult(success=False, output="pytest not found — is the venv active?")


def run_security_scan(working_dir: Path) -> ToolResult:
    """Run Bandit security scan on all Python source files."""
    py_files: list[str] = []
    for p in working_dir.rglob("*.py"):
        rel = p.relative_to(working_dir)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if p.name.startswith("test_"):
            continue
        py_files.append(str(p))

    if not py_files:
        return ToolResult(success=True, output="No Python source files found to scan")

    try:
        result = subprocess.run(
            ["python", "-m", "bandit", "-r", "-ll"] + py_files,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (result.stdout + result.stderr).strip()
        clean = result.returncode == 0

        return ToolResult(
            success=clean,
            output=output[:3000] if len(output) > 3000 else output,
            data={"clean": clean, "files_scanned": len(py_files)},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="Security scan timed out")
    except FileNotFoundError:
        return ToolResult(success=False, output="bandit not found — is the venv active?")
