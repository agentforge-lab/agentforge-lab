"""
Executor Agent — runs code in an isolated subprocess sandbox.
Phase 1: subprocess with timeout + output capture.
Phase 5+: swap in Docker/E2B for full isolation.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False
    command: str = ""


# Commands that are never allowed regardless of context
_BLOCKLIST = [
    # Disk destruction — original set
    "rm -rf /",
    "rm -rf ~",
    ":(){ :|:& };:",   # fork bomb
    "mkfs",
    "dd if=/dev/zero",
    "> /dev/sda",
    # Disk destruction — extended
    "shred",
    "wipefs",
    "fdisk",
    "parted",
    "diskutil erasedisk",
    # Network exfiltration (pipe-to-shell attacks) — match "| sh" or "|sh" anywhere
    "| sh",
    "|sh",
    "| bash",
    "|bash",
    # Reverse shells
    "nc -e",
    "ncat --exec",
    "ncat -e",
    # Privilege escalation
    "sudo ",
    "sudo\t",
    "su -",
    "doas ",
    "chmod 777 /",
    "chown -r root",
    # Database destruction
    "drop database",
    "dropdb",
    "mongodrop",
    "redis-cli flushall",
    "redis-cli flushdb",
    # Dangerous process manipulation
    "kill -9 1",
    "killall -9",
    "pkill -9 -f",
    # Cryptocurrency mining
    "xmrig",
    "minerd",
    "cryptonight",
]


def _is_blocked(command: list[str]) -> str | None:
    cmd_str = " ".join(command).lower()
    for pattern in _BLOCKLIST:
        if pattern.lower() in cmd_str:
            return pattern
    return None


class ExecutorAgent:
    """
    Runs commands and Python code snippets in a sandboxed subprocess.
    All runs are time-limited. stdout/stderr always captured.
    """

    DEFAULT_TIMEOUT = 30   # seconds for normal commands
    TEST_TIMEOUT    = 120  # seconds for test suite runs

    def __init__(self, working_dir: Path = Path(".")):
        self.working_dir = working_dir

    # ── Core runner ────────────────────────────────────────────────────────

    def run_command(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        env: dict | None = None,
    ) -> ExecutionResult:
        """Run an arbitrary command. Returns stdout/stderr/exit code."""
        blocked = _is_blocked(command)
        if blocked:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Blocked command: '{blocked}'",
                exit_code=-1,
                duration_ms=0,
                command=" ".join(command),
            )

        cwd = cwd or self.working_dir
        t0 = time.monotonic()

        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration_ms=duration_ms,
                command=" ".join(command),
            )

        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Timed out after {timeout}s",
                exit_code=-1,
                duration_ms=duration_ms,
                timed_out=True,
                command=" ".join(command),
            )

        except FileNotFoundError as e:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Command not found: {e}",
                exit_code=-1,
                duration_ms=duration_ms,
                command=" ".join(command),
            )

        except Exception as e:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_ms=duration_ms,
                command=" ".join(command),
            )

    # ── Python-specific runners ────────────────────────────────────────────

    def run_python_file(
        self,
        file_path: Path,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ExecutionResult:
        """Run a Python file using the project venv."""
        python = self._python_executable()
        return self.run_command(
            [str(python), str(file_path)],
            cwd=file_path.parent,
            timeout=timeout,
        )

    def run_python_snippet(
        self,
        code: str,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ExecutionResult:
        """
        Write code to a temp file, execute it, clean up.
        Use for quick validation of LLM-generated snippets.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=self.working_dir,
            delete=False,
            prefix=".agentforge_exec_",
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            return self.run_python_file(tmp_path, timeout=timeout)
        finally:
            tmp_path.unlink(missing_ok=True)

    def run_tests(
        self,
        test_path: str = "tests/",
        timeout: int = TEST_TIMEOUT,
        extra_args: list[str] | None = None,
    ) -> ExecutionResult:
        """Run pytest on the test suite."""
        import os
        python = self._python_executable()
        cmd = [str(python), "-m", "pytest", test_path, "-v", "--tb=short"]
        if extra_args:
            cmd.extend(extra_args)
        # Ensure root-level modules (e.g. calculator.py) are importable from tests/
        env = {**os.environ, "PYTHONPATH": str(self.working_dir)}
        return self.run_command(cmd, cwd=self.working_dir, timeout=timeout, env=env)

    def run_python_module(
        self,
        module: str,
        args: list[str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ExecutionResult:
        """Run `python -m module [args]`."""
        python = self._python_executable()
        cmd = [str(python), "-m", module] + (args or [])
        return self.run_command(cmd, cwd=self.working_dir, timeout=timeout)

    # ── Syntax check ──────────────────────────────────────────────────────

    def syntax_check(self, code: str) -> ExecutionResult:
        """Quick syntax check via `python -c 'compile(...)'` — no execution."""
        escaped = code.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        check_code = f"compile('{escaped}', '<string>', 'exec')"
        python = self._python_executable()
        return self.run_command(
            [str(python), "-c", check_code],
            timeout=5,
        )

    def check_file_syntax(self, file_path: Path) -> ExecutionResult:
        """Check a .py file for syntax errors using `python -m py_compile`."""
        python = self._python_executable()
        return self.run_command(
            [str(python), "-m", "py_compile", str(file_path)],
            timeout=5,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _python_executable(self) -> Path:
        """Return the venv Python: target project venv > current process > system."""
        import sys
        candidates = [
            self.working_dir / ".venv" / "bin" / "python3.11",
            self.working_dir / ".venv" / "bin" / "python3",
            Path(sys.executable),   # the Python that launched AgentForge (has pytest)
            Path("/opt/homebrew/bin/python3.11"),
            Path("/usr/bin/python3"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return Path("python3")

    def format_result(self, result: ExecutionResult, max_lines: int = 50) -> str:
        """Human-readable summary for logging / decision tree nodes."""
        status = "PASSED" if result.success else ("TIMEOUT" if result.timed_out else "FAILED")
        lines = [f"[{status}] {result.command} ({result.duration_ms}ms, exit {result.exit_code})"]
        if result.stdout.strip():
            out_lines = result.stdout.strip().splitlines()
            if len(out_lines) > max_lines:
                out_lines = out_lines[:max_lines] + [f"... ({len(out_lines) - max_lines} more lines)"]
            lines.append("STDOUT:\n" + "\n".join(out_lines))
        if result.stderr.strip():
            err_lines = result.stderr.strip().splitlines()
            if len(err_lines) > max_lines:
                err_lines = err_lines[:max_lines] + [f"... ({len(err_lines) - max_lines} more lines)"]
            lines.append("STDERR:\n" + "\n".join(err_lines))
        return "\n".join(lines)
