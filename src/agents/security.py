"""
Security Checker Agent — runs Bandit (Python SAST) on generated code.
Blocks git commit on CRITICAL or HIGH severity findings.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SecurityFinding:
    severity: str       # CRITICAL | HIGH | MEDIUM | LOW
    confidence: str     # HIGH | MEDIUM | LOW
    file: str
    line: int
    issue_id: str       # e.g. B105
    issue_text: str
    code: str = ""

    def __str__(self) -> str:
        return f"[{self.severity}/{self.confidence}] {self.file}:{self.line} — {self.issue_text} ({self.issue_id})"


@dataclass
class SecurityResult:
    passed: bool            # True = no Critical/High findings
    blocks_commit: bool     # same as not passed — kept explicit for clarity
    findings: list[SecurityFinding] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    scanned_files: int = 0
    error: str | None = None
    skipped: bool = False   # True if bandit not installed

    def summary(self) -> str:
        if self.skipped:
            return f"[SKIPPED] {self.error}"
        status = "PASSED" if self.passed else "BLOCKED"
        return (
            f"[{status}] Security scan: "
            f"{self.critical_count} critical, {self.high_count} high, "
            f"{self.medium_count} medium, {self.low_count} low "
            f"across {self.scanned_files} file(s)"
        )

    def blocking_findings(self) -> list[SecurityFinding]:
        return [f for f in self.findings if f.severity in ("CRITICAL", "HIGH")]

    def format_report(self) -> str:
        if self.skipped:
            return self.summary()
        lines = [self.summary(), ""]
        if self.blocks_commit:
            lines.append("BLOCKING issues (must fix before commit):")
            for f in self.blocking_findings():
                lines.append(f"  {f}")
                if f.code:
                    lines.append(f"    Code: {f.code.strip()}")
            lines.append("")
        if self.medium_count + self.low_count > 0:
            lines.append("Non-blocking issues:")
            for f in self.findings:
                if f.severity in ("MEDIUM", "LOW"):
                    lines.append(f"  {f}")
        return "\n".join(lines)


def _parse_bandit_json(raw: str) -> list[SecurityFinding]:
    """Parse `bandit -f json` output into SecurityFinding list."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    findings = []
    for result in data.get("results", []):
        findings.append(SecurityFinding(
            severity=result.get("issue_severity", "LOW").upper(),
            confidence=result.get("issue_confidence", "LOW").upper(),
            file=result.get("filename", ""),
            line=result.get("line_number", 0),
            issue_id=result.get("test_id", ""),
            issue_text=result.get("issue_text", ""),
            code=result.get("code", ""),
        ))
    return findings


def _count_by_severity(findings: list[SecurityFinding]) -> dict[str, int]:
    counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts


class SecurityAgent:
    """
    Runs Bandit static analysis on Python source files.
    Returns SecurityResult; blocks_commit=True if any CRITICAL or HIGH found.
    """

    def __init__(self, working_dir: Path = Path(".")):
        self.working_dir = working_dir

    # ── Public API ─────────────────────────────────────────────────────────

    def scan_path(self, path: str = "src/") -> SecurityResult:
        """
        Scan a file or directory with Bandit.
        `path` is relative to working_dir.
        """
        if not self._bandit_available():
            return SecurityResult(
                passed=True,
                blocks_commit=False,
                skipped=True,
                error=(
                    "Bandit not installed — security scan skipped. "
                    "Install: pip install bandit"
                ),
            )

        target = str(self.working_dir / path)
        cmd = ["bandit", "-r", target, "-f", "json", "-q"]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.working_dir),
            )
            # Bandit exits 1 when issues found, 0 when clean — both are valid runs
            output = proc.stdout or proc.stderr
            findings = _parse_bandit_json(output)
        except subprocess.TimeoutExpired:
            return SecurityResult(
                passed=False,
                blocks_commit=True,
                error="Bandit scan timed out after 60s",
            )
        except Exception as e:
            return SecurityResult(
                passed=False,
                blocks_commit=True,
                error=f"Bandit scan failed: {e}",
            )

        counts = _count_by_severity(findings)
        has_blocking = counts["CRITICAL"] > 0 or counts["HIGH"] > 0
        scanned = self._count_py_files(path)

        return SecurityResult(
            passed=not has_blocking,
            blocks_commit=has_blocking,
            findings=findings,
            critical_count=counts["CRITICAL"],
            high_count=counts["HIGH"],
            medium_count=counts["MEDIUM"],
            low_count=counts["LOW"],
            scanned_files=scanned,
        )

    def scan_files(self, file_paths: list[str]) -> SecurityResult:
        """Scan specific files (paths relative to working_dir)."""
        if not file_paths:
            return SecurityResult(passed=True, blocks_commit=False)
        # Write paths to a temp file list and scan each
        all_findings: list[SecurityFinding] = []
        for path in file_paths:
            result = self.scan_path(path)
            if result.skipped:
                return result
            all_findings.extend(result.findings)

        counts = _count_by_severity(all_findings)
        has_blocking = counts["CRITICAL"] > 0 or counts["HIGH"] > 0
        return SecurityResult(
            passed=not has_blocking,
            blocks_commit=has_blocking,
            findings=all_findings,
            critical_count=counts["CRITICAL"],
            high_count=counts["HIGH"],
            medium_count=counts["MEDIUM"],
            low_count=counts["LOW"],
            scanned_files=len(file_paths),
        )

    def quick_check(self, code: str) -> SecurityResult:
        """
        Scan a code string without writing to disk permanently.
        Uses a temp file, cleans up after.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=self.working_dir,
            delete=False, prefix=".agentforge_sec_"
        ) as f:
            f.write(code)
            tmp = Path(f.name)
        try:
            return self.scan_path(str(tmp.relative_to(self.working_dir)))
        finally:
            tmp.unlink(missing_ok=True)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _bandit_available() -> bool:
        try:
            subprocess.run(
                ["bandit", "--version"],
                capture_output=True, timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _count_py_files(self, path: str) -> int:
        target = self.working_dir / path
        if target.is_file():
            return 1
        return len(list(target.rglob("*.py"))) if target.is_dir() else 0
