"""
Tester Agent — writes and runs pytest tests for Developer-generated code.
Returns structured pass/fail results that feed back into the Developer retry loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.developer import DeveloperAgent, CodeEdit
from src.agents.executor import ExecutorAgent, ExecutionResult
from src.llm.client import LLMClient
from src.llm.prompts import TESTER_SYSTEM


@dataclass
class FailureInfo:
    test_name: str
    error: str
    file: str = ""


@dataclass
class TesterResult:
    success: bool          # True only if all tests pass
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    failures: list[FailureInfo] = field(default_factory=list)
    coverage_pct: float | None = None
    test_file: str = ""
    raw_output: str = ""
    wrote_tests: bool = False
    error: str | None = None   # agent-level error (not test failures)

    def format_for_developer(self) -> str:
        """Format failures as context for Developer agent retry."""
        if self.success:
            return f"All {self.total} tests passed."
        lines = [f"{self.failed} of {self.total} tests failed:\n"]
        for f in self.failures:
            lines.append(f"  FAILED {f.test_name}\n  {f.error}\n")
        return "\n".join(lines)

    @property
    def blocks_developer(self) -> bool:
        return not self.success and self.total > 0


# ── Pytest output parser ───────────────────────────────────────────────────

# Matches both  tests/file.py::test_func  and  tests/file.py::Class::test_method
_RESULT_LINE = re.compile(
    r"(tests/\S+\.py)::(\S+)\s+(PASSED|FAILED|ERROR)"
)
# Separate regexes per counter — pytest prints them in any order (failed, error, passed)
_PASSED_COUNT  = re.compile(r"(\d+) passed")
_FAILED_COUNT  = re.compile(r"(\d+) failed")
_ERROR_COUNT   = re.compile(r"(\d+) error")
_FAILED_LINE   = re.compile(r"FAILED (\S+) - (.+)")
_COVERAGE_LINE = re.compile(r"TOTAL\s+\d+\s+\d+\s+(\d+)%")


def parse_pytest_output(output: str) -> TesterResult:
    """Parse raw pytest -v --tb=short output into a TesterResult."""
    failures: list[FailureInfo] = []
    passed = failed = errors = 0

    for m in _RESULT_LINE.finditer(output):
        status = m.group(3)
        if status == "PASSED":
            passed += 1
        elif status == "FAILED":
            failed += 1
        elif status == "ERROR":
            errors += 1

    # Collect failure messages from FAILED ... - ... lines
    for m in _FAILED_LINE.finditer(output):
        full_name = m.group(1)
        error_msg = m.group(2).strip()
        parts = full_name.split("::")
        failures.append(FailureInfo(
            test_name=parts[-1] if len(parts) > 1 else full_name,
            error=error_msg,
            file=parts[0] if parts else "",
        ))

    # Fall back to summary line counts if per-line parsing found nothing
    if passed == 0 and failed == 0 and errors == 0:
        # Use only the final summary segment to avoid false matches
        summary = output.rsplit("=", 1)[-1]
        m_p = _PASSED_COUNT.search(summary)
        m_f = _FAILED_COUNT.search(summary)
        m_e = _ERROR_COUNT.search(summary)
        passed = int(m_p.group(1)) if m_p else 0
        failed = int(m_f.group(1)) if m_f else 0
        errors = int(m_e.group(1)) if m_e else 0

    coverage_pct: float | None = None
    cm = _COVERAGE_LINE.search(output)
    if cm:
        coverage_pct = float(cm.group(1))

    total = passed + failed + errors
    return TesterResult(
        success=(failed == 0 and errors == 0 and total > 0),
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        failures=failures,
        coverage_pct=coverage_pct,
        raw_output=output,
    )


def _derive_test_path(source_path: str) -> str:
    """src/utils/math.py  →  tests/test_math.py"""
    stem = Path(source_path).stem
    return f"tests/test_{stem}.py"


class TesterAgent:
    """
    Given source files produced by the Developer agent:
    1. Uses LLM to write pytest tests
    2. Writes tests to tests/ directory
    3. Runs them via ExecutorAgent
    4. Returns structured TesterResult
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        working_dir: Path = Path("."),
    ):
        self.llm = llm or LLMClient()
        self.working_dir = working_dir
        self._developer = DeveloperAgent(llm=self.llm, working_dir=working_dir)
        self._executor = ExecutorAgent(working_dir=working_dir)

    # ── Public API ─────────────────────────────────────────────────────────

    def test_edits(
        self,
        source_files: dict[str, str],
        task_description: str = "",
    ) -> TesterResult:
        """
        Write tests for the given source files, run them, return results.
        `source_files`: {relative_path: file_content}
        """
        if not source_files:
            return TesterResult(success=False, error="No source files provided")

        # Step 1: generate tests via LLM
        test_code_result = self._generate_tests(source_files, task_description)
        if not test_code_result.success:
            return TesterResult(
                success=False,
                error=f"Test generation failed: {test_code_result.error}",
            )

        # Step 2: write test files to disk
        self._developer.apply_edits(test_code_result.edits)
        test_files = [e.file_path for e in test_code_result.edits]

        # Step 3: run the tests
        run_result = self._executor.run_tests(
            test_path=" ".join(test_files) if test_files else "tests/",
        )

        # Step 4: parse output
        result = parse_pytest_output(run_result.stdout + run_result.stderr)
        result.raw_output = run_result.stdout + run_result.stderr
        result.test_file = test_files[0] if test_files else ""
        result.wrote_tests = True

        if run_result.timed_out:
            result.success = False
            result.error = "Test run timed out"

        return result

    def run_existing_tests(self, test_path: str = "tests/") -> TesterResult:
        """Run the existing test suite without writing new tests."""
        run_result = self._executor.run_tests(test_path=test_path)
        result = parse_pytest_output(run_result.stdout + run_result.stderr)
        result.raw_output = run_result.stdout + run_result.stderr
        return result

    def run_and_report(self, test_path: str = "tests/") -> str:
        """Run tests and return a human-readable report string."""
        result = self.run_existing_tests(test_path)
        status = "PASSED" if result.success else "FAILED"
        lines = [f"[{status}] {result.passed}/{result.total} tests passed"]
        if result.coverage_pct is not None:
            lines.append(f"Coverage: {result.coverage_pct}%")
        for f in result.failures:
            lines.append(f"  ✗ {f.test_name}: {f.error}")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────

    def _generate_tests(self, source_files: dict[str, str], task: str):
        """Ask the LLM to write tests for the given source files."""
        file_sections = []
        for path, content in source_files.items():
            file_sections.append(f"## File: {path}\n```python\n{content}\n```")

        user_prompt = "\n\n".join(
            file_sections
            + ([f"## Goal\n{task}"] if task else [])
            + [
                "## Task\n"
                "Write pytest tests for the source code above.\n\n"
                "CRITICAL RULES:\n"
                "- Read each function body carefully — test the ACTUAL behaviour, not what you wish it did.\n"
                "- If a function returns a value on error (e.g. a string or None), assert that return value.\n"
                "- ONLY use `pytest.raises(...)` if the source code actually contains `raise` for that case.\n"
                "- Test every public function: happy path + the edge/error cases shown in the source.\n"
                "- Use the <agentforge_edits> format to write the test files."
            ]
        )

        return self._developer.dry_run(
            task=user_prompt,
            context={},
        )
