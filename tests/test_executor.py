"""Tests for ExecutorAgent — no LLM required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.executor import ExecutorAgent, ExecutionResult


@pytest.fixture
def executor(tmp_path):
    return ExecutorAgent(working_dir=tmp_path)


class TestRunCommand:
    def test_simple_success(self, executor):
        result = executor.run_command(["echo", "hello"])
        assert result.success
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_failing_command(self, executor):
        result = executor.run_command(["python3", "-c", "raise ValueError('boom')"])
        assert not result.success
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    def test_timeout(self, executor):
        result = executor.run_command(["sleep", "10"], timeout=1)
        assert not result.success
        assert result.timed_out

    def test_unknown_command(self, executor):
        result = executor.run_command(["this_command_does_not_exist_xyz"])
        assert not result.success
        assert result.exit_code == -1

    def test_blocked_command(self, executor):
        result = executor.run_command(["sh", "-c", "rm -rf /"])
        assert not result.success
        assert "Blocked" in result.stderr

    def test_duration_recorded(self, executor):
        result = executor.run_command(["echo", "hi"])
        assert result.duration_ms >= 0

    def test_command_stored(self, executor):
        result = executor.run_command(["echo", "stored"])
        assert "echo" in result.command


class TestRunPythonSnippet:
    def test_basic_snippet(self, executor):
        result = executor.run_python_snippet("print('hello from snippet')")
        assert result.success
        assert "hello from snippet" in result.stdout

    def test_snippet_with_error(self, executor):
        result = executor.run_python_snippet("x = 1/0")
        assert not result.success
        assert "ZeroDivisionError" in result.stderr

    def test_snippet_arithmetic(self, executor):
        result = executor.run_python_snippet("print(2 + 2)")
        assert result.success
        assert "4" in result.stdout

    def test_snippet_multiline(self, executor):
        code = "def add(a, b):\n    return a + b\nprint(add(3, 4))"
        result = executor.run_python_snippet(code)
        assert result.success
        assert "7" in result.stdout

    def test_tmp_file_cleaned_up(self, executor, tmp_path):
        executor.run_python_snippet("pass")
        leftover = list(tmp_path.glob(".agentforge_exec_*.py"))
        assert len(leftover) == 0


class TestSyntaxCheck:
    def test_valid_syntax(self, executor):
        result = executor.check_file_syntax(Path(__file__))
        assert result.success

    def test_format_result(self, executor):
        result = executor.run_command(["echo", "format test"])
        formatted = executor.format_result(result)
        assert "PASSED" in formatted
        assert "echo" in formatted
