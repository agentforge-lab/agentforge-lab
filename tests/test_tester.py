"""Tests for TesterAgent — focuses on output parser (no LLM required)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.tester import parse_pytest_output, TesterResult, FailureInfo, _derive_test_path


PASSING_OUTPUT = """
tests/test_math.py::TestAdd::test_add_integers PASSED            [ 33%]
tests/test_math.py::TestAdd::test_add_floats PASSED              [ 66%]
tests/test_math.py::TestAdd::test_add_negative PASSED            [100%]

========================= 3 passed in 0.08s ===========================
"""

FAILING_OUTPUT = """
tests/test_math.py::TestAdd::test_add_integers PASSED            [ 50%]
tests/test_math.py::TestAdd::test_add_wrong FAILED               [100%]

FAILED tests/test_math.py::TestAdd::test_add_wrong - AssertionError: assert 3 == 99

========================= 1 failed, 1 passed in 0.12s =================
"""

MIXED_OUTPUT = """
tests/test_math.py::test_add PASSED                              [ 25%]
tests/test_math.py::test_subtract PASSED                         [ 50%]
tests/test_math.py::test_divide FAILED                           [ 75%]
tests/test_math.py::test_multiply ERROR                          [100%]

FAILED tests/test_math.py::test_divide - ZeroDivisionError: division by zero

========================= 1 failed, 1 error, 2 passed in 0.15s ========
"""

EMPTY_OUTPUT = ""

COVERAGE_OUTPUT = """
tests/test_math.py::test_add PASSED                              [100%]

----------- coverage: platform darwin, python 3.11 ----------
Name              Stmts   Miss  Cover
-------------------------------------
src/utils/math.py     8      2    75%
-------------------------------------
TOTAL                 8      2    75%

========================= 1 passed in 0.22s ===========================
"""


class TestParsePytestOutput:
    def test_all_passing(self):
        r = parse_pytest_output(PASSING_OUTPUT)
        assert r.success
        assert r.passed == 3
        assert r.failed == 0
        assert r.total == 3
        assert r.failures == []

    def test_with_failure(self):
        r = parse_pytest_output(FAILING_OUTPUT)
        assert not r.success
        assert r.passed == 1
        assert r.failed == 1
        assert len(r.failures) == 1
        assert r.failures[0].test_name == "test_add_wrong"
        assert "AssertionError" in r.failures[0].error

    def test_mixed_results(self):
        r = parse_pytest_output(MIXED_OUTPUT)
        assert not r.success
        assert r.passed == 2
        assert r.failed == 1
        assert r.errors == 1
        assert r.total == 4

    def test_empty_output(self):
        r = parse_pytest_output(EMPTY_OUTPUT)
        assert not r.success
        assert r.total == 0

    def test_coverage_parsed(self):
        r = parse_pytest_output(COVERAGE_OUTPUT)
        assert r.success
        assert r.coverage_pct == 75.0

    def test_no_coverage_returns_none(self):
        r = parse_pytest_output(PASSING_OUTPUT)
        assert r.coverage_pct is None


class TestTesterResult:
    def test_format_for_developer_on_success(self):
        r = TesterResult(success=True, total=3, passed=3)
        msg = r.format_for_developer()
        assert "3" in msg
        assert "passed" in msg

    def test_format_for_developer_on_failure(self):
        r = TesterResult(
            success=False, total=2, passed=1, failed=1,
            failures=[FailureInfo("test_foo", "AssertionError: 1 != 2", "tests/test_foo.py")]
        )
        msg = r.format_for_developer()
        assert "test_foo" in msg
        assert "AssertionError" in msg

    def test_blocks_developer_when_failures(self):
        r = TesterResult(success=False, total=1, failed=1)
        assert r.blocks_developer

    def test_does_not_block_on_success(self):
        r = TesterResult(success=True, total=1, passed=1)
        assert not r.blocks_developer

    def test_does_not_block_on_zero_tests(self):
        r = TesterResult(success=False, total=0)
        assert not r.blocks_developer


class TestDeriveTestPath:
    def test_src_file(self):
        assert _derive_test_path("src/utils/math.py") == "tests/test_math.py"

    def test_nested_file(self):
        assert _derive_test_path("src/agents/developer.py") == "tests/test_developer.py"

    def test_root_file(self):
        assert _derive_test_path("calculator.py") == "tests/test_calculator.py"
