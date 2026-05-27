"""Tests for SecurityAgent — focuses on parser and graceful degradation."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.security import (
    SecurityAgent, SecurityResult, SecurityFinding,
    _parse_bandit_json, _count_by_severity,
)

BANDIT_JSON_WITH_HIGH = """{
  "results": [
    {
      "filename": "src/app.py",
      "issue_severity": "HIGH",
      "issue_confidence": "HIGH",
      "line_number": 42,
      "test_id": "B106",
      "issue_text": "Possible hardcoded password",
      "code": "password = 'hunter2'"
    }
  ],
  "metrics": {}
}"""

BANDIT_JSON_WITH_MEDIUM = """{
  "results": [
    {
      "filename": "src/utils.py",
      "issue_severity": "MEDIUM",
      "issue_confidence": "MEDIUM",
      "line_number": 10,
      "test_id": "B101",
      "issue_text": "Use of assert detected",
      "code": "assert x == 1"
    }
  ],
  "metrics": {}
}"""

BANDIT_JSON_CLEAN = '{"results": [], "metrics": {}}'
BANDIT_JSON_INVALID = "not valid json at all"


class TestParseBanditJson:
    def test_parses_high_severity(self):
        findings = _parse_bandit_json(BANDIT_JSON_WITH_HIGH)
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"
        assert findings[0].file == "src/app.py"
        assert findings[0].line == 42
        assert findings[0].issue_id == "B106"
        assert "password" in findings[0].issue_text.lower()

    def test_parses_medium(self):
        findings = _parse_bandit_json(BANDIT_JSON_WITH_MEDIUM)
        assert findings[0].severity == "MEDIUM"

    def test_clean_output(self):
        findings = _parse_bandit_json(BANDIT_JSON_CLEAN)
        assert findings == []

    def test_invalid_json(self):
        findings = _parse_bandit_json(BANDIT_JSON_INVALID)
        assert findings == []


class TestCountBySeverity:
    def test_counts_correctly(self):
        findings = [
            SecurityFinding("HIGH", "HIGH", "f.py", 1, "B1", "issue1"),
            SecurityFinding("HIGH", "MEDIUM", "f.py", 2, "B2", "issue2"),
            SecurityFinding("MEDIUM", "LOW", "f.py", 3, "B3", "issue3"),
            SecurityFinding("LOW", "LOW", "f.py", 4, "B4", "issue4"),
        ]
        counts = _count_by_severity(findings)
        assert counts["HIGH"] == 2
        assert counts["MEDIUM"] == 1
        assert counts["LOW"] == 1
        assert counts["CRITICAL"] == 0


class TestSecurityResult:
    def test_blocks_on_high(self):
        result = SecurityResult(
            passed=False, blocks_commit=True, high_count=1,
            findings=[SecurityFinding("HIGH", "HIGH", "f.py", 1, "B1", "hardcoded password")]
        )
        assert result.blocks_commit
        assert len(result.blocking_findings()) == 1

    def test_does_not_block_on_medium(self):
        result = SecurityResult(
            passed=True, blocks_commit=False, medium_count=1,
            findings=[SecurityFinding("MEDIUM", "LOW", "f.py", 1, "B1", "assert")]
        )
        assert not result.blocks_commit
        assert result.blocking_findings() == []

    def test_summary_skipped(self):
        result = SecurityResult(passed=True, blocks_commit=False, skipped=True, error="Bandit not installed")
        assert "SKIPPED" in result.summary()

    def test_summary_passed(self):
        result = SecurityResult(passed=True, blocks_commit=False, scanned_files=5)
        assert "PASSED" in result.summary()

    def test_summary_blocked(self):
        result = SecurityResult(passed=False, blocks_commit=True, high_count=2, scanned_files=3)
        assert "BLOCKED" in result.summary()

    def test_format_report_no_findings(self):
        result = SecurityResult(passed=True, blocks_commit=False, scanned_files=2)
        report = result.format_report()
        assert "PASSED" in report

    def test_finding_str(self):
        f = SecurityFinding("HIGH", "HIGH", "src/app.py", 10, "B106", "Hardcoded password")
        assert "HIGH" in str(f)
        assert "src/app.py" in str(f)
        assert "10" in str(f)


class TestSecurityAgentGracefulDegradation:
    """These tests verify the agent handles missing Bandit gracefully."""

    def test_skips_cleanly_when_bandit_missing(self, tmp_path, monkeypatch):
        # Simulate bandit not installed by patching the availability check
        agent = SecurityAgent(working_dir=tmp_path)
        monkeypatch.setattr(SecurityAgent, "_bandit_available", staticmethod(lambda: False))
        result = agent.scan_path(".")
        assert result.skipped
        assert result.passed      # skipped = non-blocking
        assert not result.blocks_commit
        assert "not installed" in result.error.lower()
