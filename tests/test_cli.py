"""
Tests for the AgentForge CLI.
Uses Click's CliRunner — no real LLM, no Ollama, no git.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import cli
from src.orchestrator.runner import RunResult


# ── fixtures ───────────────────────────────────────────────────────────────

def _success_result(goal="test goal") -> RunResult:
    return RunResult(
        success=True,
        goal=goal,
        commit_sha="abc1234",
        branch="agent/test-goal",
        tests_passed=True,
        security_passed=True,
        retry_count=0,
        session_log=["[planner] planned", "[git] committed"],
    )


def _failure_result(goal="test goal") -> RunResult:
    return RunResult(
        success=False,
        goal=goal,
        error="Developer exhausted retries",
    )


# ── version ────────────────────────────────────────────────────────────────

class TestVersion:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ── run command ────────────────────────────────────────────────────────────

class TestRunCommand:
    def test_run_success_exits_0(self, tmp_path):
        runner = CliRunner()
        with patch("src.cli.AgentForgeRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run.return_value = _success_result()
            with patch("src.cli._stream_run", return_value=_success_result()):
                result = runner.invoke(cli, ["run", "build a calculator", "--working-dir", str(tmp_path)])
        assert result.exit_code == 0

    def test_run_failure_exits_1(self, tmp_path):
        runner = CliRunner()
        with patch("src.cli._stream_run", return_value=_failure_result()):
            with patch("src.cli.AgentForgeRunner"):
                result = runner.invoke(cli, ["run", "build a calculator", "--working-dir", str(tmp_path)])
        assert result.exit_code == 1

    def test_run_shows_goal(self, tmp_path):
        runner = CliRunner()
        with patch("src.cli._stream_run", return_value=_success_result("my special goal")):
            with patch("src.cli.AgentForgeRunner"):
                result = runner.invoke(cli, ["run", "my special goal", "--working-dir", str(tmp_path)])
        assert "my special goal" in result.output

    def test_run_shows_working_dir(self, tmp_path):
        runner = CliRunner()
        with patch("src.cli._stream_run", return_value=_success_result()):
            with patch("src.cli.AgentForgeRunner"):
                result = runner.invoke(cli, ["run", "goal", "--working-dir", str(tmp_path)])
        assert str(tmp_path) in result.output

    def test_run_bad_working_dir_exits_1(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "goal", "--working-dir", "/nonexistent/path/xyz"])
        assert result.exit_code == 1
        assert "does not exist" in result.output.lower() or result.exit_code == 1

    def test_run_auto_approve_flag_passed(self, tmp_path):
        runner = CliRunner()
        created_runner = None

        def capture_runner(*args, **kwargs):
            nonlocal created_runner
            mock = MagicMock()
            mock.run.return_value = _success_result()
            mock._get_app.return_value = MagicMock()
            created_runner = kwargs
            return mock

        with patch("src.cli.AgentForgeRunner", side_effect=capture_runner):
            with patch("src.cli._stream_run", return_value=_success_result()):
                runner.invoke(cli, ["run", "goal", "--auto-approve", "--working-dir", str(tmp_path)])

        assert created_runner is not None
        assert created_runner.get("auto_approve") is True

    def test_run_max_retries_passed(self, tmp_path):
        runner = CliRunner()
        created_runner = None

        def capture_runner(*args, **kwargs):
            nonlocal created_runner
            mock = MagicMock()
            mock._get_app.return_value = MagicMock()
            created_runner = kwargs
            return mock

        with patch("src.cli.AgentForgeRunner", side_effect=capture_runner):
            with patch("src.cli._stream_run", return_value=_success_result()):
                runner.invoke(cli, ["run", "goal", "--max-retries", "5", "--working-dir", str(tmp_path)])

        assert created_runner is not None
        assert created_runner.get("max_retries") == 5

    def test_dry_run_no_runner_created(self, tmp_path):
        runner = CliRunner()
        mock_plan = MagicMock()
        mock_plan.task_description = "do something"
        mock_plan.branch_name = "agent/do-something"

        with patch("src.cli.AgentForgeRunner") as MockRunner:
            with patch("src.cli.PlannerAgent") as MockPlanner:
                MockPlanner.return_value.plan.return_value = mock_plan
                result = runner.invoke(cli, ["run", "dry goal", "--dry-run", "--working-dir", str(tmp_path)])

        MockRunner.assert_not_called()
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower() or "Dry" in result.output

    def test_no_stream_uses_runner_run(self, tmp_path):
        runner = CliRunner()
        with patch("src.cli.AgentForgeRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run.return_value = _success_result()
            result = runner.invoke(cli, ["run", "goal", "--no-stream", "--working-dir", str(tmp_path)])

        instance.run.assert_called_once_with("goal")


# ── init command ───────────────────────────────────────────────────────────

class TestInitCommand:
    def test_init_writes_profile(self, tmp_path):
        runner = CliRunner()
        mock_profile = MagicMock()
        mock_profile.chip_name = "Apple M5"
        mock_profile.ram_gb = 16.0
        mock_profile.effective_vram_gb = 9.6
        mock_profile.recommended_model = "qwen2.5-coder:7b"
        mock_profile.has_api_key = False

        with patch("src.cli.detect_hardware", return_value=mock_profile):
            with patch("src.cli.write_hardware_profile") as mock_write:
                result = runner.invoke(cli, ["init", "--working-dir", str(tmp_path)])

        assert result.exit_code == 0
        mock_write.assert_called_once()
        assert "Apple M5" in result.output

    def test_init_shows_model_tier(self, tmp_path):
        runner = CliRunner()
        mock_profile = MagicMock()
        mock_profile.chip_name = "Apple M5"
        mock_profile.ram_gb = 16.0
        mock_profile.effective_vram_gb = 9.6
        mock_profile.recommended_model = "qwen2.5-coder:7b"
        mock_profile.has_api_key = False

        with patch("src.cli.detect_hardware", return_value=mock_profile):
            with patch("src.cli.write_hardware_profile"):
                result = runner.invoke(cli, ["init", "--working-dir", str(tmp_path)])

        assert "qwen2.5-coder:7b" in result.output

    def test_init_creates_agentforge_dir(self, tmp_path):
        runner = CliRunner()
        mock_profile = MagicMock()
        mock_profile.chip_name = "M5"
        mock_profile.ram_gb = 16.0
        mock_profile.effective_vram_gb = 9.6
        mock_profile.recommended_model = "qwen2.5-coder:7b"
        mock_profile.has_api_key = False

        with patch("src.cli.detect_hardware", return_value=mock_profile):
            with patch("src.cli.write_hardware_profile"):
                runner.invoke(cli, ["init", "--working-dir", str(tmp_path)])

        assert (tmp_path / ".agentforge").exists()


# ── status command ─────────────────────────────────────────────────────────

class TestStatusCommand:
    def test_status_no_profile(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--working-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "not found" in result.output or "agentforge init" in result.output

    def test_status_with_profile(self, tmp_path):
        profile_dir = tmp_path / ".agentforge"
        profile_dir.mkdir()
        (profile_dir / "hardware_profile.md").write_text("Chip: Apple M5\nRAM: 16 GB\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--working-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Apple M5" in result.output

    def test_status_lists_session_summaries(self, tmp_path):
        profile_dir = tmp_path / ".agentforge"
        summaries_dir = profile_dir / "session_summaries"
        summaries_dir.mkdir(parents=True)
        (summaries_dir / "day-01.md").write_text("# Session 1\n")
        (summaries_dir / "day-02.md").write_text("# Session 2\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--working-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "day-01.md" in result.output or "day-02.md" in result.output

    def test_status_bad_working_dir(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--working-dir", "/does/not/exist"])
        assert result.exit_code == 1
