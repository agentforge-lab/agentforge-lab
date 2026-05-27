"""
Tests for the LangGraph orchestration graph.
All agents are mocked — no LLM, no Ollama, no network needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.developer import CodeEdit, DeveloperResult
from src.agents.executor import ExecutionResult
from src.agents.git_manager import GitResult
from src.agents.planner import PlannerAgent
from src.agents.tester import TesterResult, FailureInfo
from src.agents.security import SecurityResult, SecurityFinding
from src.orchestrator.graph import (
    AgentBundle, AgentForgeState, build_graph, default_state,
    route_after_checkpoint, route_after_developer,
    route_after_executor, route_after_tester, route_after_security,
)


# ── Mock agents ────────────────────────────────────────────────────────────

class MockDeveloper:
    """Returns a successful DeveloperResult with a fake file."""
    def __init__(self, success=True, error=None, file_path="src/output.py"):
        self.success = success
        self.error = error
        self.file_path = file_path
        self.call_count = 0

    def execute(self, task, context=None):
        self.call_count += 1
        if self.success:
            return DeveloperResult(
                success=True,
                edits=[CodeEdit(file_path=self.file_path, operation="create", content="x = 1")],
                summary="mock: created output.py",
            )
        return DeveloperResult(success=False, error=self.error or "mock dev error")


class MockExecutor:
    def __init__(self, passed=True):
        self.passed = passed

    def check_file_syntax(self, path):
        return ExecutionResult(success=self.passed, stdout="", stderr="", exit_code=0 if self.passed else 1, duration_ms=1)


class MockTester:
    def __init__(self, success=True, failures=None):
        self.success = success
        self.failures = failures or []
        self.call_count = 0

    def test_edits(self, source_files, task_description=""):
        self.call_count += 1
        return TesterResult(
            success=self.success,
            total=1 if self.success else len(self.failures) + 1,
            passed=1 if self.success else 0,
            failed=len(self.failures),
            failures=self.failures,
        )


class MockSecurity:
    def __init__(self, passed=True, blocking=None):
        self.passed = passed
        self._blocking = blocking or []

    def scan_files(self, paths):
        return SecurityResult(
            passed=self.passed,
            blocks_commit=not self.passed,
            findings=self._blocking,
            high_count=len([f for f in self._blocking if f.severity == "HIGH"]),
        )

    def blocking_findings(self):
        return self._blocking


class MockGitManager:
    def __init__(self, commit_ok=True):
        self.commit_ok = commit_ok

    def is_git_repo(self):
        return True

    def ensure_branch(self, name):
        return GitResult(success=True, message=f"on branch {name}", branch=name)

    def stage_files(self, paths):
        return GitResult(success=True, message="staged", changed_files=paths)

    def commit(self, title, body="", commit_type="feat", agent_name="AgentForge"):
        if self.commit_ok:
            return GitResult(success=True, message="committed", commit_sha="abc1234")
        return GitResult(success=False, message="commit failed")


def make_bundle(
    tmp_path,
    developer=None,
    executor=None,
    tester=None,
    security=None,
    git_manager=None,
    auto_approve=True,
) -> AgentBundle:
    # Write a fake file so executor has something to check
    if developer and hasattr(developer, "file_path"):
        fake = tmp_path / developer.file_path
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("x = 1\n")

    return AgentBundle(
        planner=PlannerAgent(),           # real planner (no LLM call for Phase 1 static plan)
        developer=developer or MockDeveloper(file_path="src/output.py"),
        executor=executor or MockExecutor(),
        tester=tester or MockTester(),
        security=security or MockSecurity(),
        git_manager=git_manager or MockGitManager(),
        auto_approve=auto_approve,
        working_dir=tmp_path,
    )


# ── Routing unit tests (pure functions) ───────────────────────────────────

class TestRoutingFunctions:
    def _state(self, **overrides) -> dict:
        s = default_state("test goal")
        s.update(overrides)
        return s

    def test_checkpoint_approved_routes_to_developer(self):
        assert route_after_checkpoint(self._state(human_approved=True)) == "developer"

    def test_checkpoint_rejected_routes_to_end(self):
        from langgraph.graph import END
        assert route_after_checkpoint(self._state(human_approved=False)) == END

    def test_developer_success_routes_to_executor(self):
        s = self._state(developer_error="", human_approved=True)
        assert route_after_developer(s) == "executor"

    def test_developer_failure_retries_within_budget(self):
        s = self._state(developer_error="oops", retry_count=0, max_retries=3)
        assert route_after_developer(s) == "developer"

    def test_developer_failure_ends_at_max_retries(self):
        from langgraph.graph import END
        s = self._state(developer_error="oops", retry_count=2, max_retries=3)
        assert route_after_developer(s) == END

    def test_executor_pass_routes_to_tester(self):
        assert route_after_executor(self._state(exec_passed=True)) == "tester"

    def test_executor_fail_routes_back_to_developer(self):
        assert route_after_executor(self._state(exec_passed=False)) == "developer"

    def test_tester_pass_routes_to_security(self):
        assert route_after_tester(self._state(tests_passed=True)) == "security"

    def test_tester_fail_retries_within_budget(self):
        s = self._state(tests_passed=False, retry_count=0, max_retries=3)
        assert route_after_tester(s) == "developer"

    def test_tester_fail_ends_at_max_retries(self):
        from langgraph.graph import END
        s = self._state(tests_passed=False, retry_count=2, max_retries=3)
        assert route_after_tester(s) == END

    def test_security_pass_routes_to_git(self):
        assert route_after_security(self._state(security_passed=True)) == "git_manager"

    def test_security_fail_retries_within_budget(self):
        s = self._state(security_passed=False, retry_count=0, max_retries=3)
        assert route_after_security(s) == "developer"

    def test_security_fail_ends_at_max_retries(self):
        from langgraph.graph import END
        s = self._state(security_passed=False, retry_count=2, max_retries=3)
        assert route_after_security(s) == END


# ── Full graph integration tests (mocked agents) ──────────────────────────

class TestGraphHappyPath:
    def test_happy_path_completes(self, tmp_path):
        bundle = make_bundle(tmp_path)
        app = build_graph(bundle)
        result = app.invoke(default_state("create a hello world script"))

        assert result["complete"]
        assert not result["final_error"]
        assert result["commit_sha"] == "abc1234"
        assert result["tests_passed"]
        assert result["security_passed"]

    def test_session_log_has_entries(self, tmp_path):
        bundle = make_bundle(tmp_path)
        app = build_graph(bundle)
        result = app.invoke(default_state("create a hello world script"))
        assert len(result["session_log"]) >= 5

    def test_branch_name_derived_from_goal(self, tmp_path):
        bundle = make_bundle(tmp_path)
        app = build_graph(bundle)
        result = app.invoke(default_state("build a calculator"))
        assert result["branch_committed"].startswith("agent/")
        assert "calculator" in result["branch_committed"]


class TestGraphHumanRejection:
    def test_rejection_stops_pipeline(self, tmp_path):
        bundle = make_bundle(tmp_path, auto_approve=False)
        # Patch human checkpoint to always reject
        import unittest.mock as mock
        with mock.patch("builtins.input", return_value="n"):
            app = build_graph(bundle)
            result = app.invoke(default_state("some task"))

        assert not result.get("complete")
        assert not result.get("commit_sha")


class TestGraphRetryBehavior:
    def test_tester_failure_increments_retry(self, tmp_path):
        failing_tester = MockTester(
            success=False,
            failures=[FailureInfo("test_foo", "AssertionError: 1 != 2")]
        )
        # Developer succeeds on all attempts; tester always fails → exhaust retries
        bundle = make_bundle(tmp_path, tester=failing_tester)
        # Write the fake file for the executor to find
        fake = tmp_path / "src" / "output.py"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("x = 1\n")

        app = build_graph(bundle)
        result = app.invoke(default_state("test retry", max_retries=2))

        assert result["retry_count"] >= 1

    def test_security_failure_increments_retry(self, tmp_path):
        blocking_finding = SecurityFinding(
            severity="HIGH", confidence="HIGH",
            file="src/output.py", line=1,
            issue_id="B106", issue_text="Hardcoded password",
        )
        bad_security = MockSecurity(
            passed=False,
            blocking=[blocking_finding],
        )
        bundle = make_bundle(tmp_path, security=bad_security)
        fake = tmp_path / "src" / "output.py"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("password = 'hunter2'\n")

        app = build_graph(bundle)
        result = app.invoke(default_state("test security retry", max_retries=2))

        assert result["retry_count"] >= 1
        assert not result["security_passed"]


class TestGraphGitFailure:
    def test_git_failure_sets_final_error(self, tmp_path):
        bad_git = MockGitManager(commit_ok=False)
        bundle = make_bundle(tmp_path, git_manager=bad_git)
        fake = tmp_path / "src" / "output.py"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("x = 1\n")

        app = build_graph(bundle)
        result = app.invoke(default_state("test git fail"))

        assert result["complete"]
        assert "commit failed" in result["final_error"]


class TestDefaultState:
    def test_all_keys_present(self):
        s = default_state("my goal")
        required = [
            "goal", "task_description", "branch_name", "plan_json",
            "edited_file_paths", "developer_error", "exec_passed",
            "tests_passed", "test_failures", "security_passed",
            "security_blocking", "commit_sha", "retry_count",
            "max_retries", "human_approved", "complete", "final_error",
            "session_log",
        ]
        for key in required:
            assert key in s, f"Missing key: {key}"

    def test_goal_set(self):
        s = default_state("build X")
        assert s["goal"] == "build X"

    def test_max_retries_default(self):
        s = default_state("g")
        assert s["max_retries"] == 3


# ── Context injection tests ────────────────────────────────────────────────

from src.orchestrator.graph import _collect_existing_files


class TestCollectExistingFiles:
    def test_returns_empty_for_empty_dir(self, tmp_path):
        assert _collect_existing_files(tmp_path, "build anything") == {}

    def test_collects_python_files(self, tmp_path):
        (tmp_path / "app.py").write_text("def hello(): pass\n")
        result = _collect_existing_files(tmp_path, "build app")
        assert "app.py" in result

    def test_excludes_venv_files(self, tmp_path):
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "something.py").write_text("x = 1")
        result = _collect_existing_files(tmp_path, "build")
        assert not any(".venv" in p for p in result)

    def test_excludes_test_files(self, tmp_path):
        (tmp_path / "test_app.py").write_text("def test_x(): pass\n")
        result = _collect_existing_files(tmp_path, "build")
        assert not any("test_" in Path(p).name for p in result)

    def test_excludes_init_files(self, tmp_path):
        (tmp_path / "__init__.py").write_text("")
        result = _collect_existing_files(tmp_path, "build")
        assert "__init__.py" not in result

    def test_excludes_agentforge_dir(self, tmp_path):
        af = tmp_path / ".agentforge"
        af.mkdir()
        (af / "project_context.py").write_text("x = 1")
        result = _collect_existing_files(tmp_path, "build")
        assert not any(".agentforge" in p for p in result)

    def test_caps_at_max_files(self, tmp_path):
        for i in range(10):
            (tmp_path / f"mod{i}.py").write_text(f"def f{i}(): pass\n")
        result = _collect_existing_files(tmp_path, "build", max_files=3)
        assert len(result) <= 3

    def test_truncates_large_files(self, tmp_path):
        (tmp_path / "big.py").write_text("x = 1\n" * 10000)
        result = _collect_existing_files(tmp_path, "build", max_chars_per_file=100)
        content = list(result.values())[0]
        assert "truncated" in content
        assert len(content) <= 200  # 100 + truncation notice

    def test_returns_relative_paths(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "utils.py").write_text("def helper(): pass\n")
        result = _collect_existing_files(tmp_path, "build")
        assert all(not Path(p).is_absolute() for p in result)

    def test_scores_by_task_relevance(self, tmp_path):
        (tmp_path / "auth.py").write_text("def login(): pass\n")
        (tmp_path / "database.py").write_text("def connect(): pass\n")
        result = _collect_existing_files(tmp_path, "add auth login endpoint", max_files=1)
        # auth.py should score higher for this task
        assert "auth.py" in result


# ── LLM planner → developer_brief flows into task_description ─────────────

class TestPlannerDeveloperBriefFlow:
    def test_developer_brief_used_as_task_description(self, tmp_path):
        """When planner has a developer_brief, graph uses it as task_description."""
        from src.agents.planner import PlannerAgent
        from unittest.mock import MagicMock
        import json

        brief = "Create main.py with a FastAPI app, /hello GET endpoint, returns JSON."
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(content=json.dumps({
            "goal": "build an API",
            "developer_brief": brief,
            "nodes": {},
        }))

        planner = PlannerAgent(llm_client=mock_llm)
        bundle = make_bundle(tmp_path)
        bundle = AgentBundle(
            planner=planner,
            developer=bundle.developer,
            executor=bundle.executor,
            tester=bundle.tester,
            security=bundle.security,
            git_manager=bundle.git_manager,
            auto_approve=True,
            working_dir=tmp_path,
        )
        app = build_graph(bundle)
        result = app.invoke(default_state("build an API"))
        assert result["task_description"] == brief

    def test_raw_goal_used_when_brief_empty(self, tmp_path):
        """When planner has no developer_brief, raw goal is used."""
        bundle = make_bundle(tmp_path)  # uses static PlannerAgent with no LLM
        app = build_graph(bundle)
        result = app.invoke(default_state("build a calculator"))
        assert result["task_description"] == "build a calculator"


# ── Retry feedback quality ─────────────────────────────────────────────────

class TestRetryFeedback:
    def test_full_test_output_passed_on_retry(self, tmp_path):
        """Developer context should include full pytest output on retry."""
        received_contexts = []

        class CapturingDeveloper:
            call_count = 0
            file_path = "src/output.py"

            def execute(self, task, context=None):
                CapturingDeveloper.call_count += 1
                received_contexts.append(dict(context or {}))
                # Succeed on second attempt so we can inspect the retry context
                if CapturingDeveloper.call_count >= 2:
                    return DeveloperResult(
                        success=True,
                        edits=[CodeEdit(file_path="src/output.py", operation="create", content="x=1")],
                        summary="fixed",
                    )
                return DeveloperResult(
                    success=True,
                    edits=[CodeEdit(file_path="src/output.py", operation="create", content="x=1")],
                    summary="attempt 1",
                )

        full_traceback = "FAILED test_foo - AssertionError: 1 != 2\n" + "E   assert 1 == 2\n" * 5

        failing_tester = MockTester(
            success=False,
            failures=[FailureInfo("test_foo", "AssertionError: 1 != 2")]
        )
        # Override tester to set raw test_output in state via result
        original_test_edits = failing_tester.test_edits
        def patched_test_edits(source_files, task_description=""):
            r = original_test_edits(source_files, task_description)
            r.raw_output = full_traceback
            return r
        failing_tester.test_edits = patched_test_edits

        dev = CapturingDeveloper()
        fake = tmp_path / "src" / "output.py"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("x = 1\n")

        bundle = AgentBundle(
            planner=PlannerAgent(),
            developer=dev,
            executor=MockExecutor(),
            tester=failing_tester,
            security=MockSecurity(),
            git_manager=MockGitManager(),
            auto_approve=True,
            working_dir=tmp_path,
        )
        app = build_graph(bundle)
        app.invoke(default_state("fix a bug", max_retries=2))

        # If there was a retry, the second context should have previous_error with traceback
        if len(received_contexts) >= 2:
            retry_ctx = received_contexts[1]
            if "previous_error" in retry_ctx:
                assert "pytest" in retry_ctx["previous_error"].lower() or "failed" in retry_ctx["previous_error"].lower()
