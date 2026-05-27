"""
AgentForgeRunner — the single entry point for a full agent run.

Usage:
    runner = AgentForgeRunner(working_dir=Path("."))
    result = runner.run("build a CLI calculator in Python")
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.agents.developer import DeveloperAgent
from src.agents.executor import ExecutorAgent
from src.agents.git_manager import GitManagerAgent
from src.agents.planner import PlannerAgent
from src.agents.security import SecurityAgent
from src.agents.tester import TesterAgent
from src.llm.client import LLMClient
from src.orchestrator.graph import AgentBundle, build_graph, default_state


@dataclass
class RunResult:
    success: bool
    goal: str
    commit_sha: str | None = None
    branch: str | None = None
    tests_passed: bool = False
    security_passed: bool = False
    retry_count: int = 0
    session_log: list[str] = field(default_factory=list)
    error: str | None = None

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"\n{'='*60}",
            f"  AgentForge Run — {status}",
            f"{'='*60}",
            f"  Goal:     {self.goal}",
        ]
        if self.branch:
            lines.append(f"  Branch:   {self.branch}")
        if self.commit_sha:
            lines.append(f"  Commit:   {self.commit_sha}")
        lines += [
            f"  Tests:    {'passed' if self.tests_passed else 'failed/skipped'}",
            f"  Security: {'passed' if self.security_passed else 'failed/skipped'}",
            f"  Retries:  {self.retry_count}",
        ]
        if self.error:
            lines.append(f"  Error:    {self.error}")
        lines.append(f"{'='*60}")
        if self.session_log:
            lines.append("\n  Session log:")
            for entry in self.session_log:
                lines.append(f"    {entry}")
        return "\n".join(lines)


class AgentForgeRunner:
    """
    Orchestrates a full agent run: plan → develop → test → security → commit.

    Parameters
    ----------
    working_dir  : root of the project being built
    auto_approve : skip the human checkpoint (useful for tests and scripted runs)
    max_retries  : how many times to retry Developer on test/security failure
    llm          : LLMClient instance; defaults to hardware-profile-detected model
    """

    def __init__(
        self,
        working_dir: Path = Path("."),
        auto_approve: bool = False,
        max_retries: int = 3,
        llm: LLMClient | None = None,
    ):
        self.working_dir = working_dir
        self.auto_approve = auto_approve
        self.max_retries = max_retries
        self.llm = llm or LLMClient.from_hardware_profile()
        self._app = None   # compiled graph, built lazily

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, goal: str) -> RunResult:
        """
        Run the full agent pipeline for `goal`.
        Blocks until complete (or max_retries exceeded).
        """
        from src.api.events import emit, E
        emit(E.RUN_STARTED, "", goal=goal, working_dir=str(self.working_dir))

        app = self._get_app(goal)
        initial = default_state(goal, max_retries=self.max_retries)

        try:
            final = app.invoke(initial)
        except Exception as e:
            emit(E.RUN_FAILED, "", error=str(e)[:200])
            return RunResult(
                success=False,
                goal=goal,
                error=f"Graph execution error: {e}",
            )

        result = RunResult(
            success=final.get("complete", False) and not final.get("final_error"),
            goal=goal,
            commit_sha=final.get("commit_sha") or None,
            branch=final.get("branch_committed") or None,
            tests_passed=final.get("tests_passed", False),
            security_passed=final.get("security_passed", False),
            retry_count=final.get("retry_count", 0),
            session_log=final.get("session_log", []),
            error=final.get("final_error") or None,
        )
        # Emit through the event stream so the frontend banner works even if
        # the final WebSocket send_json in main.py races with client close.
        emit(E.RUN_COMPLETED, "",
             success=result.success,
             commit_sha=result.commit_sha,
             branch=result.branch,
             tests_passed=result.tests_passed,
             security_passed=result.security_passed,
             retry_count=result.retry_count,
             error=result.error)
        return result

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_bundle(self, goal: str = "") -> AgentBundle:
        return AgentBundle(
            planner=PlannerAgent(llm_client=self.llm),
            developer=DeveloperAgent(llm=self.llm, working_dir=self.working_dir, goal=goal),
            executor=ExecutorAgent(working_dir=self.working_dir),
            tester=TesterAgent(llm=self.llm, working_dir=self.working_dir),
            security=SecurityAgent(working_dir=self.working_dir),
            git_manager=GitManagerAgent(working_dir=self.working_dir),
            auto_approve=self.auto_approve,
            working_dir=self.working_dir,
        )

    def _get_app(self, goal: str = ""):
        if self._app is None:
            bundle = self._build_bundle(goal)
            self._app = build_graph(bundle)
        return self._app
