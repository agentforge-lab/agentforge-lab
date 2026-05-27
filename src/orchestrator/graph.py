"""
LangGraph orchestration graph — the engine that connects all agents.

Flow:
  planner → human_checkpoint → developer → executor → tester ─┐
                 (reject→END)   ↑________________________________│ (fail, retry<3)
                                                    ↓ (pass)
                                                 security ──────┐
                                                    ↑ ___________│ (blocking, retry<3)
                                                    ↓ (pass)
                                                git_manager → END

After max_retries: → END with final_error set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from src.api.events import emit, E, get_ctx


# ── State ──────────────────────────────────────────────────────────────────

class AgentForgeState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    goal: str
    task_description: str   # refined by planner
    branch_name: str

    # ── Plan ───────────────────────────────────────────────
    plan_json: str          # PlannerAgent output, serialised

    # ── Developer ──────────────────────────────────────────
    edited_file_paths: list[str]
    developer_summary: str
    developer_error: str

    # ── Executor (syntax / import check) ───────────────────
    exec_passed: bool
    exec_output: str

    # ── Tester ─────────────────────────────────────────────
    tests_passed: bool
    test_output: str
    test_failures: list[dict]   # [{test_name, error}]

    # ── Security ───────────────────────────────────────────
    security_passed: bool
    security_report: str
    security_blocking: list[dict]

    # ── Git ────────────────────────────────────────────────
    commit_sha: str
    branch_committed: str

    # ── Control ────────────────────────────────────────────
    retry_count: int
    max_retries: int
    human_approved: bool
    complete: bool
    final_error: str

    # ── Log ────────────────────────────────────────────────
    session_log: list[str]


def _log(state: AgentForgeState, message: str) -> list[str]:
    ts = datetime.now().strftime("%H:%M:%S")
    return list(state.get("session_log", [])) + [f"[{ts}] {message}"]


# ── Agent bundle (dependency injection) ───────────────────────────────────

@dataclass
class AgentBundle:
    """All agent instances the graph nodes need. Swap out for testing."""
    planner: object
    developer: object
    executor: object
    tester: object
    security: object
    git_manager: object
    auto_approve: bool = False
    working_dir: Path = Path(".")


# ── Context helpers ────────────────────────────────────────────────────────

_SKIP_DIRS = {".venv", "__pycache__", ".agentforge", ".git", "node_modules", "dist", "build"}


def _collect_existing_files(
    working_dir: Path,
    task: str,
    max_files: int = 3,
    max_chars_per_file: int = 3000,
) -> dict[str, str]:
    """
    Collect existing Python source files from working_dir to give the developer
    codebase awareness before writing. Returns {rel_path: truncated_content}.
    Excludes: venv, cache, .agentforge, test files, __init__.py.
    """
    candidates: list[tuple[str, str]] = []

    for py_file in working_dir.rglob("*.py"):
        rel = py_file.relative_to(working_dir)
        # Skip any file whose path passes through an excluded directory
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        name = py_file.name
        if name.startswith("test_") or name == "__init__.py":
            continue
        rel_str = str(rel)
        if "tests/" in rel_str or "tests\\" in rel_str:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if content.strip():
            candidates.append((rel_str, content))

    if not candidates:
        return {}

    # Score candidates by keyword overlap with the task
    task_words = set(task.lower().split())

    def _score(item: tuple[str, str]) -> int:
        path, content = item
        text_words = set((path + " " + content[:400]).lower().split())
        return len(text_words & task_words)

    candidates.sort(key=_score, reverse=True)
    candidates = candidates[:max_files]

    result = {}
    for path, content in candidates:
        truncated = content[:max_chars_per_file]
        if len(content) > max_chars_per_file:
            truncated += "\n# ... (truncated)"
        result[path] = truncated
    return result


# ── Node factories ─────────────────────────────────────────────────────────
# Each factory closes over `bundle` and returns a plain function
# that LangGraph calls with the current state dict.

def make_planner_node(bundle: AgentBundle):
    def planner_node(state: AgentForgeState) -> dict:
        goal = state["goal"]
        emit(E.NODE_ENTERED, "planner", task=goal)

        plan = bundle.planner.plan(goal)
        plan_json = plan.to_json()

        task_description = plan.developer_brief if plan.developer_brief else goal

        slug = goal.lower()[:30].replace(" ", "-").replace("/", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        date = datetime.now().strftime("%Y%m%d")
        branch = f"agent/{slug}-{date}"

        source = "llm brief" if plan.developer_brief else "raw goal"

        if plan.developer_brief:
            emit(E.DECISION_MADE, "planner",
                 decision="Implementation spec drafted",
                 reasoning=plan.developer_brief[:400],
                 node_count=len(plan.nodes))

        emit(E.NODE_COMPLETED, "planner",
             task_description=task_description[:120],
             branch=branch,
             source=source)

        return {
            "plan_json": plan_json,
            "task_description": task_description,
            "branch_name": branch,
            "session_log": _log(state, f"planner: task set via {source} — '{task_description[:60]}'"),
        }
    return planner_node


def make_human_checkpoint_node(bundle: AgentBundle):
    def human_checkpoint_node(state: AgentForgeState) -> dict:
        brief = state.get("task_description", "")
        branch = state.get("branch_name", "")
        emit(E.NODE_ENTERED, "human_checkpoint",
             task=brief[:120],
             branch=branch,
             developer_brief=brief,
             working_dir=str(bundle.working_dir))

        if bundle.auto_approve:
            emit(E.NODE_COMPLETED, "human_checkpoint", approved=True, mode="auto",
                 developer_brief=brief, branch=branch)
            return {
                "human_approved": True,
                "session_log": _log(state, "human_checkpoint: auto-approved"),
            }

        print("\n" + "=" * 60)
        print("  AgentForge — Human Checkpoint")
        print("=" * 60)
        print(f"  Goal:   {state['goal']}")
        print(f"  Task:   {state['task_description']}")
        print(f"  Branch: {state['branch_name']}")
        print("-" * 60)

        try:
            answer = input("  Approve and start coding? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        approved = answer not in ("n", "no")
        verdict = "approved" if approved else "rejected"
        print(f"  → {verdict.upper()}")
        print("=" * 60 + "\n")

        if approved:
            emit(E.NODE_COMPLETED, "human_checkpoint", approved=True, mode="manual")
        else:
            emit(E.NODE_FAILED, "human_checkpoint", reason="rejected by user")

        return {
            "human_approved": approved,
            "session_log": _log(state, f"human_checkpoint: {verdict} by user"),
        }
    return human_checkpoint_node


def make_developer_node(bundle: AgentBundle):
    def developer_node(state: AgentForgeState) -> dict:
        retry = state.get("retry_count", 0)
        task = state["task_description"]

        emit(E.NODE_ENTERED, "developer", task=task[:120], attempt=retry + 1)

        context: dict = {}

        project_ctx_path = bundle.working_dir / ".agentforge" / "project_context.md"
        if project_ctx_path.exists():
            context["project_context"] = project_ctx_path.read_text()

        existing = _collect_existing_files(bundle.working_dir, task)
        if existing:
            context["existing_files"] = existing

        retry_reason = ""
        if retry > 0:
            test_output    = state.get("test_output", "")
            exec_output    = state.get("exec_output", "")
            security_issues = state.get("security_blocking", [])

            if test_output and not state.get("tests_passed"):
                retry_reason = "test failures"
                context["previous_error"] = (
                    f"Tests failed on attempt {retry}. Full pytest output:\n\n"
                    + test_output[:2500]
                )
            elif exec_output and not state.get("exec_passed"):
                retry_reason = "syntax error"
                context["previous_error"] = (
                    f"Syntax/import error on attempt {retry}:\n{exec_output[:800]}"
                )
            elif security_issues:
                retry_reason = "security findings"
                context["previous_error"] = (
                    f"Security scan blocked on attempt {retry}:\n"
                    + "\n".join(f"  [{f['severity']}] {f['issue_text']}" for f in security_issues)
                )

            emit(E.NODE_RETRYING, "developer",
                 attempt=retry + 1,
                 reason=retry_reason,
                 error_preview=(context.get("previous_error", "")[:200]))

        result = bundle.developer.execute(task, context)

        if result.success:
            paths = [e.file_path for e in result.edits]
            emit(E.NODE_COMPLETED, "developer",
                 files=paths,
                 summary=result.summary,
                 attempt=retry + 1)
            return {
                "edited_file_paths": paths,
                "developer_summary": result.summary,
                "developer_error": "",
                "session_log": _log(state, f"developer: wrote {len(paths)} file(s) — {result.summary}"),
            }
        else:
            emit(E.NODE_FAILED, "developer",
                 error=result.error or "Unknown error",
                 attempt=retry + 1)
            return {
                "edited_file_paths": [],
                "developer_error": result.error or "Unknown error",
                "session_log": _log(state, f"developer: failed (attempt {retry + 1}) — {result.error}"),
            }
    return developer_node


def make_executor_node(bundle: AgentBundle):
    def executor_node(state: AgentForgeState) -> dict:
        paths = state.get("edited_file_paths", [])
        emit(E.NODE_ENTERED, "executor", files=paths)

        if not paths:
            emit(E.NODE_SKIPPED, "executor", reason="no files to check")
            return {
                "exec_passed": False,
                "exec_output": "No files to check",
                "session_log": _log(state, "executor: skipped — no files"),
            }

        outputs = []
        all_passed = True

        for rel_path in paths:
            if not rel_path.endswith(".py"):
                continue
            abs_path = bundle.working_dir / rel_path
            if not abs_path.exists():
                continue
            result = bundle.executor.check_file_syntax(abs_path)
            if not result.success:
                all_passed = False
                outputs.append(f"SYNTAX ERROR in {rel_path}:\n{result.stderr}")
            else:
                outputs.append(f"OK {rel_path}")

        summary = "\n".join(outputs) if outputs else "No Python files to check"

        if all_passed:
            emit(E.NODE_COMPLETED, "executor", files_checked=len(paths), summary="all syntax OK")
        else:
            errors = [o for o in outputs if o.startswith("SYNTAX")]
            emit(E.NODE_FAILED, "executor",
                 error_count=len(errors),
                 summary=summary[:200])

        return {
            "exec_passed": all_passed,
            "exec_output": summary,
            "session_log": _log(state, f"executor: {'passed' if all_passed else 'syntax errors'} — {len(paths)} file(s)"),
        }
    return executor_node


def make_tester_node(bundle: AgentBundle):
    def tester_node(state: AgentForgeState) -> dict:
        paths = state.get("edited_file_paths", [])
        emit(E.NODE_ENTERED, "tester", files=paths)

        source_files: dict[str, str] = {}
        for rel_path in paths:
            if not rel_path.endswith(".py"):
                continue
            name = Path(rel_path).name
            if name.startswith("test_") or name == "__init__.py":
                continue
            if "tests/" in rel_path or "tests\\" in rel_path:
                continue
            abs_path = bundle.working_dir / rel_path
            if abs_path.exists():
                source_files[rel_path] = abs_path.read_text()

        if not source_files:
            emit(E.NODE_SKIPPED, "tester", reason="no Python source files")
            return {
                "tests_passed": True,
                "test_output": "No Python source files to test",
                "test_failures": [],
                "session_log": _log(state, "tester: skipped — no Python files"),
            }

        result = bundle.tester.test_edits(source_files, state.get("task_description", ""))

        failures = [
            {"test_name": f.test_name, "error": f.error}
            for f in result.failures
        ]

        emit(E.TEST_RESULT, "tester",
             total=result.total,
             passed=result.passed,
             failed=result.failed,
             success=result.success,
             failures=failures[:5])  # cap at 5 for the event payload

        status = "passed" if result.success else f"failed ({result.failed}/{result.total})"

        if result.success:
            emit(E.NODE_COMPLETED, "tester",
                 total=result.total,
                 passed=result.passed)
        else:
            emit(E.NODE_FAILED, "tester",
                 failed=result.failed,
                 total=result.total,
                 failures=failures[:3])

        return {
            "tests_passed": result.success,
            "test_output": result.raw_output,
            "test_failures": failures,
            "session_log": _log(state, f"tester: {status}"),
        }
    return tester_node


def make_security_node(bundle: AgentBundle):
    def security_node(state: AgentForgeState) -> dict:
        paths = state.get("edited_file_paths", [])
        py_paths = [p for p in paths if p.endswith(".py")]
        emit(E.NODE_ENTERED, "security", files=py_paths)

        if not py_paths:
            emit(E.NODE_SKIPPED, "security", reason="no Python files")
            return {
                "security_passed": True,
                "security_report": "No Python files to scan",
                "security_blocking": [],
                "session_log": _log(state, "security: skipped — no Python files"),
            }

        result = bundle.security.scan_files(py_paths)

        blocking = [
            {"severity": f.severity, "issue_text": f.issue_text,
             "file": f.file, "line": f.line}
            for f in result.blocking_findings()
        ]

        for finding in blocking[:5]:
            emit(E.SECURITY_FINDING, "security",
                 severity=finding["severity"],
                 issue=finding["issue_text"],
                 file=finding["file"],
                 line=finding.get("line"))

        status = "passed" if result.passed else f"BLOCKED ({result.critical_count} critical, {result.high_count} high)"

        if result.passed:
            emit(E.NODE_COMPLETED, "security",
                 files_scanned=len(py_paths),
                 findings_count=len(result.findings) if hasattr(result, "findings") else 0)
        else:
            emit(E.NODE_FAILED, "security",
                 blocking_count=len(blocking),
                 summary=status)

        return {
            "security_passed": result.passed,
            "security_report": result.format_report(),
            "security_blocking": blocking,
            "session_log": _log(state, f"security: {status}"),
        }
    return security_node


def make_git_node(bundle: AgentBundle):
    def git_node(state: AgentForgeState) -> dict:
        branch = state.get("branch_name", "agent/work")
        paths = state.get("edited_file_paths", [])
        summary = state.get("developer_summary", state["goal"][:72])
        emit(E.NODE_ENTERED, "git_manager", branch=branch, files=paths)

        # Ensure git repo exists
        if not bundle.git_manager.is_git_repo():
            init = bundle.git_manager.init_repo()
            if not init.success:
                return {
                    "final_error": f"git init failed: {init.message}",
                    "complete": True,
                    "session_log": _log(state, f"git_manager: init failed — {init.message}"),
                }

        # Switch to feature branch
        br = bundle.git_manager.ensure_branch(branch)
        if not br.success:
            return {
                "final_error": f"branch failed: {br.message}",
                "complete": True,
                "session_log": _log(state, f"git_manager: branch failed — {br.message}"),
            }

        # Stage changed files
        if paths:
            stage = bundle.git_manager.stage_files(paths)
            if not stage.success:
                return {
                    "final_error": f"stage failed: {stage.message}",
                    "complete": True,
                    "session_log": _log(state, f"git_manager: stage failed — {stage.message}"),
                }

        # Commit
        body = f"Goal: {state['goal']}\nFiles: {', '.join(paths)}"
        commit = bundle.git_manager.commit(summary, body=body, commit_type="feat")
        if not commit.success:
            return {
                "final_error": f"commit failed: {commit.message}",
                "complete": True,
                "session_log": _log(state, f"git_manager: commit failed — {commit.message}"),
            }

        emit(E.NODE_COMPLETED, "git_manager",
             commit_sha=commit.commit_sha,
             branch=branch,
             files=paths)

        return {
            "commit_sha": commit.commit_sha or "",
            "branch_committed": branch,
            "complete": True,
            "session_log": _log(state, f"git_manager: committed {commit.commit_sha} on '{branch}'"),
        }
    return git_node


# ── Routing conditions ─────────────────────────────────────────────────────

def route_after_checkpoint(state: AgentForgeState) -> str:
    return "developer" if state.get("human_approved") else END


def route_after_developer(state: AgentForgeState) -> str:
    if not state.get("developer_error"):
        return "executor"
    # Developer failed — retry if budget remains
    if state.get("retry_count", 0) < state.get("max_retries", 3) - 1:
        return "developer"
    return END


def route_after_executor(state: AgentForgeState) -> str:
    return "tester" if state.get("exec_passed", True) else "developer"


def route_after_tester(state: AgentForgeState) -> str:
    if state.get("tests_passed"):
        return "security"
    retries_left = state.get("retry_count", 0) < state.get("max_retries", 3) - 1
    return "developer" if retries_left else END


def route_after_security(state: AgentForgeState) -> str:
    if state.get("security_passed"):
        return "git_manager"
    retries_left = state.get("retry_count", 0) < state.get("max_retries", 3) - 1
    return "developer" if retries_left else END


def _increment_retry(state: AgentForgeState) -> dict:
    """Thin node that increments retry_count when looping back to developer."""
    new_count = state.get("retry_count", 0) + 1
    emit(E.NODE_RETRYING, "developer",
         attempt=new_count + 1,
         reason="pipeline retry triggered")
    return {
        "retry_count": new_count,
        "session_log": _log(state, f"retry: attempt {new_count + 1}"),
    }


# ── Graph builder ──────────────────────────────────────────────────────────

def build_graph(bundle: AgentBundle):
    """Compile and return the full agent pipeline as a LangGraph app."""

    g = StateGraph(AgentForgeState)

    # Register nodes
    g.add_node("planner",           make_planner_node(bundle))
    g.add_node("human_checkpoint",  make_human_checkpoint_node(bundle))
    g.add_node("developer",         make_developer_node(bundle))
    g.add_node("increment_retry",   _increment_retry)
    g.add_node("executor",          make_executor_node(bundle))
    g.add_node("tester",            make_tester_node(bundle))
    g.add_node("security",          make_security_node(bundle))
    g.add_node("git_manager",       make_git_node(bundle))

    # Entry point
    g.add_edge(START, "planner")
    g.add_edge("planner", "human_checkpoint")

    # Checkpoint → developer or END
    g.add_conditional_edges("human_checkpoint", route_after_checkpoint,
                            {"developer": "developer", END: END})

    # Developer → executor or retry
    g.add_conditional_edges("developer", route_after_developer,
                            {"executor": "executor", "developer": "increment_retry", END: END})
    g.add_edge("increment_retry", "developer")

    # Executor → tester or back to developer on syntax error
    g.add_conditional_edges("executor", route_after_executor,
                            {"tester": "tester", "developer": "increment_retry"})

    # Tester → security or retry
    g.add_conditional_edges("tester", route_after_tester,
                            {"security": "security", "developer": "increment_retry", END: END})

    # Security → git or retry
    g.add_conditional_edges("security", route_after_security,
                            {"git_manager": "git_manager", "developer": "increment_retry", END: END})

    # Git → done
    g.add_edge("git_manager", END)

    return g.compile()


def default_state(goal: str, max_retries: int = 3) -> dict:
    """Return a fully-initialised state dict for a new run."""
    return {
        "goal": goal,
        "task_description": "",
        "branch_name": "",
        "plan_json": "",
        "edited_file_paths": [],
        "developer_summary": "",
        "developer_error": "",
        "exec_passed": True,
        "exec_output": "",
        "tests_passed": False,
        "test_output": "",
        "test_failures": [],
        "security_passed": False,
        "security_report": "",
        "security_blocking": [],
        "commit_sha": "",
        "branch_committed": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "human_approved": False,
        "complete": False,
        "final_error": "",
        "session_log": [],
    }
