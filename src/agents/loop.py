"""
AgentLoop — the ReAct engine that makes AgentForge truly agentic.

Instead of a fixed plan→code→test→commit pipeline, the model:
  1. Observes the project (read files, list structure)
  2. Thinks (LLM decides what to do next)
  3. Acts (calls a tool)
  4. Observes the result
  5. Repeats until it calls done() or hits MAX_STEPS

The model controls the entire flow. Python only enforces safety limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from src.tools.registry import TOOL_SCHEMAS, ToolResult
from src.tools.file_ops import read_file, write_file, list_files, search_code
from src.tools.execution import run_tests, run_security_scan
from src.tools.git_ops import git_commit

MAX_STEPS = 30          # hard cap — prevents infinite loops
MAX_COST_USD = 2.00     # hard cap on API spend per run


@dataclass
class AgentResult:
    success: bool
    summary: str = ""
    steps: int = 0
    commit_sha: str | None = None
    branch: str | None = None
    tests_passed: bool = False
    security_passed: bool = False
    error: str | None = None
    session_log: list[str] = field(default_factory=list)


@dataclass
class StepEvent:
    """Emitted for every tool call so the UI can animate the agent's reasoning."""
    step: int
    tool: str
    args: dict
    result: ToolResult
    thinking: str = ""   # model's text before the tool call


# ── System prompt ──────────────────────────────────────────────────────────

_AGENT_SYSTEM = """You are AgentForge's autonomous coding agent.

Your job: complete the given goal by reading, writing, testing, and committing code.
You have access to tools. Use them in a logical sequence.

## How to work

1. Start by understanding the project: call list_files(), then read_file() on relevant files.
2. Make the code changes needed: call write_file() for each file.
3. Run tests: call run_tests(). If tests fail, read the output carefully and fix the code.
4. Run security scan: call run_security_scan(). Fix any HIGH or CRITICAL findings.
5. Commit: call git_commit() with a clear message.
6. Signal completion: call done() with a one-sentence summary.

## Rules

- Always read before modifying — never overwrite a file you haven't read.
- When tests fail, fix the ROOT CAUSE shown in the traceback, not a symptom.
- If tests still fail after 2 fix attempts, read the test file to understand what it expects.
- Unary functions (sqrt, sin) and binary functions (add, multiply) need different call signatures in dispatchers.
- Never hardcode secrets — use os.environ.get().
- File I/O functions must accept an optional filepath parameter.
- Keep the same architectural style as existing code — don't rewrite what works.
- Call done() only when: tests pass AND security scan is clean AND code is committed.
"""


class AgentLoop:
    """
    The agentic core. Replaces the fixed pipeline for complex tasks.

    Usage:
        loop = AgentLoop(llm_client, working_dir)
        result = await loop.run(goal, send_event=callback)
    """

    def __init__(self, llm_client, working_dir: Path, goal: str = ""):
        self.llm = llm_client
        self.working_dir = working_dir
        self.goal = goal
        self._messages: list[dict] = []
        self._total_cost = 0.0
        self._steps = 0
        self._log: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        goal: str,
        send_event: Callable[[dict], Awaitable[None]] | None = None,
    ) -> AgentResult:
        self.goal = goal
        self._messages = [{"role": "user", "content": goal}]
        self._steps = 0
        self._log = []

        _emit = send_event or (lambda _: None)

        await _emit({"type": "agent_started", "node_id": "agent", "data": {"goal": goal}})

        while self._steps < MAX_STEPS:
            if self._total_cost >= MAX_COST_USD:
                return self._fail(f"Cost limit reached (${self._total_cost:.2f})")

            # ── LLM call ───────────────────────────────────────────────────
            try:
                response = self.llm.chat(
                    messages=self._messages,
                    tools=TOOL_SCHEMAS,
                    _purpose=f"agent step {self._steps + 1}",
                )
            except Exception as e:
                return self._fail(f"LLM error: {e}")

            self._total_cost += response.cost_usd
            content = response.content
            tool_calls = response.tool_calls or []

            # Model responded with text but no tool call — likely confused
            if not tool_calls:
                self._messages.append({"role": "assistant", "content": content})
                self._messages.append({
                    "role": "user",
                    "content": (
                        "You must call a tool to make progress. "
                        "If the task is complete, call done(). "
                        "Otherwise call the next appropriate tool."
                    ),
                })
                continue

            # ── Execute each tool call ─────────────────────────────────────
            assistant_msg: dict = {"role": "assistant", "content": content, "tool_calls": []}
            tool_results: list[dict] = []

            for tc in tool_calls:
                self._steps += 1
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}

                result = self._execute_tool(name, args)

                step_event: dict = {
                    "type": "agent_tool_call",
                    "node_id": "agent",
                    "data": {
                        "step": self._steps,
                        "tool": name,
                        "args": args,
                        "success": result.success,
                        "output_preview": result.output[:300],
                        "data": result.data,
                    },
                }
                await _emit(step_event)
                self._log.append(f"step {self._steps}: {name}({self._args_summary(args)}) → {'ok' if result.success else 'error'}")

                # done() — task complete
                if name == "done":
                    commit_sha = self._extract_last_commit()
                    branch     = self._extract_last_branch()
                    await _emit({
                        "type": "agent_done",
                        "node_id": "agent",
                        "data": {
                            "summary": args.get("summary", ""),
                            "steps": self._steps,
                            "commit_sha": commit_sha,
                            "branch": branch,
                        },
                    })
                    return AgentResult(
                        success=True,
                        summary=args.get("summary", ""),
                        steps=self._steps,
                        commit_sha=commit_sha,
                        branch=branch,
                        session_log=self._log,
                    )

                assistant_msg["tool_calls"].append({
                    "id": tc.get("id", f"call_{self._steps}"),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{self._steps}"),
                    "content": result.output if result.success else f"ERROR: {result.output}",
                })

            self._messages.append(assistant_msg)
            self._messages.extend(tool_results)

        return self._fail(f"Reached {MAX_STEPS} steps without completing the task")

    # ── Tool dispatch ──────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> ToolResult:
        wd = self.working_dir
        try:
            if name == "read_file":
                return read_file(wd, args["path"])
            if name == "write_file":
                return write_file(wd, args["path"], args["content"])
            if name == "list_files":
                return list_files(wd)
            if name == "search_code":
                return search_code(wd, args["query"])
            if name == "run_tests":
                return run_tests(wd)
            if name == "run_security_scan":
                return run_security_scan(wd)
            if name == "git_commit":
                return git_commit(wd, args["message"], goal=self.goal)
            if name == "done":
                return ToolResult(success=True, output=args.get("summary", "done"))
            return ToolResult(success=False, output=f"Unknown tool: {name}")
        except KeyError as e:
            return ToolResult(success=False, output=f"Missing required argument: {e}")
        except Exception as e:
            return ToolResult(success=False, output=f"Tool execution error: {e}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _fail(self, error: str) -> AgentResult:
        return AgentResult(
            success=False,
            error=error,
            steps=self._steps,
            session_log=self._log,
        )

    def _args_summary(self, args: dict) -> str:
        if "path" in args:
            return args["path"]
        if "query" in args:
            return f'"{args["query"]}"'
        if "message" in args:
            return f'"{args["message"][:40]}"'
        if "summary" in args:
            return f'"{args["summary"][:40]}"'
        return ", ".join(f"{k}={str(v)[:20]}" for k, v in args.items())

    def _extract_last_commit(self) -> str | None:
        for msg in reversed(self._messages):
            if isinstance(msg.get("content"), str) and "Committed" in msg["content"]:
                parts = msg["content"].split()
                if len(parts) >= 2:
                    return parts[1]
        return None

    def _extract_last_branch(self) -> str | None:
        for msg in reversed(self._messages):
            content = msg.get("content", "")
            if isinstance(content, str) and "branch '" in content:
                start = content.index("branch '") + 8
                end   = content.index("'", start)
                return content[start:end]
        return None
