"""
Event bus for AgentForge's decision graph.

Uses ContextVar so any code in the call stack can emit events without
needing the callback threaded through every constructor.

Thread safety: the ContextVar is set inside the worker thread before the
pipeline starts, so LLM calls and node functions emit directly into the
thread's ContextVar. loop.call_soon_threadsafe() crosses back to asyncio.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


# ── Event types ────────────────────────────────────────────────────────────

class E:
    RUN_STARTED      = "run_started"
    RUN_COMPLETED    = "run_completed"
    RUN_FAILED       = "run_failed"

    NODE_ENTERED     = "node_entered"
    NODE_COMPLETED   = "node_completed"
    NODE_FAILED      = "node_failed"
    NODE_RETRYING    = "node_retrying"
    NODE_SKIPPED     = "node_skipped"

    LLM_CALL_STARTED   = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    LLM_CALL_FAILED    = "llm_call_failed"

    DECISION_MADE    = "decision_made"
    TEST_RESULT      = "test_result"
    SECURITY_FINDING = "security_finding"


# ── Run context ────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    run_id: str
    emit_cb: Callable[[dict], None]
    current_node: str = ""
    _seq: int = field(default=0, init=False, repr=False)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def set_node(self, node_id: str) -> None:
        self.current_node = node_id

    def emit(self, event_type: str, node_id: str = "", **data) -> None:
        # Track the current node so LLM calls emitted without a node_id
        # (e.g. from LLMClient.chat) inherit the active node automatically.
        if node_id and event_type == E.NODE_ENTERED:
            self.current_node = node_id
        event = {
            "type":      event_type,
            "node_id":   node_id or self.current_node,
            "run_id":    self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seq":       self._next_seq(),
            "data":      {k: v for k, v in data.items() if v is not None},
        }
        try:
            self.emit_cb(event)
        except Exception:
            pass  # never let event emission break the pipeline


# ── Module-level ContextVar ────────────────────────────────────────────────

_run_ctx: ContextVar[RunContext | None] = ContextVar("agentforge_run_ctx", default=None)


def get_ctx() -> RunContext | None:
    return _run_ctx.get()


def emit(event_type: str, node_id: str = "", **data) -> None:
    """Emit an event if a run is active. No-op otherwise."""
    ctx = _run_ctx.get()
    if ctx:
        ctx.emit(event_type, node_id, **data)


def new_run_context(emit_cb: Callable[[dict], None]) -> tuple[RunContext, object]:
    """
    Create and install a fresh RunContext for a new pipeline run.
    Returns (context, token) — call `_run_ctx.reset(token)` when done.
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ctx = RunContext(run_id=run_id, emit_cb=emit_cb)
    token = _run_ctx.set(ctx)
    return ctx, token
