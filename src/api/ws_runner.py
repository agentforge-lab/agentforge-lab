"""
Async WebSocket-aware pipeline runner.

Bridges the sync LangGraph pipeline (running in a ThreadPoolExecutor) to an
async WebSocket using a thread-safe asyncio.Queue + asyncio.Event.

Drain pattern:
  - Worker thread puts events via loop.call_soon_threadsafe → queue.put_nowait
  - Async task awaits queue.get() with a 1s timeout
  - Worker signals done via loop.call_soon_threadsafe → done.set
  - After done fires, we do one final drain to catch any last events
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Awaitable, Callable

from src.api.events import _run_ctx, new_run_context
from src.orchestrator.runner import AgentForgeRunner, RunResult


async def run_pipeline_ws(
    goal: str,
    working_dir: Path,
    auto_approve: bool,
    max_retries: int,
    send_event: Callable[[dict], Awaitable[None]],
) -> RunResult:
    """
    Run the AgentForge pipeline in a thread, streaming every emitted event
    to `send_event` in real time.

    Python 3.11's run_in_executor does NOT copy the current contextvars context
    into the thread, so we manually copy it and use ctx.run() to ensure the
    RunContext ContextVar is visible inside the worker thread.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()
    done = asyncio.Event()           # set by worker thread when pipeline finishes
    result_holder: list = [None]
    error_holder:  list = [None]

    def emit_cb(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    # Install the RunContext into the current async context, then snapshot it
    # so the worker thread can run inside that snapshot via ctx.run().
    _, token = new_run_context(emit_cb)
    thread_ctx = contextvars.copy_context()

    runner = AgentForgeRunner(
        working_dir=working_dir,
        auto_approve=auto_approve,
        max_retries=max_retries,
    )

    def run_sync() -> None:
        try:
            result_holder[0] = runner.run(goal)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            loop.call_soon_threadsafe(done.set)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentforge")
    # Use executor.submit(thread_ctx.run, run_sync) so the thread inherits the
    # ContextVar snapshot that contains our RunContext (emit_cb).
    executor.submit(thread_ctx.run, run_sync)

    try:
        # Drain events with a 1s timeout per wait.  When the timeout fires and
        # done is set, do one final drain to catch any events queued right before
        # done.set() was called, then break.
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                await send_event(event)
            except asyncio.TimeoutError:
                if done.is_set():
                    while not queue.empty():
                        await send_event(queue.get_nowait())
                    break
    finally:
        _run_ctx.reset(token)
        executor.shutdown(wait=False)

    if error_holder[0] is not None:
        raise error_holder[0]  # type: ignore[misc]
    return result_holder[0]    # type: ignore[return-value]
