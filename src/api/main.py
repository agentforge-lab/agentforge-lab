"""
FastAPI application — AgentForge backend.

Endpoints:
  GET  /health          — liveness check
  GET  /api/hardware    — hardware profile text
  POST /api/plan        — run just the planner; returns plan JSON for UI review
  WS   /ws/run          — stream pipeline events to the decision graph UI

Static frontend served from frontend/dist when present (production build).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.ws_runner import run_pipeline_ws

# Directory that contains the AgentForge application source itself.
_AGENTFORGE_ROOT = Path(__file__).resolve().parents[2]


def _create_project_workspace(goal: str) -> Path:
    """
    Create a fresh, sandboxed directory for a new project run.
    Placed at ~/agentforge_projects/<date>_<slug>/ so generated code
    is never written into the AgentForge source tree.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower().strip())[:40].strip("-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = Path.home() / "agentforge_projects" / f"{timestamp}_{slug}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace

app = FastAPI(title="AgentForge API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/api/hardware")
async def hardware_profile():
    path = Path(".agentforge/hardware_profile.md")
    if path.exists():
        return {"profile": path.read_text()}
    return {"error": "Hardware profile not found — run: agentforge init"}


# ── Plan endpoint ─────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    goal: str


@app.post("/api/plan")
async def plan_goal(req: PlanRequest):
    """
    Run just the Planner agent and return the implementation plan as JSON.
    The frontend calls this first so the user can review before execution starts.
    No files are written; no workspace is created yet.
    """
    goal = req.goal.strip()
    if not goal:
        return {"error": "goal is required"}

    import asyncio as _asyncio
    from concurrent.futures import ThreadPoolExecutor

    from src.agents.planner import PlannerAgent
    from src.llm.client import LLMClient

    def _run_planner() -> dict:
        llm = LLMClient.from_hardware_profile()
        planner = PlannerAgent(llm_client=llm)
        try:
            plan = planner.plan(goal)
        except Exception as exc:
            return {"error": str(exc)}

        slug = re.sub(r"[^a-z0-9]+", "-", goal.lower().strip())[:30].strip("-")
        date = datetime.now().strftime("%Y%m%d")
        branch = f"agent/{slug}-{date}"

        return {
            "goal": plan.goal or goal,
            "developer_brief": plan.developer_brief,
            "branch": branch,
            "workspace_parent": str(Path.home() / "agentforge_projects"),
            "nodes": {nid: n.to_dict() for nid, n in plan.nodes.items()},
        }

    loop = _asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        result = await loop.run_in_executor(ex, _run_planner)
    return result


# ── Folder picker endpoint ────────────────────────────────────────────────

@app.get("/api/pick-folder")
async def pick_folder():
    """
    Open a native macOS folder-picker dialog on the server side (server and
    browser are the same machine for a local tool) and return the chosen path.
    Uses osascript (AppleScript) — always available on macOS, no dependencies.
    Returns {"path": "/chosen/path"} or {"cancelled": true} if user dismissed.
    """
    import asyncio as _asyncio
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    def _open_dialog() -> dict:
        script = 'POSIX path of (choose folder with prompt "Select your project folder")'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=120,   # wait up to 2 min for user to pick
            )
            if result.returncode == 0:
                return {"path": result.stdout.strip()}
            # User pressed Cancel — osascript exits with code 1
            return {"cancelled": True}
        except subprocess.TimeoutExpired:
            return {"cancelled": True}
        except FileNotFoundError:
            return {"error": "osascript not found — folder picker only works on macOS"}
        except Exception as e:
            return {"error": str(e)}

    loop = _asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        result = await loop.run_in_executor(ex, _open_dialog)
    return result


# ── Analyze endpoint ─────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    path: str


@app.post("/api/analyze")
async def analyze_project(req: AnalyzeRequest):
    """
    Scan an existing project folder and return a codebase summary.
    Uses the Explainer Agent to produce a plain-English description of the project.
    """
    path = Path(req.path.strip()).expanduser().resolve()

    if not path.exists():
        return {"error": f"Path does not exist: {path}"}
    if not path.is_dir():
        return {"error": f"Not a directory: {path}"}

    import asyncio as _asyncio
    from concurrent.futures import ThreadPoolExecutor
    from src.agents.explainer import ExplainerAgent
    from src.orchestrator.graph import _collect_existing_files

    def _run_analysis() -> dict:
        files = _collect_existing_files(path, task="", max_files=10, max_chars_per_file=4000)
        if not files:
            return {"error": "No Python source files found in this directory"}

        summary = ExplainerAgent().explain(files, goal="")
        return {
            "path": str(path),
            "file_count": len(files),
            "files": list(files.keys()),
            "summary": summary,
        }

    loop = _asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        result = await loop.run_in_executor(ex, _run_analysis)
    return result


# ── WebSocket endpoint ────────────────────────────────────────────────────

@app.websocket("/ws/run")
async def run_ws(websocket: WebSocket):
    """
    Stream all pipeline events to the frontend decision graph.

    Client sends one JSON message to start:
      {"goal": "...", "auto_approve": true, "max_retries": 3, "working_dir": "."}

    Server streams events as JSON objects, then closes with a run_result message.
    """
    await websocket.accept()

    try:
        data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        await websocket.close(code=1000)
        return

    goal         = (data.get("goal") or "").strip()
    auto_approve = bool(data.get("auto_approve", True))
    max_retries  = int(data.get("max_retries", 3))
    mode         = data.get("mode", "pipeline")   # "pipeline" or "agent"

    if not goal:
        await websocket.send_json({"type": "error", "data": {"message": "goal is required"}})
        await websocket.close(code=1003)
        return

    # If the client sends the default "." working_dir, or it resolves to the
    # AgentForge source tree, sandbox the run in a fresh project directory so
    # the agent can never accidentally overwrite AgentForge's own source files.
    raw_wd = Path(data.get("working_dir") or ".").resolve()
    if raw_wd == _AGENTFORGE_ROOT or str(data.get("working_dir", ".")).strip() in (".", ""):
        working_dir = _create_project_workspace(goal)
    else:
        working_dir = raw_wd

    async def send_event(event: dict) -> None:
        try:
            await websocket.send_json(event)
        except Exception:
            pass

    await send_event({
        "type": "workspace_ready",
        "node_id": "",
        "data": {"path": str(working_dir)},
    })

    try:
        result = await run_pipeline_ws(
            goal=goal,
            working_dir=working_dir,
            auto_approve=auto_approve,
            max_retries=max_retries,
            send_event=send_event,
            mode=mode,
        )
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_event({"type": "run_failed", "node_id": "", "data": {"error": str(e)}})
    else:
        # run_result is a convenience duplicate — the same data already came
        # through the event stream as RUN_COMPLETED.  Catch disconnect because
        # the client may have closed as soon as it received that event.
        try:
            await send_event({
                "type": "run_completed", "node_id": "",
                "data": {
                    "success":         result.success,
                    "commit_sha":      result.commit_sha,
                    "branch":          result.branch,
                    "tests_passed":    result.tests_passed,
                    "security_passed": result.security_passed,
                    "retry_count":     result.retry_count,
                    "error":           result.error,
                },
            })
        except Exception:
            pass
