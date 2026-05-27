"""
Planner Agent — breaks a user goal into a structured task DAG.
Outputs a directed acyclic graph of subtasks with dependencies.
Handles backtracking: on downstream failure, re-plans from failure point.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    OVERRIDDEN = "overridden"
    SKIPPED = "skipped"
    EXPLAIN = "explain"  # third-party service detected


@dataclass
class TaskNode:
    id: str
    title: str
    description: str
    agent: str  # which agent executes this node
    status: NodeStatus = NodeStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    result: Any = None
    error: str | None = None
    reasoning: str = ""
    alternatives: list[str] = field(default_factory=list)
    requires_explain: bool = False  # triggers Explainer Agent
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "agent": self.agent,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "retry_count": self.retry_count,
            "result": self.result,
            "error": self.error,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "requires_explain": self.requires_explain,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class TaskPlan:
    goal: str
    developer_brief: str = ""   # refined task spec produced by LLM planner
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.id] = node
        self.updated_at = datetime.now().isoformat()

    def get_ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all PASSED."""
        ready = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps_done = all(
                self.nodes[dep].status == NodeStatus.PASSED
                for dep in node.dependencies
                if dep in self.nodes
            )
            if deps_done:
                ready.append(node)
        return ready

    def mark_failed_subtree(self, failed_id: str) -> None:
        """Mark all nodes that depend (transitively) on a failed node as SKIPPED."""
        def dependents(nid: str) -> list[str]:
            return [n.id for n in self.nodes.values() if nid in n.dependencies]

        to_skip = set()
        queue = dependents(failed_id)
        while queue:
            nid = queue.pop()
            if nid not in to_skip:
                to_skip.add(nid)
                queue.extend(dependents(nid))

        for nid in to_skip:
            self.nodes[nid].status = NodeStatus.SKIPPED

    def to_json(self) -> str:
        return json.dumps({
            "goal": self.goal,
            "developer_brief": self.developer_brief,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }, indent=2)


THIRD_PARTY_SERVICES = {
    "oauth", "google", "github", "stripe", "sendgrid", "twilio",
    "aws", "gcp", "azure", "supabase", "neon", "planetscale",
    "cloudflare", "terraform", "docker", "auth0", "clerk",
    "resend", "railway", "render", "fly.io", "lemon squeezy", "paddle",
}


def _parse_planner_json(text: str) -> dict:
    """
    Extract and parse JSON from planner LLM response.
    Handles markdown code fences and leading/trailing prose.
    """
    # Strip ```json ... ``` or ``` ... ``` wrappers
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Find first { ... } block
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError("No JSON object found in planner response")


def _default_nodes() -> list[TaskNode]:
    return [
        TaskNode(
            id="develop",
            title="Develop: Implement the goal",
            description="Write all required code",
            agent="developer",
        ),
        TaskNode(
            id="test",
            title="Test: Unit + integration suite",
            description="Write and run tests, ensure coverage > 80%",
            agent="tester",
            dependencies=["develop"],
        ),
        TaskNode(
            id="security",
            title="Security: Vulnerability scan",
            description="Run Bandit + Semgrep, block on Critical/High findings",
            agent="security",
            dependencies=["develop"],
        ),
        TaskNode(
            id="commit",
            title="Commit: Push feature branch",
            description="Git commit on feature branch",
            agent="git_manager",
            dependencies=["test", "security"],
        ),
    ]


class PlannerAgent:
    """
    Decomposes a user goal into a TaskPlan (DAG).
    Calls the LLM to produce a developer_brief + structured task nodes.
    Falls back to a static plan if LLM is unavailable or output is unparseable.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def plan(self, goal: str) -> TaskPlan:
        """Break a goal into a TaskPlan with a refined developer_brief."""
        if self.llm is not None:
            try:
                return self._llm_plan(goal)
            except Exception:
                pass  # fall through to static plan
        return self._static_plan(goal)

    # ── LLM planning ──────────────────────────────────────────────────────

    def _llm_plan(self, goal: str) -> TaskPlan:
        from src.llm.prompts import PLANNER_SYSTEM
        response = self.llm.complete(PLANNER_SYSTEM, goal)
        data = _parse_planner_json(response.content)

        plan = TaskPlan(
            goal=data.get("goal", goal),
            developer_brief=data.get("developer_brief", "").strip(),
        )

        nodes_data: dict = data.get("nodes", {})
        if nodes_data:
            for node_id, nd in nodes_data.items():
                node = TaskNode(
                    id=node_id,
                    title=nd.get("title", node_id),
                    description=nd.get("description", ""),
                    agent=nd.get("agent", "developer"),
                    dependencies=nd.get("dependencies", []),
                    reasoning=nd.get("reasoning", ""),
                    alternatives=nd.get("alternatives", []),
                    requires_explain=bool(nd.get("requires_explain", False)),
                )
                if node.requires_explain:
                    node.status = NodeStatus.EXPLAIN
                # Also flag nodes mentioning third-party services
                text = (node.title + node.description).lower()
                if any(svc in text for svc in THIRD_PARTY_SERVICES):
                    node.requires_explain = True
                    node.status = NodeStatus.EXPLAIN
                plan.add_node(node)
        else:
            for node in _default_nodes():
                plan.add_node(node)

        return plan

    # ── Static fallback ───────────────────────────────────────────────────

    def _static_plan(self, goal: str) -> TaskPlan:
        plan = TaskPlan(goal=goal)
        for node in _default_nodes():
            text = (node.title + node.description).lower()
            if any(svc in text for svc in THIRD_PARTY_SERVICES):
                node.requires_explain = True
                node.status = NodeStatus.EXPLAIN
            plan.add_node(node)
        return plan

    def replan_from(self, plan: TaskPlan, failed_node_id: str) -> TaskPlan:
        """
        Backtrack: mark failed node's subtree SKIPPED, re-plan from parent.
        Called by orchestrator after max_retries exceeded.
        """
        if failed_node_id not in plan.nodes:
            raise ValueError(f"Node {failed_node_id} not in plan")

        plan.nodes[failed_node_id].status = NodeStatus.FAILED
        plan.mark_failed_subtree(failed_node_id)
        plan.updated_at = datetime.now().isoformat()
        return plan

    def apply_override(self, plan: TaskPlan, node_id: str, instruction: str) -> TaskPlan:
        """
        Human override: user changes a node decision.
        Marks node OVERRIDDEN, logs original, re-plans children.
        """
        node = plan.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not in plan")

        original_desc = node.description
        node.status = NodeStatus.OVERRIDDEN
        node.description = instruction
        node.reasoning = f"Human override. Original: {original_desc}"
        plan.updated_at = datetime.now().isoformat()
        return plan
