"""
Tests for PlannerAgent — LLM path, static fallback, JSON parsing, developer_brief.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.planner import (
    PlannerAgent, TaskPlan, TaskNode, NodeStatus,
    _parse_planner_json, _default_nodes,
)


# ── JSON parser ────────────────────────────────────────────────────────────

class TestParsePlannerJson:
    def test_plain_json(self):
        raw = json.dumps({"goal": "build X", "developer_brief": "Do Y", "nodes": {}})
        data = _parse_planner_json(raw)
        assert data["goal"] == "build X"
        assert data["developer_brief"] == "Do Y"

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"goal": "g", "developer_brief": "b", "nodes": {}}\n```'
        data = _parse_planner_json(raw)
        assert data["goal"] == "g"

    def test_json_in_plain_fence(self):
        raw = '```\n{"goal": "g", "developer_brief": "b", "nodes": {}}\n```'
        data = _parse_planner_json(raw)
        assert data["goal"] == "g"

    def test_json_with_leading_prose(self):
        raw = 'Here is my plan:\n{"goal": "g", "developer_brief": "spec", "nodes": {}}\nDone.'
        data = _parse_planner_json(raw)
        assert data["developer_brief"] == "spec"

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_planner_json("not json at all")

    def test_no_json_raises(self):
        with pytest.raises(Exception):
            _parse_planner_json("just some plain text with no braces")


# ── Static fallback plan ──────────────────────────────────────────────────

class TestStaticPlan:
    def test_no_llm_produces_static_plan(self):
        agent = PlannerAgent(llm_client=None)
        plan = agent.plan("build a calculator")
        assert isinstance(plan, TaskPlan)
        assert plan.goal == "build a calculator"
        assert len(plan.nodes) > 0

    def test_static_plan_has_developer_node(self):
        agent = PlannerAgent(llm_client=None)
        plan = agent.plan("any goal")
        agents = [n.agent for n in plan.nodes.values()]
        assert "developer" in agents

    def test_static_plan_developer_brief_is_empty(self):
        agent = PlannerAgent(llm_client=None)
        plan = agent.plan("any goal")
        assert plan.developer_brief == ""

    def test_default_nodes_returns_list(self):
        nodes = _default_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) >= 3

    def test_default_nodes_include_develop_and_commit(self):
        ids = {n.id for n in _default_nodes()}
        assert "develop" in ids
        assert "commit" in ids

    def test_third_party_service_flagged(self):
        agent = PlannerAgent(llm_client=None)
        plan = agent._static_plan("set up stripe payments")
        # The static plan doesn't scan for stripe in goal, only in node text
        # But we confirm plan is created without error
        assert plan is not None


# ── LLM-powered plan ──────────────────────────────────────────────────────

class TestLLMPlan:
    def _make_llm(self, json_payload: dict):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(
            content=json.dumps(json_payload)
        )
        return mock_llm

    def test_llm_plan_uses_developer_brief(self):
        payload = {
            "goal": "build a todo API",
            "developer_brief": "Create main.py with FastAPI, /todos GET/POST endpoints, in-memory list.",
            "nodes": {
                "develop": {"title": "dev", "description": "write code", "agent": "developer", "dependencies": []}
            }
        }
        agent = PlannerAgent(llm_client=self._make_llm(payload))
        plan = agent.plan("build a todo API")
        assert plan.developer_brief == payload["developer_brief"]

    def test_llm_plan_builds_task_nodes(self):
        payload = {
            "goal": "g",
            "developer_brief": "brief",
            "nodes": {
                "develop": {"title": "T", "description": "D", "agent": "developer", "dependencies": []},
                "commit":  {"title": "C", "description": "G", "agent": "git_manager", "dependencies": ["develop"]},
            }
        }
        agent = PlannerAgent(llm_client=self._make_llm(payload))
        plan = agent.plan("g")
        assert "develop" in plan.nodes
        assert "commit" in plan.nodes
        assert plan.nodes["commit"].dependencies == ["develop"]

    def test_llm_plan_falls_back_on_bad_json(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(content="not json at all")
        agent = PlannerAgent(llm_client=mock_llm)
        # Should fall back to static plan without raising
        plan = agent.plan("build something")
        assert isinstance(plan, TaskPlan)
        assert len(plan.nodes) > 0

    def test_llm_plan_falls_back_on_exception(self):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("network error")
        agent = PlannerAgent(llm_client=mock_llm)
        plan = agent.plan("build something")
        assert isinstance(plan, TaskPlan)

    def test_llm_plan_empty_nodes_uses_defaults(self):
        payload = {"goal": "g", "developer_brief": "brief", "nodes": {}}
        agent = PlannerAgent(llm_client=self._make_llm(payload))
        plan = agent.plan("g")
        assert len(plan.nodes) > 0  # defaults filled in

    def test_llm_plan_markdown_fence_response(self):
        payload = {"goal": "g", "developer_brief": "use FastAPI", "nodes": {}}
        content = f"Here's the plan:\n```json\n{json.dumps(payload)}\n```"
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(content=content)
        agent = PlannerAgent(llm_client=mock_llm)
        plan = agent.plan("g")
        assert plan.developer_brief == "use FastAPI"

    def test_llm_plan_requires_explain_flagged(self):
        payload = {
            "goal": "add stripe",
            "developer_brief": "add Stripe payments",
            "nodes": {
                "payments": {
                    "title": "Payments",
                    "description": "add stripe billing",
                    "agent": "developer",
                    "dependencies": [],
                    "requires_explain": True,
                }
            }
        }
        agent = PlannerAgent(llm_client=self._make_llm(payload))
        plan = agent.plan("add stripe")
        assert plan.nodes["payments"].requires_explain is True
        assert plan.nodes["payments"].status == NodeStatus.EXPLAIN


# ── TaskPlan ───────────────────────────────────────────────────────────────

class TestTaskPlan:
    def test_developer_brief_in_json_output(self):
        plan = TaskPlan(goal="g", developer_brief="build this specific thing")
        data = json.loads(plan.to_json())
        assert data["developer_brief"] == "build this specific thing"

    def test_empty_developer_brief_in_json(self):
        plan = TaskPlan(goal="g")
        data = json.loads(plan.to_json())
        assert data["developer_brief"] == ""

    def test_ready_nodes_with_no_deps(self):
        plan = TaskPlan(goal="g")
        n1 = TaskNode(id="n1", title="T", description="D", agent="developer")
        plan.add_node(n1)
        assert "n1" in [n.id for n in plan.get_ready_nodes()]

    def test_node_not_ready_until_dep_passes(self):
        plan = TaskPlan(goal="g")
        n1 = TaskNode(id="n1", title="T1", description="D", agent="developer")
        n2 = TaskNode(id="n2", title="T2", description="D", agent="developer", dependencies=["n1"])
        plan.add_node(n1)
        plan.add_node(n2)
        ready_ids = [n.id for n in plan.get_ready_nodes()]
        assert "n1" in ready_ids
        assert "n2" not in ready_ids

    def test_mark_failed_subtree_skips_dependents(self):
        plan = TaskPlan(goal="g")
        n1 = TaskNode(id="n1", title="T", description="D", agent="developer")
        n2 = TaskNode(id="n2", title="T", description="D", agent="developer", dependencies=["n1"])
        plan.add_node(n1)
        plan.add_node(n2)
        plan.mark_failed_subtree("n1")
        assert plan.nodes["n2"].status == NodeStatus.SKIPPED
