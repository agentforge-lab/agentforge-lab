# AgentForge

> Autonomous coding agent that runs entirely on your machine.
> No cloud. No API key required. Shows every decision it makes in a live graph.

![AgentForge Demo](docs/demo.gif)

---

## What it does

Give it a goal. AgentForge plans, writes code, runs tests, scans for security issues, and commits — autonomously. Watch every decision it makes in a live interactive graph.

- ✅ Runs 100% local via Ollama (qwen2.5-coder:7b)
- ✅ Live decision graph — watch every agent decision animate in real time
- ✅ Plan review before execution — see the full implementation spec, approve or cancel before any code runs
- ✅ Sandboxed workspace — generated code never touches your existing files
- ✅ Auto test runner (pytest / Jest)
- ✅ Security scan before every commit (Bandit — blocks on HIGH/CRITICAL)
- ✅ Git commit to feature branch automatically (never touches main)
- ✅ Zero cost per run — no API tokens burned

---

## Quick start

**Prerequisites:** Python 3.11+, Node 20+, [Ollama](https://ollama.ai)

```bash
# 1. Pull the local model
ollama pull qwen2.5-coder:7b

# 2. Clone and install
git clone https://github.com/agentforge-lab/agentforge-lab
cd agentforge-lab
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Terminal 1 — start the backend
agentforge serve

# 4. Terminal 2 — start the UI
cd frontend && npm install && npm run dev

# 5. Open http://localhost:3000
# Type a goal. Click Run.
```

---

## How it works

```
You type a goal
    ↓
Planner agent → implementation spec (you review + approve)
    ↓
Developer agent → writes code to sandboxed workspace
    ↓
Syntax check → Test runner (pytest) → Security scan (Bandit)
    ↓
Git Manager → commits to feature branch
```

Every step is visible in the decision graph UI. Click any node to see what the agent decided, why, and what it retried.

---

## Why local-first?

Every other autonomous coding agent (Devin, GitHub Copilot Workspace, Amazon Q) sends your code to a cloud server. That's a non-starter for developers in fintech, healthcare, defence, and enterprise where code cannot leave the building.

AgentForge runs entirely on your machine. Your code never leaves.

---

## Current status

Phase 2 of 7 complete. Active development.

| Feature | Status |
|---|---|
| Core agent pipeline | ✅ Working |
| Decision graph UI | ✅ Working |
| REST APIs, CLI tools, scripts | ✅ Works well |
| Multi-file large codebases | 🔧 In progress |
| Explainer Agent (OAuth, Stripe, etc.) | 📋 Planned |
| Document generation | 📋 Planned |

---

## Stack

Python · FastAPI · LangGraph · React · Ollama · WebSocket

---

## License

MIT — free to use, modify, and distribute.

---

## Follow the build

This project is being built in public. Star the repo to follow progress.
