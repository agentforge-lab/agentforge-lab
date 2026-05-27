# AgentForge

Autonomous software development system with a live decision graph.

**Status:** Phase 1 — Core agent loop

## Quick start

```bash
# Set up environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your API key
echo "ANTHROPIC_API_KEY=your_key_here" >> .env.local

# Run (once Phase 1 CLI is complete)
agentforge run "your goal here"
```

## Architecture

- **Layer 1:** Hardware intelligence — detects GPU/CPU, routes to local model or API
- **Layer 2:** Orchestrator — LangGraph state machine, builds decision tree
- **Layer 3:** Agent swarm — Planner, Developer, Tester, Security, Git Manager
- **Layer 4:** Intelligence — Decision graph UI, cost calculator, self-improvement
- **Layer 5:** Post-build — Explainer agent, document generation
- **Layer 6:** Memory — `.agentforge/` session files, vector search

See `AGENTFORGE.md` for full specification.
