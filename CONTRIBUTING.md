# Contributing to AgentForge

Thanks for your interest. AgentForge is early-stage and actively developed — contributions are welcome.

## Before opening a PR

Open an issue first for anything beyond a small bug fix. This avoids wasted effort if the direction doesn't fit the roadmap.

## What we're looking for

- Bug fixes with a clear reproduction case
- Improvements to the agent prompts (with benchmark evidence)
- New benchmark goals in `docs/benchmarks.md`
- Documentation improvements

## What to avoid

- Adding dependencies without discussion
- Large refactors without prior agreement
- Features that require cloud infrastructure (AgentForge is local-first by design)

## Setup

```bash
git clone https://github.com/agentforge-lab/agentforge-lab
cd agentforge-lab
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
ollama pull qwen2.5-coder:7b
```

## Running the smoke tests before submitting

```bash
mkdir -p /tmp/smoke_calc /tmp/smoke_stack /tmp/smoke_todo
agentforge run "Build a Python calculator" --working-dir /tmp/smoke_calc --auto-approve
agentforge run "Build a Python stack data structure with push, pop, peek, and is_empty" --working-dir /tmp/smoke_stack --auto-approve
agentforge run "Build a Python todo list manager that can add, remove, and list tasks, saved to a JSON file" --working-dir /tmp/smoke_todo --auto-approve
```

All three must pass with 0 retries. If any fail, fix before submitting.

## Style

- Python 3.11+, standard library preferred over new dependencies
- No comments explaining what the code does — only why if the reason is non-obvious
- Match the existing code style in the file you're editing
