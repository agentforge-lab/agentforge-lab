# AgentForge — E2E Benchmark Tests

This document tracks end-to-end test results across a range of goal types.
Run these after any significant change to agents, prompts, or orchestration logic.

**Models tested with:**
- Planner: `qwen2.5-coder:1.5b`
- Developer: `qwen2.5-coder:7b`
- Tester: `qwen2.5-coder:7b`

**Infrastructure (day-09+):**
- `num_ctx=16384` on all Ollama calls (was default 2048)
- Multi-turn conversation threading on developer retries (model sees its own prior output, not a rebuilt prompt)

---

## How to run a benchmark goal

```bash
mkdir -p /tmp/af_test_<name>
agentforge run "<goal>" --working-dir /tmp/af_test_<name> --auto-approve
```

A run **passes** when the final output shows `Tests: passed` and `Security: passed`.
A run **fails** if the final output shows `FAILED` after exhausting all retries (default: 3).

---

## Benchmark results

| # | Goal | Complexity | Result | Retries | Notes |
|---|------|-----------|--------|---------|-------|
| 1 | Word counter (words, sentences, characters in a string) | Low | ✅ Pass | 0 | — |
| 2 | Stack data structure (push, pop, peek, is_empty) | Low | ✅ Pass | 0 | — |
| 3 | Prime checker + Sieve of Eratosthenes | Low | ✅ Pass | 0 | — |
| 4 | Calculator (add, subtract, multiply, divide) | Low | ✅ Pass | 0 | — |
| 5 | Password generator (length, charset options) | Low | ✅ Pass | 1 | First attempt writes wrong test assertions for random output; retry fixes them |
| 6 | Unit converter (meters/km/miles, C/F/K) | Medium | ✅ Pass | 0 | Requires `pytest.approx(rel=1e-3)` in tests; tester prompt enforces this |
| 7 | Statistics calculator (mean, median, mode, std dev) | Medium | ✅ Pass | 0 | — |
| 8 | Todo list manager (add/remove/list, JSON persistence) | Medium | ✅ Pass | 0 | Developer must use optional `filepath` param; tester uses `tmp_path` directly |
| 9 | CSV manager (read, write, filter by column) | Medium | ✅ Pass | 1 | Developer retry needed when tester writes wrong expected types (strings vs ints) |
| 10 | Bank account (OOP, deposit/withdraw/transfer, ValueError on invalid ops) | Medium | ✅ Pass | 0 | — |
| 11 | Inventory management (add/remove/update/search, JSON persistence, category field) | Medium | ✅ Pass | 0 | Goal must mention `category` as a product field; search functions need `filepath` param |
| 12 | Markdown → HTML converter (h1-h3, bold, italic, code, unordered lists) | Hard | ❌ Fail | 2 | Multi-turn threading improved retry 1 from 1/5 → 2/6 tests; stuck at 2/6 on retry 2. Confirmed 7b capability limit, not context truncation. |

---

## Known limitations

### Markdown / stateful multi-line parsing

**Goal:** Markdown to HTML with unordered list support  
**Symptom:** `- Item 1\n- Item 2` produces `<ul><li>Item 1</li></ul><ul><li>Item 2</li></ul>` instead of one `<ul>` wrapping both items  
**Root cause:** `qwen2.5-coder:7b` consistently uses a per-line regex approach and cannot self-correct to a stateful line-grouping algorithm even with full context (num_ctx=16384) and multi-turn retry threading. Retry 1 improved from 1/5 → 2/6 (threading helps), but the algorithm reasoning failure is a genuine 7b ceiling.  
**Workaround:** Not fixable at 7b. Requires either a larger model (13b+) or rewriting the function manually after generation.  
**Affects:** Any goal that requires grouping consecutive input lines into a single output block (e.g. table rows, multi-line code fences, paragraph detection)

---

## What to retest after changes

| Change type | Goals to retest |
|-------------|----------------|
| `src/agents/tester.py` | All goals — especially #5 (random output), #8 and #11 (file I/O) |
| `src/agents/developer.py` | #6 (float math), #10 (OOP + exceptions), #12 (complex logic) |
| `src/llm/prompts.py` (tester prompt) | #5, #8, #9, #11 — anything with file I/O or numeric assertions |
| `src/llm/prompts.py` (developer prompt) | #8, #11 — file I/O dependency injection pattern |
| `src/orchestrator/graph.py` | Full run of all goals |
| Model swap (tester or developer) | Full run — all 12 goals |

---

## Regression checklist (run before every public commit)

```bash
# Quick smoke test — run these 3 in parallel
agentforge run "Build a Python calculator" --working-dir /tmp/smoke_calc --auto-approve
agentforge run "Build a Python stack data structure with push, pop, peek, and is_empty" --working-dir /tmp/smoke_stack --auto-approve
agentforge run "Build a Python todo list manager that can add, remove, and list tasks, saved to a JSON file" --working-dir /tmp/smoke_todo --auto-approve
```

All 3 should pass with 0 retries. If any fail, do not push.
