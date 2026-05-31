"""
AgentForge CLI — `agentforge run "your goal"`

Commands:
  run     Plan, code, test, secure, and commit a goal.
  init    Detect hardware and write .agentforge/hardware_profile.md.
  status  Show hardware profile and recent session summaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from src.hardware.detector import detect_hardware, write_hardware_profile
from src.agents.planner import PlannerAgent
from src.llm.model_config import (
    AgentModelConfig, detect_ollama_models, load_model_config,
    save_model_config, suggest_config,
)
from src.orchestrator.runner import AgentForgeRunner, RunResult

# ── helpers ────────────────────────────────────────────────────────────────

def _working_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise click.BadParameter(f"Directory does not exist: {p}")
    return p


def _stream_run(runner: AgentForgeRunner, goal: str) -> RunResult:
    """Stream node-by-node progress then return the final RunResult."""
    from src.orchestrator.graph import default_state

    app = runner._get_app()
    initial = default_state(goal, max_retries=runner.max_retries)

    node_labels = {
        "planner":          "Planning task",
        "human_checkpoint": "Awaiting approval",
        "increment_retry":  "Incrementing retry",
        "developer":        "Writing code",
        "executor":         "Checking syntax",
        "tester":           "Running tests",
        "security":         "Security scan",
        "git_manager":      "Committing",
    }

    full_state: dict = {}
    last_log_len = 0
    try:
        for full_state in app.stream(initial, stream_mode="values"):
            log = full_state.get("session_log", [])
            if len(log) > last_log_len:
                new_entry = log[-1]
                node_name = new_entry.split("] ")[-1].split(":")[0].strip() if "] " in new_entry else ""
                label = node_labels.get(node_name, node_name or "...")
                retry = full_state.get("retry_count", 0)
                suffix = f"  (retry {retry})" if retry and node_name == "developer" else ""
                click.echo(f"  ▸ {label}{suffix}")
                click.echo(f"    {new_entry}")
                last_log_len = len(log)
    except Exception as e:
        return RunResult(success=False, goal=goal, error=f"Graph execution error: {e}")

    return RunResult(
        success=full_state.get("complete", False) and not full_state.get("final_error"),
        goal=goal,
        commit_sha=full_state.get("commit_sha") or None,
        branch=full_state.get("branch_committed") or None,
        tests_passed=full_state.get("tests_passed", False),
        security_passed=full_state.get("security_passed", False),
        retry_count=full_state.get("retry_count", 0),
        session_log=full_state.get("session_log", []),
        error=full_state.get("final_error") or None,
    )


# ── CLI group ──────────────────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="agentforge")
def cli():
    """AgentForge — autonomous software development on your laptop."""


# ── run ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("goal")
@click.option("--auto-approve", is_flag=True, default=False,
              help="Skip the human checkpoint (non-interactive runs).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Plan only — do not write files or commit.")
@click.option("--max-retries", default=3, show_default=True,
              help="Max developer retries on test/security failure.")
@click.option("--working-dir", default=".", show_default=True,
              help="Root of the project being built.")
@click.option("--stream/--no-stream", default=True, show_default=True,
              help="Stream node-by-node progress.")
@click.option("--mode", default="pipeline",
              type=click.Choice(["pipeline", "agent"]),
              help="pipeline: fixed plan→code→test→commit graph (default). "
                   "agent: truly agentic ReAct loop — model decides every step.")
def run(goal, auto_approve, dry_run, max_retries, working_dir, stream, mode):
    """Plan, code, test, secure, and commit GOAL.

    \b
    Pipeline mode (default): fixed 7-node graph, local models work.
    Agent mode:              ReAct loop with tool use — needs an API key.
                             Set one first: agentforge keys set gemini YOUR_KEY
                             Then:          agentforge models set agent_loop gemini/gemini-1.5-flash
    """
    try:
        wd = _working_dir(working_dir)
    except click.BadParameter as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if mode == "agent" and not dry_run:
        from src.llm.keys import load_keys_to_env, has_key
        from src.llm.model_config import load_model_config
        load_keys_to_env()
        cfg = load_model_config()
        agent_model = cfg.for_agent("agent_loop")
        from src.llm.keys import detect_provider
        provider = detect_provider(agent_model)
        if provider and not has_key(provider):
            click.echo(f"\n  ✗ Agent mode needs an API key for '{provider}'.")
            click.echo(f"    Run: agentforge keys set {provider} YOUR_KEY")
            click.echo(f"    Free options: agentforge keys free")
            sys.exit(1)

    click.echo(f"\nAgentForge  ▸  {goal}")
    click.echo(f"  Working dir : {wd}")
    click.echo(f"  Mode        : {mode}")
    if mode == "pipeline":
        click.echo(f"  Auto-approve: {auto_approve}")
    if dry_run:
        click.echo("  Dry-run mode: plan only (no writes, no commits)")
    click.echo()

    if dry_run:
        _do_dry_run(goal, wd)
        return

    runner = AgentForgeRunner(
        working_dir=wd,
        auto_approve=auto_approve,
        max_retries=max_retries,
        mode=mode,
    )

    if mode == "agent":
        result = _stream_agent_run(runner, goal)
    elif stream:
        result = _stream_run(runner, goal)
    else:
        result = runner.run(goal)

    click.echo(str(result))
    sys.exit(0 if result.success else 1)


def _stream_agent_run(runner: AgentForgeRunner, goal: str):
    """Stream agentic tool-call steps to the terminal."""
    import asyncio
    from src.agents.loop import AgentLoop
    from src.llm.model_config import load_model_config, make_llm_client

    cfg    = load_model_config()
    model  = cfg.for_agent("agent_loop")
    llm    = make_llm_client(model)
    loop   = AgentLoop(llm=llm, working_dir=runner.working_dir, goal=goal)

    click.echo(f"  Model: {model}\n")

    async def _run():
        async def on_event(event: dict):
            t = event.get("type", "")
            d = event.get("data", {})
            if t == "agent_tool_call":
                step    = d.get("step", "?")
                tool    = d.get("tool", "")
                success = d.get("success", True)
                preview = d.get("output_preview", "")
                marker  = "✓" if success else "✗"
                click.echo(f"  [{step:>2}] {marker} {tool:<20} {preview[:80]}")
            elif t == "agent_done":
                click.echo(f"\n  ✓ Done — {d.get('summary', '')}")
                if d.get("commit_sha"):
                    click.echo(f"     Commit: {d['commit_sha']}  Branch: {d.get('branch', '')}")
        return await loop.run(goal, send_event=on_event)

    result = asyncio.run(_run())

    from src.orchestrator.runner import RunResult
    return RunResult(
        success=result.success,
        goal=goal,
        commit_sha=result.commit_sha,
        branch=result.branch,
        tests_passed=result.tests_passed,
        security_passed=result.security_passed,
        retry_count=result.steps,
        session_log=result.session_log,
        error=result.error,
    )


def _do_dry_run(goal: str, wd: Path) -> None:
    click.echo("  ▸ Planning task (dry-run — no LLM call)")
    planner = PlannerAgent()
    plan = planner.plan(goal)
    click.echo(f"\n  Task       : {plan.task_description}")
    click.echo(f"  Branch     : {plan.branch_name}")
    click.echo(f"  Plan items : {len(plan.steps) if hasattr(plan, 'steps') else 'n/a'}")
    click.echo("\n  (Dry-run complete — no files written, no commit made)")


# ── agent ──────────────────────────────────────────────────────────────────

@cli.group()
def agent():
    """Commands for the truly agentic ReAct mode."""


@agent.command("test")
@click.option("--working-dir", default=None,
              help="Directory to run in (default: fresh temp dir).")
@click.option("--model", default=None,
              help="Override the agent_loop model for this test.")
def agent_test(working_dir, model):
    """
    Run a minimal end-to-end test of the agentic loop.

    \b
    What it does:
      1. Writes a simple add() function to add.py
      2. Runs the tests (which the agent writes itself)
      3. Runs the security scan
      4. Commits the result

    Use this to confirm your API key and model work before running
    real goals. Takes ~30-60 seconds with Gemini Flash.

    \b
    Setup:
      agentforge keys set gemini YOUR_KEY
      agentforge models set agent_loop gemini/gemini-1.5-flash
      agentforge agent test
    """
    import asyncio
    import tempfile
    from src.llm.keys import load_keys_to_env, has_key, detect_provider
    from src.llm.model_config import load_model_config, make_llm_client
    from src.agents.loop import AgentLoop

    load_keys_to_env()

    cfg         = load_model_config()
    agent_model = model or cfg.for_agent("agent_loop")
    provider    = detect_provider(agent_model)

    if provider and not has_key(provider):
        click.echo(f"\n  ✗ No API key for '{provider}'.")
        click.echo(f"    Run: agentforge keys set {provider} YOUR_KEY")
        click.echo(f"    Free options: agentforge keys free\n")
        sys.exit(1)

    # Use a temp dir if none given
    if working_dir:
        wd = Path(working_dir).expanduser().resolve()
        wd.mkdir(parents=True, exist_ok=True)
    else:
        _tmp = tempfile.mkdtemp(prefix="agentforge_test_")
        wd   = Path(_tmp)

    goal = (
        "Write a Python file called add.py with a single function add(a, b) "
        "that returns a + b. Write pytest tests for it. Run the tests to confirm "
        "they pass, run the security scan, then commit."
    )

    click.echo(f"\n  AgentForge agent test")
    click.echo(f"  Model      : {agent_model}")
    click.echo(f"  Working dir: {wd}")
    click.echo(f"  Goal       : {goal[:80]}...")
    click.echo()

    llm  = make_llm_client(agent_model)
    loop = AgentLoop(llm=llm, working_dir=wd, goal=goal)
    steps_seen: list[str] = []

    async def _run():
        async def on_event(event: dict):
            t = event.get("type", "")
            d = event.get("data", {})
            if t == "agent_tool_call":
                step    = d.get("step", "?")
                tool    = d.get("tool", "")
                success = d.get("success", True)
                preview = (d.get("output_preview") or "")[:72]
                marker  = "✓" if success else "✗"
                line    = f"  [{step:>2}] {marker} {tool:<22} {preview}"
                click.echo(line)
                steps_seen.append(tool)
            elif t == "agent_done":
                click.echo(f"\n  ✓ Complete — {d.get('summary', '')}")
                if d.get("commit_sha"):
                    click.echo(f"     Commit : {d['commit_sha']}")
                    click.echo(f"     Branch : {d.get('branch', '')}")
        return await loop.run(goal, send_event=on_event)

    result = asyncio.run(_run())

    click.echo()
    if result.success:
        click.echo("  ✓ Agent test PASSED")
        click.echo(f"    Steps taken : {result.steps}")
        click.echo(f"    Tools used  : {', '.join(dict.fromkeys(steps_seen))}")
        click.echo(f"    Working dir : {wd}")
        click.echo()
        click.echo("  The agentic loop is working. You can now run:")
        click.echo(f'    agentforge run "your goal" --mode agent')
    else:
        click.echo(f"  ✗ Agent test FAILED: {result.error}")
        click.echo()
        click.echo("  Troubleshooting:")
        click.echo("    agentforge keys test gemini       ← verify key works")
        click.echo("    agentforge keys free              ← see all free options")
        sys.exit(1)


# ── serve ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Interface to bind (use 0.0.0.0 for LAN access).")
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False,
              help="Auto-reload on code changes (dev mode).")
def serve(host, port, reload):
    """Start the AgentForge decision graph web UI."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is not installed. Run: pip install 'uvicorn[standard]'", err=True)
        sys.exit(1)

    display_host = "localhost" if host == "127.0.0.1" else host
    click.echo(f"\nAgentForge UI  →  http://{display_host}:{port}")
    click.echo("  Decision graph streams live pipeline events in your browser.")
    click.echo("  Press Ctrl+C to stop.\n")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)


# ── init ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--working-dir", default=".", show_default=True)
def init(working_dir):
    """Detect hardware and initialise .agentforge/ configuration."""
    try:
        wd = _working_dir(working_dir)
    except click.BadParameter as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    click.echo("\nAgentForge init")
    click.echo(f"  Working dir: {wd}")
    click.echo()

    click.echo("  ▸ Detecting hardware ...")
    profile = detect_hardware()

    profile_dir = wd / ".agentforge"
    profile_dir.mkdir(exist_ok=True)
    write_hardware_profile(profile, profile_dir / "hardware_profile.md")

    click.echo(f"  Chip        : {profile.chip_name}")
    click.echo(f"  RAM         : {profile.ram_gb:.1f} GB")
    click.echo(f"  Eff. VRAM   : {profile.effective_vram_gb:.1f} GB")
    click.echo(f"  Model tier  : {profile.recommended_model}")
    click.echo(f"  API key     : {'yes' if profile.has_api_key else 'no (local-only mode)'}")
    click.echo(f"\n  Written     : {profile_dir / 'hardware_profile.md'}")
    click.echo("\n  AgentForge is ready. Run: agentforge run \"your goal\"")


# ── status ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--working-dir", default=".", show_default=True)
def status(working_dir):
    """Show hardware profile and recent session summaries."""
    try:
        wd = _working_dir(working_dir)
    except click.BadParameter as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    click.echo("\nAgentForge status")
    click.echo(f"  Working dir: {wd}")
    click.echo()

    profile_path = wd / ".agentforge" / "hardware_profile.md"
    if profile_path.exists():
        click.echo("  Hardware profile:")
        for line in profile_path.read_text().splitlines():
            if line.strip():
                click.echo(f"    {line}")
    else:
        click.echo("  Hardware profile: not found  (run: agentforge init)")

    summaries_dir = wd / ".agentforge" / "session_summaries"
    if summaries_dir.exists():
        summaries = sorted(summaries_dir.glob("*.md"))
        if summaries:
            click.echo(f"\n  Session summaries ({len(summaries)} found):")
            for f in summaries[-3:]:
                click.echo(f"    {f.name}")
        else:
            click.echo("\n  No session summaries yet.")
    else:
        click.echo("\n  No session summaries directory found.")


# ── models ─────────────────────────────────────────────────────────────────

@cli.group()
def models():
    """Configure which model each agent uses."""


@models.command("list")
def models_list():
    """Show the current per-agent model configuration."""
    from src.llm.model_config import _GLOBAL_CONFIG, _PROJECT_CONFIG
    cfg = load_model_config()
    click.echo(f"\n{cfg.display()}")
    click.echo()
    if _PROJECT_CONFIG.exists():
        click.echo(f"  Source: {_PROJECT_CONFIG} (project)")
    elif _GLOBAL_CONFIG.exists():
        click.echo(f"  Source: {_GLOBAL_CONFIG} (global)")
    else:
        click.echo("  Source: built-in defaults (no config file found)")
    click.echo()
    click.echo("  To change: agentforge models set <agent> <model>")
    click.echo("  To detect: agentforge models detect")


@models.command("set")
@click.argument("agent", type=click.Choice(["planner", "developer", "tester", "agent_loop", "default"]))
@click.argument("model")
@click.option("--project", is_flag=True, default=False,
              help="Save to project (.agentforge/) instead of global (~/.agentforge/).")
def models_set(agent, model, project):
    """
    Set the model for AGENT.

    \b
    Examples:
      agentforge models set developer qwen2.5-coder:7b
      agentforge models set planner llama3.1:8b
      agentforge models set tester qwen2.5-coder:1.5b
      agentforge models set developer claude-haiku-4-5-20251001
    """
    cfg = load_model_config()
    setattr(cfg, agent, model)
    path = save_model_config(cfg, global_scope=not project)
    scope = "project" if project else "global"
    click.echo(f"\n  {agent} → {model}  (saved to {scope} config: {path})")
    click.echo()
    click.echo(cfg.display())


@models.command("detect")
@click.option("--save", is_flag=True, default=False,
              help="Save the suggested config without prompting.")
def models_detect(save):
    """
    Query Ollama for installed models and suggest a per-agent configuration.

    The suggestion assigns your largest/best model to developer + planner,
    and the smallest to tester (which is less critical and saves VRAM).
    """
    click.echo("\n  Querying Ollama for installed models...")
    available = detect_ollama_models()

    if not available:
        click.echo("  Ollama is not running or has no models installed.")
        click.echo("  Start it with: ollama serve")
        click.echo("  Pull a model:  ollama pull qwen2.5-coder:1.5b")
        return

    click.echo(f"\n  Found {len(available)} model(s):")
    for m in available:
        click.echo(f"    - {m}")

    suggested = suggest_config(available)
    click.echo(f"\n  Suggested configuration:")
    click.echo(f"  {suggested.display()}")
    click.echo()

    if save:
        path = save_model_config(suggested)
        click.echo(f"  Saved to: {path}")
    else:
        if click.confirm("  Save this configuration?"):
            path = save_model_config(suggested)
            click.echo(f"  Saved to: {path}")
        else:
            click.echo("  Not saved. Run with --save to skip the prompt.")


# ── keys ───────────────────────────────────────────────────────────────────

@cli.group()
def keys():
    """Manage API keys for all LLM providers (Anthropic, OpenAI, Gemini, Groq…)."""


@keys.command("set")
@click.argument("provider")
@click.argument("key")
def keys_set(provider, key):
    """Save an API key. PROVIDER: anthropic, openai, gemini, groq, openrouter, together."""
    from src.llm.keys import save_key, ENV_MAP
    try:
        save_key(provider, key)
        click.echo(f"\n  ✓ {provider} key saved  →  {ENV_MAP[provider]}")
    except ValueError as e:
        click.echo(f"\n  ✗ {e}", err=True)
        sys.exit(1)


@keys.command("remove")
@click.argument("provider")
def keys_remove(provider):
    """Remove a saved API key for PROVIDER."""
    from src.llm.keys import remove_key
    if remove_key(provider):
        click.echo(f"\n  ✓ {provider} key removed")
    else:
        click.echo(f"\n  {provider} key not found")


@keys.command("list")
def keys_list():
    """Show all saved API keys (masked) and their status."""
    import os
    from src.llm.keys import list_keys, ENV_MAP
    click.echo("\n  API key status:\n")
    saved = list_keys()
    for provider, env_var in ENV_MAP.items():
        if provider in saved:
            click.echo(f"    {provider:<12} {saved[provider]}  (saved)")
        elif os.environ.get(env_var):
            val = os.environ[env_var]
            masked = f"{val[:4]}...{val[-4:]}" if len(val) > 8 else "****"
            click.echo(f"    {provider:<12} {masked}  (from environment)")
        else:
            click.echo(f"    {provider:<12} —  not set")
    click.echo()
    click.echo("  To add: agentforge keys set <provider> <key>")
    click.echo("  Free:   agentforge keys free")


@keys.command("free")
def keys_free():
    """Show free model options — no payment required."""
    from src.llm.keys import FREE_MODELS
    click.echo("\n  Free models (get API key, no credit card needed):\n")
    for model, desc in FREE_MODELS.items():
        click.echo(f"    {model}")
        click.echo(f"      {desc}\n")
    click.echo("  ── How to get started (recommended: Gemini Flash) ──")
    click.echo()
    click.echo("  1. Get free Gemini key: https://aistudio.google.com → 'Get API key'")
    click.echo("  2. Save it:             agentforge keys set gemini YOUR_KEY")
    click.echo("  3. Set agent model:     agentforge models set agent_loop gemini/gemini-1.5-flash")
    click.echo("  4. Run agentic mode:    agentforge run \"your goal\" --mode agent")
    click.echo()
    click.echo("  ── Groq alternative (very fast) ──")
    click.echo()
    click.echo("  1. Get free Groq key:   https://console.groq.com")
    click.echo("  2. Save it:             agentforge keys set groq YOUR_KEY")
    click.echo("  3. Set agent model:     agentforge models set agent_loop groq/llama-3.3-70b-versatile")


@keys.command("test")
@click.argument("provider")
def keys_test(provider):
    """Send a test message to PROVIDER to verify the key works."""
    import os
    from src.llm.keys import load_keys_to_env, ENV_MAP, has_key
    load_keys_to_env()
    if not has_key(provider):
        click.echo(f"\n  ✗ No key for '{provider}'. Run: agentforge keys set {provider} <key>")
        sys.exit(1)
    test_models = {
        "anthropic":  "claude-haiku-4-5-20251001",
        "openai":     "gpt-4o-mini",
        "gemini":     "gemini/gemini-1.5-flash",
        "groq":       "groq/llama-3.1-8b-instant",
        "openrouter": "openrouter/auto",
        "together":   "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
    model = test_models.get(provider)
    if not model:
        click.echo(f"\n  ✗ No test model for '{provider}'")
        sys.exit(1)
    click.echo(f"\n  Testing {provider} ({model}) ...")
    try:
        from litellm import completion
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content.strip()
        click.echo(f"  ✓ {provider} key works  —  response: {answer}")
    except Exception as e:
        click.echo(f"  ✗ {provider} key failed: {e}", err=True)
        sys.exit(1)
