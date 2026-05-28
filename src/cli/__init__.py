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
def run(goal, auto_approve, dry_run, max_retries, working_dir, stream):
    """Plan, code, test, secure, and commit GOAL."""
    try:
        wd = _working_dir(working_dir)
    except click.BadParameter as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    click.echo(f"\nAgentForge  ▸  {goal}")
    click.echo(f"  Working dir : {wd}")
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
    )

    if stream:
        result = _stream_run(runner, goal)
    else:
        result = runner.run(goal)

    click.echo(str(result))
    sys.exit(0 if result.success else 1)


def _do_dry_run(goal: str, wd: Path) -> None:
    click.echo("  ▸ Planning task (dry-run — no LLM call)")
    planner = PlannerAgent()
    plan = planner.plan(goal)
    click.echo(f"\n  Task       : {plan.task_description}")
    click.echo(f"  Branch     : {plan.branch_name}")
    click.echo(f"  Plan items : {len(plan.steps) if hasattr(plan, 'steps') else 'n/a'}")
    click.echo("\n  (Dry-run complete — no files written, no commit made)")


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
@click.argument("agent", type=click.Choice(["planner", "developer", "tester", "default"]))
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
