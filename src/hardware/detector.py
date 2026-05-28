"""
Hardware detection and model routing for AgentForge.
Detects GPU/CPU/RAM and selects optimal local model or API fallback.
"""

import platform
import subprocess
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import psutil


@dataclass
class HardwareProfile:
    os: str
    chip: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str
    vram_gb: float
    # Apple Silicon uses unified memory — Metal VRAM = portion of RAM available to GPU
    is_apple_silicon: bool
    metal_support: bool
    ollama_available: bool
    detected_at: str

    # Routing recommendations
    local_model: str | None
    api_model: str
    routing_note: str


def _get_apple_silicon_info() -> dict:
    """Extract Apple Silicon chip details via system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10
        )
        text = result.stdout
        chip = "Apple Silicon"
        cores = psutil.cpu_count(logical=False) or 0

        for line in text.splitlines():
            if "Chip:" in line:
                chip = line.split("Chip:")[-1].strip()
            elif "Total Number of Cores:" in line:
                # "10 (4 Super and 6 Efficiency)"
                cores_str = line.split(":")[-1].strip().split(" ")[0]
                try:
                    cores = int(cores_str)
                except ValueError:
                    pass

        return {"chip": chip, "cores": cores}
    except Exception:
        return {"chip": "Apple Silicon", "cores": psutil.cpu_count(logical=False) or 0}


def _get_metal_support() -> bool:
    """Check if Metal GPU acceleration is available (Apple Silicon/AMD Mac)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10
        )
        return "Metal" in result.stdout
    except Exception:
        return False


def _get_nvidia_vram() -> float:
    """Query NVIDIA GPU VRAM via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip()) / 1024  # MiB → GB
    except FileNotFoundError:
        pass
    return 0.0


def _is_ollama_available() -> bool:
    """Check if Ollama daemon is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _route_model(vram_gb: float, is_apple_silicon: bool, ram_gb: float) -> tuple[str | None, str, str]:
    """
    Determine local model + API fallback based on available VRAM.
    Apple Silicon unified memory: GPU can use a large portion of system RAM.
    We use 60% of RAM as effective VRAM estimate for routing purposes.
    Returns: (local_model, api_model, note)

    API model is optional — if ANTHROPIC_API_KEY is not set, all tasks
    route to the local model. Phase 1 works entirely without an API key.
    """
    effective_vram = vram_gb
    note = ""

    if is_apple_silicon:
        # Unified memory — GPU shares system RAM, conservatively estimate 60%
        effective_vram = ram_gb * 0.6
        note = f"Apple Silicon unified memory: ~{effective_vram:.0f}GB effective VRAM (60% of {ram_gb:.0f}GB RAM)"

    if effective_vram >= 24:
        return "qwen2.5-coder:32b", "claude-sonnet-4-6", note or "High VRAM: full local capability"
    elif effective_vram >= 16:
        return "qwen2.5-coder:14b", "claude-sonnet-4-6", note or "16GB VRAM: 14B model local"
    elif effective_vram >= 8:
        return "qwen2.5-coder:1.5b", "claude-haiku-4-5-20251001", note or "8GB VRAM: 1.5B model local"
    elif effective_vram >= 4:
        return "qwen2.5-coder:3b", "claude-haiku-4-5-20251001", note or "4GB VRAM: 3B model local"
    else:
        return None, "claude-haiku-4-5-20251001", "No GPU: API-only mode. Cost estimate shown per session."


def has_api_key() -> bool:
    """Check if an Anthropic API key is configured."""
    import os
    from pathlib import Path
    # Check env var first, then .env.local
    if os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-"):
        return True
    env_file = Path(".env.local")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY=sk-"):
                return True
    return False


def get_active_model(profile: "HardwareProfile") -> str:
    """
    Return the model to actually use right now.
    Falls back to local model if no API key is set.
    """
    if profile.local_model:
        return profile.local_model  # always prefer local when available
    if has_api_key():
        return profile.api_model
    return "No model available — set ANTHROPIC_API_KEY or install Ollama"


def detect_hardware() -> HardwareProfile:
    """Run full hardware detection and return a profile."""
    os_name = f"{platform.system()} {platform.mac_ver()[0] or platform.release()}"
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    is_apple = platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin"
    metal = False
    vram_gb = 0.0
    gpu_name = "None detected"

    if is_apple:
        apple_info = _get_apple_silicon_info()
        chip = apple_info["chip"]
        cpu_cores = apple_info["cores"] or psutil.cpu_count(logical=False) or 0
        metal = _get_metal_support()
        gpu_name = f"{chip} (unified memory)"
        # GPU uses system RAM; vram_gb left as 0 — routing uses ram_gb * 0.6
    else:
        chip = platform.processor()
        cpu_cores = psutil.cpu_count(logical=False) or 0
        vram_gb = _get_nvidia_vram()
        if vram_gb > 0:
            gpu_name = "NVIDIA GPU"

    ollama = _is_ollama_available()
    local_model, api_model, routing_note = _route_model(vram_gb, is_apple, ram_gb)

    return HardwareProfile(
        os=os_name,
        chip=chip,
        cpu_cores=cpu_cores,
        ram_gb=round(ram_gb, 1),
        gpu_name=gpu_name,
        vram_gb=round(vram_gb, 1),
        is_apple_silicon=is_apple,
        metal_support=metal,
        ollama_available=ollama,
        detected_at=datetime.now().isoformat(),
        local_model=local_model,
        api_model=api_model,
        routing_note=routing_note,
    )


def write_hardware_profile(profile: HardwareProfile, output_path: Path) -> None:
    """Write hardware_profile.md to .agentforge/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_display = profile.local_model or "None (API-only mode)"
    savings = "~70%" if profile.local_model else "0%"

    content = f"""# Hardware Profile
Generated: {profile.detected_at}

## Machine
OS: {profile.os}
Chip: {profile.chip}
CPU cores: {profile.cpu_cores}
RAM: {profile.ram_gb}GB

## GPU
GPU: {profile.gpu_name}
VRAM: {"N/A (unified memory)" if profile.is_apple_silicon else f"{profile.vram_gb}GB"}
Metal support: {"Yes" if profile.metal_support else "No"}

## Routing
Note: {profile.routing_note}
- Routine tasks (formatting, edits): {local_display}
- Feature implementation: {local_display}
- Architecture decisions: {profile.api_model} (API)
- Security analysis: {profile.api_model} (API — accuracy critical)
- Explainer/doc generation: {local_display}
- Market analysis: {profile.api_model} (API)

## Ollama
Status: {"Available" if profile.ollama_available else "Not running — start with: ollama serve"}
Install: https://ollama.ai
{"Recommended model: ollama pull " + profile.local_model if profile.local_model else "No local model recommended for this hardware tier"}

## Cost estimate
Estimated local savings vs full API: {savings}
"""

    output_path.write_text(content)


if __name__ == "__main__":
    profile = detect_hardware()
    out = Path(".agentforge/hardware_profile.md")
    write_hardware_profile(profile, out)
    print(f"Hardware profile written to {out}")
    print(f"  Chip: {profile.chip}")
    print(f"  RAM:  {profile.ram_gb}GB")
    print(f"  GPU:  {profile.gpu_name}")
    print(f"  Local model: {profile.local_model or 'None'}")
    print(f"  API model:   {profile.api_model}")
