"""
Developer Agent — writes, edits, and refactors code.
Always operates on a dedicated branch, never touches main.
Retry logic: 3 attempts with modified prompt → escalate to Planner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient, LLMResponse
from src.llm.prompts import DEVELOPER_SYSTEM, DEVELOPER_RETRY_SUFFIX


@dataclass
class CodeEdit:
    file_path: str
    operation: str  # "create" | "edit" | "delete"
    content: str | None = None
    description: str = ""


@dataclass
class DeveloperResult:
    success: bool
    edits: list[CodeEdit] = field(default_factory=list)
    summary: str = ""
    error: str | None = None
    retry_count: int = 0
    escalated: bool = False
    response: LLMResponse | None = None


# ── Response parsers ───────────────────────────────────────────────────────

_FILE_RE = re.compile(
    r'<file\s+path="([^"]+)"\s+operation="([^"]+)">(.*?)</file>',
    re.DOTALL,
)
_SUMMARY_RE    = re.compile(r"^SUMMARY:\s*(.+)$", re.MULTILINE)
_EDITS_BLOCK_RE = re.compile(r"<agentforge_edits>(.*?)</agentforge_edits>", re.DOTALL)
_CODE_FENCE_RE  = re.compile(r"^```[^\n]*\n(.*?)```\s*$", re.DOTALL)


def _strip_code_fence(content: str) -> str:
    """Remove ```lang...``` wrapper that some models add inside <file> tags."""
    m = _CODE_FENCE_RE.match(content.strip())
    return m.group(1) if m else content


def _parse_edits(text: str) -> tuple[list[CodeEdit], str]:
    """Extract CodeEdit list and summary line from LLM response text."""
    edits: list[CodeEdit] = []
    block_match = _EDITS_BLOCK_RE.search(text)
    if not block_match:
        return edits, ""
    for m in _FILE_RE.finditer(block_match.group(1)):
        path, op, content = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if op not in ("create", "edit", "delete"):
            continue
        edits.append(CodeEdit(
            file_path=path,
            operation=op,
            content=_strip_code_fence(content) if op != "delete" else None,
        ))
    summary_match = _SUMMARY_RE.search(text)
    summary = summary_match.group(1).strip() if summary_match else ""
    return edits, summary


# ── Security: protected file patterns ─────────────────────────────────────

# Hard-blocked: never written or overwritten, regardless of goal
_HARD_BLOCKED_WRITE = {
    # Live credentials
    ".env.production", ".env.staging", ".env.prod",
    # Certificates and private keys
    "*.pem", "*.key", "*.p12", "*.pfx", "*.crt", "*.cer", "*.ca-bundle",
    # SSH keys
    "id_rsa", "id_rsa.pub", "id_ed25519", "id_ed25519.pub",
    # Credential files
    "credentials.json", "service-account.json",
    "*.secret",
}

# Soft-blocked: blocked unless the user's goal explicitly requests them
_SOFT_BLOCKED_WRITE = {
    # Databases — never overwrite existing data
    "*.db", "*.sqlite", "*.sqlite3", "*.db3", "*.s3db",
    # All generic env files
    ".env", ".env.local", ".env.*", "*.env",
    # Infrastructure
    "docker-compose.yml", "docker-compose*.yml",
    "Dockerfile", "Dockerfile.*",
    "*.tf", "*.tfvars",
    "Makefile",
    # CI/CD — use directory prefix ".github/" so startswith() works correctly
    ".github/", ".gitlab-ci.yml", ".travis.yml", "circle.yml",
    # Secrets files
    "secrets.json", "secrets.yaml", "secrets.yml",
}

# Hard-blocked for DELETE: things the agent can never delete
_HARD_BLOCKED_DELETE = _HARD_BLOCKED_WRITE | {
    # Database migrations — irreversible schema changes
    "migrations/*", "alembic/*",
    # SQL files
    "*.sql",
    # Dependency manifests
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "Cargo.toml", "go.mod", "go.sum",
    # Project root docs
    "README.md", "CHANGELOG.md", "LICENSE",
}

# Keywords that unlock soft-blocked files when present in the goal
_SOFT_BLOCK_UNLOCK: dict[str, list[str]] = {
    "dockerfile":         ["docker", "container", "containerize", "image"],
    "docker-compose":     ["docker", "compose", "container", "orchestrate"],
    "makefile":           ["makefile", "make ", "build script"],
    ".github":            ["ci", "github action", "workflow", "pipeline"],
    ".gitlab-ci":         ["ci", "gitlab", "pipeline", "workflow"],
    ".travis":            ["ci", "travis", "pipeline"],
    "circle.yml":         ["ci", "circleci", "pipeline"],
    ".env":               ["env file", "environment file", "dotenv"],
    ".tf":                ["terraform", "infrastructure", "iac"],
    ".tfvars":            ["terraform", "tfvars"],
    # Database files — only when explicitly creating a db
    ".db":                ["database file", "sqlite file", "create database"],
    ".sqlite":            ["database file", "sqlite file", "create database"],
    ".sqlite3":           ["database file", "sqlite file", "create database"],
    # Secrets config files
    "secrets.":           ["secrets file", "vault", "secret config"],
}


def _matches_pattern(filename: str, pattern: str) -> bool:
    """Simple glob-style match: exact, prefix*, or *suffix."""
    name = Path(filename).name.lower()
    pat = pattern.lower()
    if pat.startswith("*") and pat.endswith("*"):
        return pat[1:-1] in name
    if pat.startswith("*"):
        return name.endswith(pat[1:])
    if pat.endswith("*"):
        return name.startswith(pat[:-1])
    return name == pat


def _is_hard_blocked(path: str, patterns: set[str]) -> bool:
    name = Path(path).name
    path_lower = path.lower()
    for pat in patterns:
        if "/" in pat:
            # Directory-scoped pattern
            if path_lower.startswith(pat.rstrip("*").lower()):
                return True
        elif _matches_pattern(name, pat):
            return True
    return False


def _is_goal_scoped(path: str, goal: str) -> bool:
    """Return True if this soft-blocked file type was explicitly requested in goal."""
    name = Path(path).name.lower()
    path_lower = path.lower()
    goal_lower = goal.lower()
    for trigger, keywords in _SOFT_BLOCK_UNLOCK.items():
        # Match filename OR full path (catches .github/workflows/ci.yml etc.)
        if trigger in name or trigger in path_lower or name.startswith(trigger.rstrip("*.")):
            return any(kw in goal_lower for kw in keywords)
    # File is soft-blocked but has no unlock trigger defined — block by default
    return False


_PLACEHOLDER_SEGMENTS = frozenset({
    "relative", "path", "to", "your", "file", "here", "example",
    "placeholder", "somefile", "myfile",
})


def _looks_like_placeholder(path: str) -> bool:
    """Return True if the path looks like a copied template placeholder."""
    parts = Path(path).parts
    if len(parts) >= 3:
        generic = sum(1 for p in parts if p.lower() in _PLACEHOLDER_SEGMENTS)
        if generic >= 2:
            return True
    stem = Path(path).stem.lower()
    if stem in ("file", "filename", "yourfile", "myfile", "somefile"):
        return True
    return False


def check_path_allowed(path: str, operation: str, goal: str = "") -> str | None:
    """
    Return an error string if this path+operation is not allowed, else None.
    Public so tests and the graph node can call it directly.
    """
    # 1. Basic path safety
    if ".." in path or path.startswith("/"):
        return f"Unsafe path rejected: '{path}' contains '..' or absolute reference"

    # 2. Placeholder path detection — the LLM copied the example from the prompt
    if _looks_like_placeholder(path):
        return (
            f"Placeholder path rejected: '{path}' looks like a template example, "
            "not a real filename. Use a meaningful name like 'app.py' or 'models/user.py'."
        )

    # 3. Hard-blocked writes
    if operation in ("create", "edit"):
        if _is_hard_blocked(path, _HARD_BLOCKED_WRITE):
            return (
                f"Hard-blocked: '{path}' is a protected file that AgentForge "
                "never overwrites (credentials, certificates, live data)."
            )
        # Soft-blocked writes: allowed only if goal explicitly mentions them
        if _is_hard_blocked(path, _SOFT_BLOCKED_WRITE) and not _is_goal_scoped(path, goal):
            return (
                f"Soft-blocked: '{path}' is an infrastructure/config file. "
                "AgentForge won't touch it unless your goal explicitly requests it. "
                "Re-run with a goal that mentions this file type."
            )

    # 3. Hard-blocked deletes
    if operation == "delete":
        if _is_hard_blocked(path, _HARD_BLOCKED_DELETE):
            return (
                f"Delete refused: '{path}' is protected from deletion "
                "(dependency manifest, migration, credential, or infrastructure file)."
            )

    return None


# ── Security: secrets detection ────────────────────────────────────────────

# Patterns that indicate a hardcoded secret in generated code
_SECRET_REGEXES: list[re.Pattern] = [
    re.compile(r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][^"\']{8,}["\']'),
    re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{3,}["\']'),
    re.compile(r'(?i)(secret[_-]?key|secret|token|auth_token)\s*[=:]\s*["\'][^"\']{8,}["\']'),
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),            # OpenAI/Anthropic keys
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),             # GitHub PATs
    re.compile(r'AKIA[0-9A-Z]{16}'),                # AWS access key IDs
    re.compile(r'(?i)bearer\s+[a-zA-Z0-9+/._-]{20,}'),
]

# Patterns that are false positives (test values, env var references)
_SECRET_ALLOWLIST: list[re.Pattern] = [
    re.compile(r'os\.environ'),
    re.compile(r'os\.getenv'),
    re.compile(r'getenv\('),
    re.compile(r'environ\.get'),
    re.compile(r'YOUR_.*_HERE'),
    re.compile(r'<.*>'),                # placeholder like <your-key>
    re.compile(r'example|placeholder|dummy|fake|test|mock', re.IGNORECASE),
]


def scan_for_secrets(code: str) -> list[str]:
    """
    Return a list of finding descriptions if hardcoded secrets are detected.
    Returns empty list if clean.
    """
    findings = []
    lines = code.splitlines()
    for i, line in enumerate(lines, 1):
        # Skip lines that look like env var references or test values
        if any(p.search(line) for p in _SECRET_ALLOWLIST):
            continue
        for pattern in _SECRET_REGEXES:
            if pattern.search(line):
                # Redact the actual value before logging
                findings.append(f"Line {i}: possible hardcoded secret matched pattern '{pattern.pattern[:40]}'")
                break
    return findings


# ── Third-party service detection ──────────────────────────────────────────

THIRD_PARTY_PATTERNS: dict[str, str] = {
    "google_oauth":  r"(google.*oauth|oauth2client|google[-_]auth)",
    "stripe":        r"\bstripe\b",
    "sendgrid":      r"\bsendgrid\b",
    "twilio":        r"\btwilio\b",
    "aws":           r"\bboto3\b|aws[_-]sdk|@aws-sdk",
    "supabase":      r"\bsupabase\b",
    "auth0":         r"\bauth0\b",
    "clerk":         r"\bclerk\b",
    "resend":        r"\bresend\b",
    "neon":          r"\bneon\b.*postgres|neontech",
    "planetscale":   r"\bplanetscale\b",
}


# ── Stdlib import auto-fixer ──────────────────────────────────────────────

_STDLIB_MODULES = frozenset({
    "string", "os", "re", "sys", "random", "math", "json",
    "collections", "itertools", "functools", "datetime", "pathlib",
    "typing", "time", "hashlib", "base64", "uuid", "argparse",
})


def _fix_missing_imports(code: str) -> str:
    """Auto-add stdlib imports that are used (as `module.X`) but not imported."""
    existing: set[str] = set()
    for line in code.splitlines():
        m = re.match(r"^import\s+(\w+)", line)
        if m:
            existing.add(m.group(1))
        m = re.match(r"^from\s+(\w+)\s+import", line)
        if m:
            existing.add(m.group(1))

    needed = sorted(
        mod for mod in _STDLIB_MODULES
        if f"{mod}." in code and mod not in existing
    )
    if not needed:
        return code

    lines = code.splitlines()
    last_import = -1
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import = i
    insert_at = last_import + 1 if last_import >= 0 else 0
    for offset, mod in enumerate(needed):
        lines.insert(insert_at + offset, f"import {mod}")
    return "\n".join(lines)


# ── Agent ──────────────────────────────────────────────────────────────────

class DeveloperAgent:
    """
    Executes code generation tasks assigned by the Planner.
    Calls the LLM, parses the structured response into CodeEdit objects,
    applies them to disk, and handles retry/escalation.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        llm: LLMClient | None = None,
        working_dir: Path = Path("."),
        goal: str = "",
    ):
        self.llm = llm or LLMClient()
        self.working_dir = working_dir
        self.goal = goal          # used for soft-block intent checks
        self._retry_count = 0
        self._last_error: str = ""

    # ── Public API ─────────────────────────────────────────────────────────

    def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> DeveloperResult:
        """
        Given a task description, call the LLM, parse edits, apply to disk.
        Retries up to MAX_RETRIES times with error feedback. Escalates on failure.
        """
        self._retry_count = 0
        self._last_error = ""

        while self._retry_count < self.MAX_RETRIES:
            result = self._attempt(task, context or {})
            if result.success:
                self.apply_edits(result.edits)
                return result
            self._retry_count += 1
            self._last_error = result.error or "Unknown error"

        return DeveloperResult(
            success=False,
            error=f"Failed after {self.MAX_RETRIES} attempts. Last error: {self._last_error}",
            retry_count=self._retry_count,
            escalated=True,
        )

    def dry_run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> DeveloperResult:
        """Same as execute() but does NOT write files to disk."""
        self._retry_count = 0
        self._last_error = ""
        return self._attempt(task, context or {})

    # ── Internal ──────────────────────────────────────────────────────────

    def _attempt(self, task: str, context: dict) -> DeveloperResult:
        system = self._build_system_prompt()
        user = self._build_user_prompt(task, context)

        try:
            response = self.llm.complete(system, user)
        except Exception as e:
            return DeveloperResult(success=False, error=f"LLM call failed: {e}")

        edits, summary = _parse_edits(response.content)

        if not edits:
            return DeveloperResult(
                success=False,
                error="No <agentforge_edits> block found in LLM response",
                response=response,
            )

        # Security validation on every edit before touching disk
        for edit in edits:
            path_error = check_path_allowed(edit.file_path, edit.operation, self.goal)
            if path_error:
                return DeveloperResult(
                    success=False,
                    error=path_error,
                    response=response,
                )

            # Secrets scan on generated content
            if edit.content:
                findings = scan_for_secrets(edit.content)
                if findings:
                    finding_summary = "; ".join(findings[:3])
                    return DeveloperResult(
                        success=False,
                        error=(
                            f"Hardcoded secret detected in '{edit.file_path}': {finding_summary}. "
                            "Use os.environ.get('VAR_NAME') instead of hardcoded values."
                        ),
                        response=response,
                    )

        return DeveloperResult(
            success=True,
            edits=edits,
            summary=summary,
            retry_count=self._retry_count,
            response=response,
        )

    def _build_system_prompt(self) -> str:
        if self._retry_count == 0:
            return DEVELOPER_SYSTEM
        hint = "syntax and imports"
        if self._last_error and "secret" in self._last_error.lower():
            hint = "never hardcode credentials — use os.environ.get('VAR_NAME')"
        elif self._last_error and "blocked" in self._last_error.lower():
            hint = "avoid writing infrastructure or credential files"
        elif self._last_error and "path" in self._last_error.lower():
            hint = "file paths"
        elif self._last_error and "parse" in self._last_error.lower():
            hint = "the exact <agentforge_edits> output format"
        return DEVELOPER_SYSTEM + DEVELOPER_RETRY_SUFFIX.format(
            attempt=self._retry_count + 1,
            error=self._last_error[:300],   # truncate long errors to save context
            hint=hint,
        )

    def _build_user_prompt(self, task: str, context: dict) -> str:
        parts: list[str] = []
        if context.get("project_context"):
            parts.append(f"## Project context\n{context['project_context'][:1000]}")
        if context.get("existing_files"):
            for path, content in context["existing_files"].items():
                # Truncate large files to preserve context window
                truncated = content[:6000]
                suffix = "\n# ... (truncated)" if len(content) > 6000 else ""
                parts.append(f"## Existing file: {path}\n```\n{truncated}{suffix}\n```")
        if context.get("previous_error"):
            err_text = context["previous_error"]
            # Send in full — truncating hides the actual assertion errors the model needs to fix
            parts.append(f"## Previous error\n{err_text}")
        parts.append(f"## Task\n{task}")
        return "\n\n".join(parts)

    # ── Disk operations ───────────────────────────────────────────────────

    def apply_edits(self, edits: list[CodeEdit]) -> None:
        """Write edits to disk. Called automatically by execute()."""
        for edit in edits:
            path = self.working_dir / edit.file_path
            if edit.operation in ("create", "edit"):
                path.parent.mkdir(parents=True, exist_ok=True)
                content = edit.content or ""
                if edit.file_path.endswith(".py"):
                    content = _fix_missing_imports(content)
                path.write_text(content)
            elif edit.operation == "delete":
                if path.exists():
                    path.unlink()

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def scan_for_third_party(code: str) -> list[str]:
        """Return names of third-party services detected in generated code."""
        found = []
        for service, pattern in THIRD_PARTY_PATTERNS.items():
            if re.search(pattern, code, re.IGNORECASE):
                found.append(service)
        return found

    @staticmethod
    def parse_edits(text: str) -> tuple[list[CodeEdit], str]:
        """Public wrapper around the response parser (useful for testing)."""
        return _parse_edits(text)
