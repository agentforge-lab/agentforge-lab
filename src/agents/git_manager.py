"""
Git Manager Agent — all git operations via GitPython.
Rules: never force-push, never touch main, structured commit messages.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import git
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    _GIT_AVAILABLE = True
except ImportError:
    _GIT_AVAILABLE = False


@dataclass
class GitResult:
    success: bool
    message: str
    branch: str | None = None
    commit_sha: str | None = None
    pr_url: str | None = None
    changed_files: list[str] = field(default_factory=list)


# Branches that are never touched directly
_PROTECTED_BRANCHES = {"main", "master", "develop", "production", "release"}

# Conventional commit types
COMMIT_TYPES = {
    "feat":     "new feature",
    "fix":      "bug fix",
    "test":     "tests added or updated",
    "refactor": "code refactoring",
    "docs":     "documentation",
    "chore":    "maintenance or build",
    "security": "security fix",
    "perf":     "performance improvement",
}


def _is_protected(branch: str) -> bool:
    return branch.lower() in _PROTECTED_BRANCHES


class GitManagerAgent:
    """
    Handles all git operations for AgentForge.
    Creates feature branches, stages files, commits, pushes, opens PRs.
    """

    def __init__(self, working_dir: Path = Path(".")):
        self.working_dir = working_dir
        self._repo: "Repo | None" = None

    # ── Repo access ────────────────────────────────────────────────────────

    @property
    def repo(self) -> "Repo":
        if not _GIT_AVAILABLE:
            raise RuntimeError("gitpython not installed: pip install gitpython")
        if self._repo is None:
            try:
                self._repo = git.Repo(self.working_dir)
            except InvalidGitRepositoryError:
                raise RuntimeError(
                    f"No git repository at {self.working_dir}. "
                    "Run `git init` first or use init_repo()."
                )
        return self._repo

    def init_repo(self) -> GitResult:
        """Initialise a new git repository in working_dir."""
        if not _GIT_AVAILABLE:
            return GitResult(success=False, message="gitpython not installed")
        try:
            self._repo = git.Repo.init(self.working_dir)
            # Create an initial commit so HEAD exists
            readme = self.working_dir / "README.md"
            if readme.exists():
                self._repo.index.add(["README.md"])
                self._repo.index.commit(
                    "chore: initial commit\n\nAgentForge project initialised."
                )
            return GitResult(success=True, message="Git repository initialised", branch="main")
        except Exception as e:
            return GitResult(success=False, message=str(e))

    def is_git_repo(self) -> bool:
        try:
            git.Repo(self.working_dir)
            return True
        except Exception:
            return False

    # ── Branch management ──────────────────────────────────────────────────

    def current_branch(self) -> str:
        try:
            return self.repo.active_branch.name
        except TypeError:
            return "HEAD (detached)"

    def ensure_branch(self, branch_name: str) -> GitResult:
        """
        Switch to branch_name, creating it if it doesn't exist.
        Refuses if branch_name is a protected branch.
        """
        if _is_protected(branch_name):
            return GitResult(
                success=False,
                message=f"Refused: '{branch_name}' is a protected branch. Use a feature branch.",
            )
        try:
            existing = [b.name for b in self.repo.branches]
            if branch_name in existing:
                self.repo.git.checkout(branch_name)
                return GitResult(success=True, message=f"Checked out existing branch '{branch_name}'", branch=branch_name)
            else:
                self.repo.git.checkout("-b", branch_name)
                return GitResult(success=True, message=f"Created and checked out branch '{branch_name}'", branch=branch_name)
        except GitCommandError as e:
            return GitResult(success=False, message=str(e))

    def list_branches(self) -> list[str]:
        return [b.name for b in self.repo.branches]

    # ── Staging & committing ───────────────────────────────────────────────

    def stage_files(self, paths: list[str]) -> GitResult:
        """Stage specific files. Refuses to stage .env.local or secrets."""
        blocked = [p for p in paths if _is_secret_file(p)]
        if blocked:
            return GitResult(
                success=False,
                message=f"Refused to stage secret files: {blocked}",
            )
        try:
            self.repo.index.add(paths)
            return GitResult(success=True, message=f"Staged {len(paths)} file(s)", changed_files=paths)
        except Exception as e:
            return GitResult(success=False, message=str(e))

    def stage_all_tracked(self) -> GitResult:
        """Stage all modified tracked files (equivalent to git add -u)."""
        try:
            self.repo.git.add("-u")
            staged = [item.a_path for item in self.repo.index.diff("HEAD")]
            return GitResult(success=True, message=f"Staged {len(staged)} tracked file(s)", changed_files=staged)
        except Exception as e:
            return GitResult(success=False, message=str(e))

    def commit(
        self,
        title: str,
        body: str = "",
        commit_type: str = "feat",
        agent_name: str = "AgentForge",
    ) -> GitResult:
        """
        Create a conventional commit. Never on protected branches.
        Format: '{type}: {title}\n\n{body}\n\nGenerated by {agent_name}'
        """
        current = self.current_branch()
        if _is_protected(current):
            return GitResult(
                success=False,
                message=f"Refused: on protected branch '{current}'. Switch to a feature branch first.",
                branch=current,
            )

        if commit_type not in COMMIT_TYPES:
            commit_type = "feat"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        message_parts = [f"{commit_type}: {title}"]
        if body:
            message_parts.append(f"\n{body}")
        message_parts.append(f"\nGenerated by {agent_name} · {timestamp}")
        message = "\n".join(message_parts)

        try:
            commit_obj = self.repo.index.commit(message)
            return GitResult(
                success=True,
                message=f"Committed: {commit_type}: {title}",
                branch=current,
                commit_sha=commit_obj.hexsha[:8],
            )
        except Exception as e:
            return GitResult(success=False, message=str(e), branch=current)

    # ── Push & PR ─────────────────────────────────────────────────────────

    def push(self, remote: str = "origin") -> GitResult:
        """Push current branch to remote. Never force-pushes."""
        branch = self.current_branch()
        if _is_protected(branch):
            return GitResult(
                success=False,
                message=f"Refused to push to protected branch '{branch}'.",
                branch=branch,
            )
        try:
            origin = self.repo.remote(remote)
            origin.push(branch)
            return GitResult(success=True, message=f"Pushed '{branch}' to {remote}", branch=branch)
        except Exception as e:
            return GitResult(success=False, message=str(e), branch=branch)

    def open_pr(self, title: str, body: str = "") -> GitResult:
        """
        Open a GitHub pull request using the `gh` CLI.
        Falls back gracefully if gh is not installed or no remote is configured.
        """
        branch = self.current_branch()
        try:
            result = subprocess.run(
                ["gh", "pr", "create",
                 "--title", title,
                 "--body", body or f"Generated by AgentForge\nBranch: {branch}",
                 "--base", "main"],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.working_dir),
            )
            if result.returncode == 0:
                pr_url = result.stdout.strip().splitlines()[-1]
                return GitResult(success=True, message="PR created", branch=branch, pr_url=pr_url)
            return GitResult(
                success=False,
                message=f"gh pr create failed: {result.stderr.strip()}",
                branch=branch,
            )
        except FileNotFoundError:
            return GitResult(
                success=False,
                message=(
                    "GitHub CLI (gh) not found. Install it: https://cli.github.com\n"
                    f"Or open a PR manually from branch: {branch}"
                ),
                branch=branch,
            )
        except Exception as e:
            return GitResult(success=False, message=str(e), branch=branch)

    # ── Status & diff ─────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return working tree status: staged, modified, untracked files."""
        try:
            staged    = [item.a_path for item in self.repo.index.diff("HEAD")]
            modified  = [item.a_path for item in self.repo.index.diff(None)]
            untracked = self.repo.untracked_files
            return {
                "branch": self.current_branch(),
                "staged": staged,
                "modified": modified,
                "untracked": untracked,
                "is_dirty": self.repo.is_dirty(),
            }
        except Exception as e:
            return {"error": str(e)}

    def diff_staged(self) -> str:
        """Return unified diff of staged changes."""
        try:
            return self.repo.git.diff("--cached")
        except Exception:
            return ""


# ── Helpers ────────────────────────────────────────────────────────────────

_SECRET_PATTERNS = {
    # Environment files
    ".env", ".env.local", ".env.production", ".env.staging",
    # SSH private keys
    "id_rsa", "id_ed25519",
    # Certificates and keys
    "*.pem", "*.key", "*.p12", "*.pfx", "*.crt",
    # Credential files
    "credentials.json", "service-account.json",
    "secrets.json", "secrets.yaml", "secrets.yml",
    "*.secret",
    # Databases (never stage live data)
    "*.db", "*.sqlite", "*.sqlite3",
}


def _is_secret_file(path: str) -> bool:
    name = Path(path).name.lower()
    for p in _SECRET_PATTERNS:
        if p.startswith("*."):
            if name.endswith(p[1:]):
                return True
        elif p.endswith(".*"):
            # e.g. ".env.*" — match ".env.anything"
            prefix = p[:-2]
            if name == prefix or name.startswith(prefix + "."):
                return True
        else:
            if name == p:
                return True
    return False
