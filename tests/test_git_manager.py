"""Tests for GitManagerAgent — uses a real temp git repo, no network."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.git_manager import GitManagerAgent, _is_secret_file, _is_protected


class TestSecretFileDetection:
    def test_env_local_is_secret(self):
        assert _is_secret_file(".env.local")

    def test_env_is_secret(self):
        assert _is_secret_file(".env")

    def test_pem_is_secret(self):
        assert _is_secret_file("server.pem")

    def test_key_is_secret(self):
        assert _is_secret_file("private.key")

    def test_py_is_not_secret(self):
        assert not _is_secret_file("main.py")

    def test_requirements_not_secret(self):
        assert not _is_secret_file("requirements.txt")


class TestProtectedBranches:
    def test_main_is_protected(self):
        assert _is_protected("main")

    def test_master_is_protected(self):
        assert _is_protected("master")

    def test_feature_branch_not_protected(self):
        assert not _is_protected("feature/my-thing")

    def test_agent_branch_not_protected(self):
        assert not _is_protected("agent/dev-2026-05-26")


class TestGitManagerWithRealRepo:
    @pytest.fixture
    def git_manager(self, tmp_path):
        # Create a README so the initial commit has something
        (tmp_path / "README.md").write_text("# Test repo")
        gm = GitManagerAgent(working_dir=tmp_path)
        gm.init_repo()
        return gm

    def test_init_creates_repo(self, tmp_path):
        (tmp_path / "README.md").write_text("# Test")
        gm = GitManagerAgent(working_dir=tmp_path)
        result = gm.init_repo()
        assert result.success
        assert (tmp_path / ".git").exists()

    def test_is_git_repo(self, git_manager):
        assert git_manager.is_git_repo()

    def test_current_branch(self, git_manager):
        branch = git_manager.current_branch()
        assert branch  # not empty

    def test_create_feature_branch(self, git_manager):
        result = git_manager.ensure_branch("feature/test-work")
        assert result.success
        assert git_manager.current_branch() == "feature/test-work"

    def test_refuses_main_branch(self, git_manager):
        result = git_manager.ensure_branch("main")
        assert not result.success
        assert "protected" in result.message.lower()

    def test_refuses_master_branch(self, git_manager):
        result = git_manager.ensure_branch("master")
        assert not result.success

    def test_stage_file(self, git_manager, tmp_path):
        git_manager.ensure_branch("feature/staging-test")
        f = tmp_path / "new_file.py"
        f.write_text("x = 1")
        result = git_manager.stage_files(["new_file.py"])
        assert result.success

    def test_refuses_to_stage_env_file(self, git_manager, tmp_path):
        (tmp_path / ".env.local").write_text("SECRET=abc")
        result = git_manager.stage_files([".env.local"])
        assert not result.success
        assert "secret" in result.message.lower()

    def test_commit_on_feature_branch(self, git_manager, tmp_path):
        git_manager.ensure_branch("feature/commit-test")
        f = tmp_path / "module.py"
        f.write_text("def hello(): pass")
        git_manager.stage_files(["module.py"])
        result = git_manager.commit("add hello function", commit_type="feat")
        assert result.success
        assert result.commit_sha is not None

    def test_commit_refused_on_main(self, git_manager, tmp_path):
        # Force back to main (init left us on initial branch)
        try:
            git_manager.repo.git.checkout("main")
        except Exception:
            try:
                git_manager.repo.git.checkout("master")
            except Exception:
                pytest.skip("Could not switch to main/master")
        f = tmp_path / "sneaky.py"
        f.write_text("bad = True")
        git_manager.repo.index.add(["sneaky.py"])
        result = git_manager.commit("sneak commit", commit_type="feat")
        assert not result.success
        assert "protected" in result.message.lower()

    def test_list_branches(self, git_manager):
        git_manager.ensure_branch("feature/branch-a")
        git_manager.ensure_branch("feature/branch-b")
        branches = git_manager.list_branches()
        assert "feature/branch-a" in branches
        assert "feature/branch-b" in branches

    def test_status(self, git_manager, tmp_path):
        git_manager.ensure_branch("feature/status-test")
        status = git_manager.status()
        assert "branch" in status
        assert status["branch"] == "feature/status-test"
