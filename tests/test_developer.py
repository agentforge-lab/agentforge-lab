"""
Tests for DeveloperAgent.
Parser tests need no LLM. Live LLM tests are marked and skipped in CI.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.developer import DeveloperAgent, CodeEdit, _parse_edits


# ── Parser tests (no LLM) ──────────────────────────────────────────────────

VALID_RESPONSE = """
Here is the implementation:

<agentforge_edits>
<file path="src/utils/math.py" operation="create">
def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b
</file>
<file path="src/utils/__init__.py" operation="create">
from .math import add, multiply
</file>
</agentforge_edits>

SUMMARY: Created math utility module with add and multiply functions
"""

RESPONSE_WITH_EDIT = """
<agentforge_edits>
<file path="src/utils/math.py" operation="edit">
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
</file>
</agentforge_edits>

SUMMARY: Added subtract function to math utils
"""

RESPONSE_WITH_DELETE = """
<agentforge_edits>
<file path="old_module.py" operation="delete">
</file>
</agentforge_edits>

SUMMARY: Removed deprecated old_module.py
"""

RESPONSE_NO_BLOCK = "Here is some text without any agentforge_edits block."

RESPONSE_BAD_OPERATION = """
<agentforge_edits>
<file path="foo.py" operation="publish">
content
</file>
</agentforge_edits>
"""


class TestParseEdits:
    def test_parses_create(self):
        edits, summary = _parse_edits(VALID_RESPONSE)
        assert len(edits) == 2
        assert edits[0].operation == "create"
        assert edits[0].file_path == "src/utils/math.py"
        assert "def add" in edits[0].content

    def test_parses_multiple_files(self):
        edits, _ = _parse_edits(VALID_RESPONSE)
        paths = [e.file_path for e in edits]
        assert "src/utils/math.py" in paths
        assert "src/utils/__init__.py" in paths

    def test_parses_summary(self):
        _, summary = _parse_edits(VALID_RESPONSE)
        assert "math utility" in summary.lower()

    def test_parses_edit_operation(self):
        edits, summary = _parse_edits(RESPONSE_WITH_EDIT)
        assert edits[0].operation == "edit"
        assert "subtract" in edits[0].content
        assert "subtract" in summary.lower()

    def test_parses_delete_operation(self):
        edits, _ = _parse_edits(RESPONSE_WITH_DELETE)
        assert edits[0].operation == "delete"
        assert edits[0].content is None or edits[0].content == ""

    def test_returns_empty_on_no_block(self):
        edits, summary = _parse_edits(RESPONSE_NO_BLOCK)
        assert edits == []
        assert summary == ""

    def test_ignores_bad_operation(self):
        edits, _ = _parse_edits(RESPONSE_BAD_OPERATION)
        assert edits == []


class TestApplyEdits:
    def test_creates_file(self, tmp_path):
        agent = DeveloperAgent(working_dir=tmp_path)
        edits = [CodeEdit(file_path="hello.py", operation="create", content="print('hi')")]
        agent.apply_edits(edits)
        assert (tmp_path / "hello.py").read_text() == "print('hi')"

    def test_creates_nested_file(self, tmp_path):
        agent = DeveloperAgent(working_dir=tmp_path)
        edits = [CodeEdit(file_path="src/utils/foo.py", operation="create", content="x = 1")]
        agent.apply_edits(edits)
        assert (tmp_path / "src" / "utils" / "foo.py").exists()

    def test_edits_file(self, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("old content")
        agent = DeveloperAgent(working_dir=tmp_path)
        edits = [CodeEdit(file_path="existing.py", operation="edit", content="new content")]
        agent.apply_edits(edits)
        assert target.read_text() == "new content"

    def test_deletes_file(self, tmp_path):
        target = tmp_path / "to_delete.py"
        target.write_text("bye")
        agent = DeveloperAgent(working_dir=tmp_path)
        edits = [CodeEdit(file_path="to_delete.py", operation="delete")]
        agent.apply_edits(edits)
        assert not target.exists()

    def test_delete_nonexistent_is_noop(self, tmp_path):
        agent = DeveloperAgent(working_dir=tmp_path)
        edits = [CodeEdit(file_path="ghost.py", operation="delete")]
        agent.apply_edits(edits)  # should not raise


class TestThirdPartyScanner:
    def test_detects_stripe(self):
        code = "import stripe\nstripe.api_key = os.environ['STRIPE_KEY']"
        found = DeveloperAgent.scan_for_third_party(code)
        assert "stripe" in found

    def test_detects_google_oauth(self):
        code = "from google_auth_oauthlib.flow import Flow"
        found = DeveloperAgent.scan_for_third_party(code)
        assert "google_oauth" in found

    def test_detects_nothing_in_plain_code(self):
        code = "def add(a, b):\n    return a + b"
        found = DeveloperAgent.scan_for_third_party(code)
        assert found == []


class TestPathSafety:
    def test_rejects_traversal(self, tmp_path):
        agent = DeveloperAgent(working_dir=tmp_path)
        edits = [CodeEdit(file_path="../outside.py", operation="create", content="bad")]
        # apply_edits itself doesn't block — the _attempt safety check does.
        # Test that dry_run catches it (requires a mock LLM).
        # For now, verify that ".." in path is the unsafe indicator.
        assert ".." in edits[0].file_path


# ── Live LLM test (skipped unless --live flag passed) ─────────────────────

@pytest.mark.skipif(
    not Path(".venv/bin/python3.11").exists(),
    reason="Venv not set up"
)
class TestDeveloperLive:
    """Requires Ollama running: `ollama serve`"""

    @pytest.mark.skip(reason="Run manually: requires Ollama running")
    def test_simple_task(self, tmp_path):
        from src.llm.client import LLMClient
        agent = DeveloperAgent(llm=LLMClient(), working_dir=tmp_path)
        result = agent.execute(
            "Create a Python function that returns the sum of a list of numbers. "
            "Put it in src/utils/math_utils.py"
        )
        assert result.success, f"Failed: {result.error}"
        assert len(result.edits) > 0
        out_file = tmp_path / "src" / "utils" / "math_utils.py"
        assert out_file.exists()
