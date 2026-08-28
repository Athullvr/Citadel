"""
Tests for repository safeguards, frozen baseline integrity checks, and CI configuration.
"""

import sys
from pathlib import Path
from unittest.mock import patch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from check_frozen_baseline import FROZEN_FILES, check_staged_baseline


def test_frozen_baseline_files_exist_and_listed():
    """Verify that both frozen baseline files exist on disk and are covered by the check."""
    for f in FROZEN_FILES:
        target = REPO_ROOT / f
        assert target.exists(), f"Protected frozen artifact missing on disk: {target}"
        assert target.is_file()


def test_safeguard_passes_when_no_frozen_files_staged():
    """Verify safeguard returns True when unrelated files are staged."""
    with patch("check_frozen_baseline.get_staged_files", return_value=["backend/main.py", "frontend/src/app/page.tsx"]):
        assert check_staged_baseline() is True


def test_safeguard_blocks_when_claude_sonnet_joblib_staged(capsys):
    """Verify safeguard returns False and outputs error when claude-sonnet.joblib is staged."""
    with patch(
        "check_frozen_baseline.get_staged_files",
        return_value=["data_collection/calibration_data/claude-sonnet.joblib", "backend/main.py"],
    ):
        assert check_staged_baseline() is False
        captured = capsys.readouterr()
        assert "CRITICAL INVARIANT VIOLATION" in captured.err
        assert "claude-sonnet.joblib" in captured.err


def test_safeguard_blocks_when_runs_jsonl_staged(capsys):
    """Verify safeguard returns False and outputs error when runs.jsonl is staged."""
    with patch(
        "check_frozen_baseline.get_staged_files",
        return_value=["data_collection/data/runs.jsonl"],
    ):
        assert check_staged_baseline() is False
        captured = capsys.readouterr()
        assert "CRITICAL INVARIANT VIOLATION" in captured.err
        assert "runs.jsonl" in captured.err


def test_ci_workflow_yaml_syntax_valid():
    """Verify that .github/workflows/ci.yml is valid YAML and contains required verification steps."""
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists()
    content = ci_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    assert "jobs" in data
    assert "backend-ci" in data["jobs"]
    steps = data["jobs"]["backend-ci"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert any("Frozen Baseline" in name for name in step_names)
    assert any("Pytest" in name for name in step_names)
