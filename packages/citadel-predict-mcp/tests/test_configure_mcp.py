"""
Unit and integration tests for scripts/configure_mcp.py.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Import configure_mcp directly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from configure_mcp import (
    build_server_config,
    find_mcp_executable,
    get_claude_desktop_config_path,
    run_configuration,
    update_config_file,
)


def test_build_server_config():
    """Verify single source of truth configuration builder."""
    cfg = build_server_config(
        command_path="/path/to/citadel-predict-mcp",
        api_key="cp_live_test123",
        api_url="https://api.citadel.dev",
    )
    assert cfg["command"] == "/path/to/citadel-predict-mcp"
    assert cfg["env"]["CITADEL_API_KEY"] == "cp_live_test123"
    assert cfg["env"]["CITADEL_API_URL"] == "https://api.citadel.dev"


def test_build_server_config_no_key():
    """Verify configuration builder when no API key is specified."""
    cfg = build_server_config(
        command_path="/path/to/citadel-predict-mcp",
        api_key=None,
        api_url="http://localhost:8000",
    )
    assert cfg["command"] == "/path/to/citadel-predict-mcp"
    assert cfg["env"] == {"CITADEL_API_URL": "http://localhost:8000"}


def test_platform_desktop_paths():
    """Verify desktop config paths resolve correctly for Windows, macOS, and Linux."""
    win_path = get_claude_desktop_config_path("Windows")
    assert win_path.name == "claude_desktop_config.json"
    assert "Claude" in str(win_path)

    mac_path = get_claude_desktop_config_path("Darwin")
    assert mac_path == Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"

    linux_path = get_claude_desktop_config_path("Linux")
    assert linux_path.name == "claude_desktop_config.json"
    assert ".config" in str(linux_path) or "Claude" in str(linux_path)


def test_executable_detection():
    """Verify find_mcp_executable finds a real, existing executable on disk."""
    exe = find_mcp_executable()
    assert isinstance(exe, str)
    assert Path(exe).exists()
    assert Path(exe).is_file()


def test_run_configuration_and_valid_json(tmp_path):
    """
    Asserts:
    (a) Resulting JSON is valid in both Claude Desktop and Claude Code targets.
    (b) 'command' path actually exists on disk.
    """
    desktop_target = tmp_path / "desktop" / "claude_desktop_config.json"
    code_target = tmp_path / "code" / ".claude" / "settings.json"

    res = run_configuration(
        api_key="cp_live_secret_key",
        api_url="http://localhost:8000",
        desktop_config_path=desktop_target,
        code_config_path=code_target,
    )

    # Check files exist
    assert desktop_target.exists()
    assert code_target.exists()

    # (a) JSON is valid
    desktop_data = json.loads(desktop_target.read_text(encoding="utf-8"))
    code_data = json.loads(code_target.read_text(encoding="utf-8"))

    assert "mcpServers" in desktop_data
    assert "citadel-predict" in desktop_data["mcpServers"]
    assert "mcpServers" in code_data
    assert "citadel-predict" in code_data["mcpServers"]

    # (b) Command path exists on disk
    cmd = desktop_data["mcpServers"]["citadel-predict"]["command"]
    assert Path(cmd).exists()
    assert Path(cmd).is_file()
    assert desktop_data["mcpServers"]["citadel-predict"]["env"]["CITADEL_API_KEY"] == "cp_live_secret_key"
    assert desktop_data["mcpServers"]["citadel-predict"] == code_data["mcpServers"]["citadel-predict"]


def test_run_configuration_idempotency(tmp_path):
    """
    Asserts:
    (c) Running configure_mcp twice produces no diff on the second run.
    """
    desktop_target = tmp_path / "desktop.json"
    code_target = tmp_path / "code.json"

    res1 = run_configuration(
        api_key="cp_live_test",
        api_url="http://localhost:8000",
        desktop_config_path=desktop_target,
        code_config_path=code_target,
    )
    assert res1["targets"]["claude_desktop"]["modified"] is True
    assert res1["targets"]["claude_code"]["modified"] is True

    content1_desktop = desktop_target.read_text(encoding="utf-8")
    content1_code = code_target.read_text(encoding="utf-8")

    # Second run with exact same parameters
    res2 = run_configuration(
        api_key="cp_live_test",
        api_url="http://localhost:8000",
        desktop_config_path=desktop_target,
        code_config_path=code_target,
    )
    assert res2["targets"]["claude_desktop"]["modified"] is False
    assert res2["targets"]["claude_code"]["modified"] is False

    content2_desktop = desktop_target.read_text(encoding="utf-8")
    content2_code = code_target.read_text(encoding="utf-8")

    # Content must be identical (zero diff)
    assert content1_desktop == content2_desktop
    assert content1_code == content2_code


def test_preserves_existing_unrelated_configs(tmp_path):
    """
    Asserts:
    (d) Existing unrelated mcpServers entries and other top-level keys are preserved.
    """
    desktop_target = tmp_path / "claude_desktop_config.json"
    existing_content = {
        "theme": "dark",
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/example/Desktop"],
            },
        },
    }
    desktop_target.write_text(json.dumps(existing_content, indent=2), encoding="utf-8")

    run_configuration(
        api_key="cp_live_citadel",
        api_url="https://citadel.example.com",
        desktop_config_path=desktop_target,
        skip_code=True,
    )

    updated_data = json.loads(desktop_target.read_text(encoding="utf-8"))

    # Top level keys preserved
    assert updated_data.get("theme") == "dark"

    # Unrelated servers preserved intact
    assert "github" in updated_data["mcpServers"]
    assert updated_data["mcpServers"]["github"]["command"] == "npx"
    assert "filesystem" in updated_data["mcpServers"]

    # Citadel Predict added/updated
    assert "citadel-predict" in updated_data["mcpServers"]
    assert updated_data["mcpServers"]["citadel-predict"]["env"]["CITADEL_API_KEY"] == "cp_live_citadel"
    assert updated_data["mcpServers"]["citadel-predict"]["env"]["CITADEL_API_URL"] == "https://citadel.example.com"


def test_configure_mcp_cli_subprocess(tmp_path):
    """Verify running the CLI as a subprocess works end-to-end with flags."""
    desktop_target = tmp_path / "cli_desktop.json"
    code_target = tmp_path / "cli_code.json"

    script_path = REPO_ROOT / "scripts" / "configure_mcp.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--non-interactive",
        "--api-key",
        "cp_live_cli_test",
        "--api-url",
        "http://127.0.0.1:8000",
        "--desktop-config",
        str(desktop_target),
        "--code-config",
        str(code_target),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Configuration completed successfully." in proc.stdout

    data = json.loads(desktop_target.read_text(encoding="utf-8"))
    assert data["mcpServers"]["citadel-predict"]["env"]["CITADEL_API_KEY"] == "cp_live_cli_test"
