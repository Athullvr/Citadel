#!/usr/bin/env python3
"""
scripts/configure_mcp.py — Citadel Predict MCP Auto-Configuration Tool.

Configures the Model Context Protocol (MCP) server for both Claude Desktop
and Claude Code across Windows, macOS, and Linux without hardcoding user- or
OS-specific paths.

Usage:
  python scripts/configure_mcp.py
  python scripts/configure_mcp.py --api-key cp_live_12345 --api-url https://api.citadel.dev
  python scripts/configure_mcp.py --non-interactive
"""

import argparse
import json
import os
import platform
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import Any, Optional


def find_repo_root() -> Path:
    """Find the root directory of the Citadel repository."""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "pyproject.toml").exists() or (candidate / "packages").exists():
        return candidate
    return Path.cwd()


def get_claude_desktop_config_path(system_name: Optional[str] = None) -> Path:
    """
    Returns the platform-specific path to claude_desktop_config.json.

    - Windows: %APPDATA%/Claude/claude_desktop_config.json
    - macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
    - Linux:   ~/.config/Claude/claude_desktop_config.json (or $XDG_CONFIG_HOME)
    """
    system = system_name or platform.system()

    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (Path.home() / "AppData" / "Roaming")
        return base / "Claude" / "claude_desktop_config.json"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:  # Linux and other Unix-like OSes
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg_config) if xdg_config else (Path.home() / ".config")
        return base / "Claude" / "claude_desktop_config.json"


def get_claude_code_settings_path(project_dir: Optional[Path] = None) -> Path:
    """Returns the project-level .claude/settings.json path for Claude Code."""
    root = project_dir or find_repo_root()
    return root / ".claude" / "settings.json"


def find_mcp_executable(custom_path: Optional[str] = None) -> str:
    """
    Locates the installed citadel-predict-mcp executable.

    Checks:
    1. Explicit custom path if provided.
    2. shutil.which('citadel-predict-mcp') on PATH.
    3. sys.prefix / Scripts (Windows) or sys.prefix / bin (Unix).
    4. sysconfig 'scripts' path.
    5. Directory of active python interpreter.
    6. Relative virtualenvs in the Citadel repo (e.g. data_collection/.venv).
    """
    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_file():
            return str(p)
        raise RuntimeError(f"Specified command path does not exist: {custom_path}")

    exe_name = "citadel-predict-mcp.exe" if platform.system() == "Windows" else "citadel-predict-mcp"

    # 1. Check active virtual environment / Python prefix first
    prefix = Path(sys.prefix)
    prefix_candidates = [
        prefix / "Scripts" / exe_name,
        prefix / "bin" / exe_name,
        Path(sysconfig.get_path("scripts")) / exe_name,
        Path(sys.executable).parent / exe_name,
    ]

    for candidate in prefix_candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    # 2. Check system PATH
    which_path = shutil.which("citadel-predict-mcp")
    if which_path and Path(which_path).is_file():
        return str(Path(which_path).resolve())

    # 3. Check known repo venv location
    repo_root = find_repo_root()
    repo_venv_candidates = [
        repo_root / "data_collection" / ".venv" / "Scripts" / exe_name,
        repo_root / "data_collection" / ".venv" / "bin" / exe_name,
        repo_root / ".venv" / "Scripts" / exe_name,
        repo_root / ".venv" / "bin" / exe_name,
    ]

    for candidate in repo_venv_candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    raise RuntimeError(
        "Executable 'citadel-predict-mcp' was not found in the active environment.\n\n"
        "Please install the package into your Python environment first:\n"
        "  pip install -e packages/citadel-predict-mcp\n"
    )


def build_server_config(
    command_path: str,
    api_key: Optional[str] = None,
    api_url: str = "https://citadel-7j9u.onrender.com",
) -> dict[str, Any]:
    """
    Single source of truth for generating the MCP server configuration block.
    Guarantees that Claude Desktop and Claude Code configurations remain identical.
    """
    env_vars: dict[str, str] = {}
    if api_key:
        env_vars["CITADEL_API_KEY"] = api_key
    if api_url:
        env_vars["CITADEL_API_URL"] = api_url

    config: dict[str, Any] = {"command": command_path}
    if env_vars:
        config["env"] = env_vars
    return config


def update_config_file(
    file_path: Path,
    server_config: dict[str, Any],
    server_name: str = "citadel-predict",
) -> tuple[bool, dict[str, Any]]:
    """
    Safely merges server_config into file_path under 'mcpServers'.
    Preserves all existing, unrelated configuration.

    Returns:
        (is_modified, resulting_data)
    """
    file_path = file_path.resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if file_path.exists():
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                data = json.loads(content)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse existing JSON in {file_path}: {exc}") from exc

    if not isinstance(data, dict):
        data = {}

    if "mcpServers" not in data or not isinstance(data["mcpServers"], dict):
        data["mcpServers"] = {}

    existing_config = data["mcpServers"].get(server_name)
    if existing_config == server_config:
        # Idempotent: already matches exactly
        return False, data

    data["mcpServers"][server_name] = server_config
    formatted_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    file_path.write_text(formatted_json, encoding="utf-8")
    return True, data


def run_configuration(
    api_key: Optional[str] = None,
    api_url: str = "https://citadel-7j9u.onrender.com",
    project_dir: Optional[Path] = None,
    desktop_config_path: Optional[Path] = None,
    code_config_path: Optional[Path] = None,
    command_path: Optional[str] = None,
    skip_desktop: bool = False,
    skip_code: bool = False,
    system_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Orchestrates the resolution and configuration of Claude Desktop and Claude Code.
    """
    executable = find_mcp_executable(command_path)
    server_config = build_server_config(
        command_path=executable,
        api_key=api_key,
        api_url=api_url,
    )

    results: dict[str, Any] = {
        "executable": executable,
        "server_config": server_config,
        "targets": {},
    }

    if not skip_desktop:
        target_desktop = desktop_config_path or get_claude_desktop_config_path(system_name)
        modified, data = update_config_file(target_desktop, server_config)
        results["targets"]["claude_desktop"] = {
            "path": str(target_desktop),
            "modified": modified,
        }

    if not skip_code:
        target_code = code_config_path or get_claude_code_settings_path(project_dir)
        modified, data = update_config_file(target_code, server_config)
        results["targets"]["claude_code"] = {
            "path": str(target_code),
            "modified": modified,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure Citadel Predict MCP server for Claude Desktop and Claude Code."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CITADEL_API_KEY"),
        help="Citadel Predict API Key (optional for local unauthenticated backend)",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("CITADEL_API_URL", "https://citadel-7j9u.onrender.com"),
        help="Citadel Predict API URL (default: https://citadel-7j9u.onrender.com)",
    )
    parser.add_argument(
        "--command-path",
        help="Explicit path to citadel-predict-mcp executable (auto-detected if omitted)",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        help="Citadel repository root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--desktop-config",
        type=Path,
        help="Override Claude Desktop config file destination",
    )
    parser.add_argument(
        "--code-config",
        type=Path,
        help="Override Claude Code settings.json destination",
    )
    parser.add_argument(
        "--skip-desktop",
        action="store_true",
        help="Skip configuring Claude Desktop",
    )
    parser.add_argument(
        "--skip-code",
        action="store_true",
        help="Skip configuring Claude Code",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without interactive prompts",
    )

    args = parser.parse_args()

    print("==========================================================")
    print(" Citadel Predict MCP Server Configuration")
    print("==========================================================\n")

    api_key = args.api_key
    api_url = args.api_url

    # Interactive prompt if running interactively and not all parameters passed via flags
    if not args.non_interactive and sys.stdin.isatty():
        try:
            print("Configure your Citadel Predict connection parameters:")
            key_prompt = f"Enter CITADEL_API_KEY [{api_key or 'None / Local'}]: "
            user_key = input(key_prompt).strip()
            if user_key:
                api_key = user_key

            url_prompt = f"Enter CITADEL_API_URL [{api_url}]: "
            user_url = input(url_prompt).strip()
            if user_url:
                api_url = user_url
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nConfiguration cancelled.")
            sys.exit(1)

    try:
        results = run_configuration(
            api_key=api_key,
            api_url=api_url,
            project_dir=args.project_dir,
            desktop_config_path=args.desktop_config,
            code_config_path=args.code_config,
            command_path=args.command_path,
            skip_desktop=args.skip_desktop,
            skip_code=args.skip_code,
        )

        print(f"Platform:   {platform.system()} ({platform.machine()})")
        print(f"Executable: {results['executable']}\n")
        print("Generated Server Config:")
        print(json.dumps(results["server_config"], indent=2))
        print("\nConfigured Target Files:")

        for target_name, info in results["targets"].items():
            status = "Updated" if info["modified"] else "Verified (Up to date)"
            print(f"  - [{target_name}] -> {info['path']} ({status})")

        print("\nConfiguration completed successfully.")
        print("Restart Claude Desktop or restart your Claude Code session to apply changes.")

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
