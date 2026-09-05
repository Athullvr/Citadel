"""
Configuration resolver for Citadel Predict.

Resolution priority:
1. Explicit function / CLI parameter
2. Environment variables (`CITADEL_API_KEY`, `CITADEL_API_URL`)
3. Config file (`~/.citadel/config.toml`)
"""

import os
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


DEFAULT_API_URL = "https://citadel-7j9u.onrender.com"
DEFAULT_CONFIG_PATH = Path.home() / ".citadel" / "config.toml"


def load_config_file(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Load configuration from ~/.citadel/config.toml if present.
    Supports top-level keys or keys under a [default] section.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists() or not path.is_file():
        return {}

    try:
        if tomllib is not None:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        else:
            # Fallback simple line-by-line parser for key = "value" pairs
            data = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("["):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        data[k] = v

        # If data has a [default] table, merge it
        if "default" in data and isinstance(data["default"], dict):
            merged = dict(data["default"])
            for k, v in data.items():
                if k != "default":
                    merged[k] = v
            return merged
        return data
    except Exception:
        # If config file is corrupted or unreadable, ignore silently and return empty dict
        return {}


def resolve_api_key(
    explicit_key: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Resolve API key according to priority:
    1. Explicit argument
    2. CITADEL_API_KEY environment variable (or legacy API_KEY)
    3. api_key in ~/.citadel/config.toml
    """
    if explicit_key:
        return explicit_key.strip()

    env_key = os.environ.get("CITADEL_API_KEY") or os.environ.get("API_KEY")
    if env_key:
        return env_key.strip()

    file_config = load_config_file(config_path)
    file_key = file_config.get("api_key")
    if file_key and isinstance(file_key, str):
        return file_key.strip()

    return None


def resolve_api_url(
    explicit_url: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> str:
    """
    Resolve base API URL according to priority:
    1. Explicit argument
    2. CITADEL_API_URL environment variable
    3. api_url in ~/.citadel/config.toml
    4. Default: http://localhost:8000
    """
    if explicit_url:
        return explicit_url.rstrip("/")

    env_url = os.environ.get("CITADEL_API_URL")
    if env_url:
        return env_url.rstrip("/")

    file_config = load_config_file(config_path)
    file_url = file_config.get("api_url")
    if file_url and isinstance(file_url, str):
        return file_url.rstrip("/")

    return DEFAULT_API_URL
