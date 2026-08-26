from pathlib import Path
from unittest.mock import patch

from citadel_predict.config import (
    DEFAULT_API_URL,
    load_config_file,
    resolve_api_key,
    resolve_api_url,
)


def test_load_config_file_nonexistent(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.toml"
    assert load_config_file(non_existent) == {}


def test_load_config_file_flat(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        'api_key = "test-flat-key"\napi_url = "https://api.citadel.example.com"\n',
        encoding="utf-8",
    )
    data = load_config_file(config_file)
    assert data["api_key"] == "test-flat-key"
    assert data["api_url"] == "https://api.citadel.example.com"


def test_load_config_file_with_section(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[default]\napi_key = "test-section-key"\napi_url = "https://custom.citadel.internal"\n',
        encoding="utf-8",
    )
    data = load_config_file(config_file)
    assert data["api_key"] == "test-section-key"
    assert data["api_url"] == "https://custom.citadel.internal"


def test_resolve_api_key_priority(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text('api_key = "file-key"\n', encoding="utf-8")

    # 1. No key anywhere -> None
    monkeypatch.delenv("CITADEL_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    assert resolve_api_key(config_path=tmp_path / "empty.toml") is None

    # 2. Config file provides key
    assert resolve_api_key(config_path=config_file) == "file-key"

    # 3. Env var overrides config file
    monkeypatch.setenv("CITADEL_API_KEY", "env-key")
    assert resolve_api_key(config_path=config_file) == "env-key"

    # 4. Explicit argument overrides env var
    assert resolve_api_key("explicit-key", config_path=config_file) == "explicit-key"


def test_resolve_api_url_priority(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text('api_url = "https://file-url.com"\n', encoding="utf-8")

    # 1. Default URL when nothing is set
    monkeypatch.delenv("CITADEL_API_URL", raising=False)
    assert resolve_api_url(config_path=tmp_path / "empty.toml") == DEFAULT_API_URL

    # 2. Config file provides URL
    assert resolve_api_url(config_path=config_file) == "https://file-url.com"

    # 3. Env var overrides config file
    monkeypatch.setenv("CITADEL_API_URL", "https://env-url.com/")
    assert resolve_api_url(config_path=config_file) == "https://env-url.com"

    # 4. Explicit argument overrides env var
    assert resolve_api_url("https://explicit-url.com/", config_path=config_file) == "https://explicit-url.com"
