import os
import sys
from pathlib import Path

import pytest

# Ensure data_collection and backend are importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
DATA_COLLECTION_DIR = ROOT_DIR / "data_collection"

sys.path.insert(0, str(DATA_COLLECTION_DIR))
sys.path.insert(0, str(BACKEND_DIR))

os.environ["DATA_COLLECTION_DIR"] = str(DATA_COLLECTION_DIR)


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    """Ensure clean env vars for each test by default."""
    monkeypatch.delenv("CITADEL_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
