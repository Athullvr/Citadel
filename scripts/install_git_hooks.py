#!/usr/bin/env python3
"""
scripts/install_git_hooks.py — Native Git Hooks Installer for Citadel Predict.

Installs lightweight, zero-dependency git hooks into .git/hooks/:
1. pre-commit: Guards frozen baseline artifacts & runs fast backend tests.
2. pre-push: Runs full test suite across all monorepo packages.

Usage:
  python scripts/install_git_hooks.py
"""

import os
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_HOOKS_DIR = REPO_ROOT / ".git" / "hooks"

PRE_COMMIT_SCRIPT = """#!/bin/sh
# Citadel Predict pre-commit hook

# Resolve active or repository virtual environment Python
if [ -f "data_collection/.venv/Scripts/python.exe" ]; then
    PY="data_collection/.venv/Scripts/python.exe"
elif [ -f "data_collection/.venv/bin/python" ]; then
    PY="data_collection/.venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

echo "[HOOK] Checking frozen baseline artifacts..."
"$PY" scripts/check_frozen_baseline.py || exit 1

echo "[HOOK] Running fast backend unit tests..."
"$PY" -m pytest backend/tests -q || exit 1

echo "[HOOK] Pre-commit checks passed."
"""

PRE_PUSH_SCRIPT = """#!/bin/sh
# Citadel Predict pre-push hook

# Resolve active or repository virtual environment Python
if [ -f "data_collection/.venv/Scripts/python.exe" ]; then
    PY="data_collection/.venv/Scripts/python.exe"
elif [ -f "data_collection/.venv/bin/python" ]; then
    PY="data_collection/.venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

echo "[HOOK] Running full pre-push test suite..."
"$PY" scripts/check_frozen_baseline.py || exit 1
"$PY" -m pytest -v || exit 1

echo "[HOOK] Pre-push checks passed. Ready to push."
"""


def install_hooks() -> None:
    if not (REPO_ROOT / ".git").exists():
        print(f"Error: No .git directory found at {REPO_ROOT}", file=sys.stderr)
        sys.exit(1)

    GIT_HOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. pre-commit hook
    pre_commit_path = GIT_HOOKS_DIR / "pre-commit"
    pre_commit_path.write_text(PRE_COMMIT_SCRIPT, encoding="utf-8")
    try:
        pre_commit_path.chmod(pre_commit_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

    # 2. pre-push hook
    pre_push_path = GIT_HOOKS_DIR / "pre-push"
    pre_push_path.write_text(PRE_PUSH_SCRIPT, encoding="utf-8")
    try:
        pre_push_path.chmod(pre_push_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

    print("=================================================================")
    print(" Citadel Git Hooks Installed Successfully")
    print("=================================================================")
    print(f"  - Pre-commit: {pre_commit_path}")
    print(f"  - Pre-push:   {pre_push_path}")
    print("\nSafeguards Active:")
    print("  1. Guards frozen baseline (claude-sonnet.joblib & runs.jsonl)")
    print("  2. Runs fast backend tests on commit")
    print("  3. Runs full test suite on push")
    print("=================================================================")


if __name__ == "__main__":
    install_hooks()
