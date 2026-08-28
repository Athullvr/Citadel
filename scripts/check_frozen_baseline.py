#!/usr/bin/env python3
"""
scripts/check_frozen_baseline.py — Git pre-commit & CI safeguard.

Prevents accidental modification or re-serialization of the frozen v1
baseline artifacts:
  - data_collection/calibration_data/claude-sonnet.joblib
  - data_collection/data/runs.jsonl

See CLAUDE.md Invariant #1 for architectural background.
"""

import subprocess
import sys

FROZEN_FILES = {
    "data_collection/calibration_data/claude-sonnet.joblib",
    "data_collection/data/runs.jsonl",
}


def get_staged_files() -> list[str]:
    """Get list of staged files in git index."""
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in res.stdout.splitlines() if line.strip()]


def check_staged_baseline() -> bool:
    """Check whether any staged file matches a frozen baseline artifact."""
    staged = get_staged_files()
    violations = [f for f in staged if f in FROZEN_FILES]

    if violations:
        print("\n" + "=" * 78, file=sys.stderr)
        print(" [BLOCKED] CRITICAL INVARIANT VIOLATION (CLAUDE.md Invariant #1)", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print("Attempted to stage/commit modifications to frozen baseline artifacts:\n", file=sys.stderr)
        for v in violations:
            print(f"  [X] {v}", file=sys.stderr)
        print(
            "\nThese files represent the finalized Claude Sonnet baseline (N=20 tasks / 80 runs)\n"
            "and MUST NOT be modified, retrained, or re-serialized.\n\n"
            "To unstage these files and keep your existing working copy, run:\n"
            f"  git restore --staged {' '.join(violations)}\n",
            file=sys.stderr,
        )
        print("=" * 78 + "\n", file=sys.stderr)
        return False

    return True


def main() -> None:
    if not check_staged_baseline():
        sys.exit(1)
    print("[OK] Frozen baseline check passed (no protected artifacts staged).")
    sys.exit(0)


if __name__ == "__main__":
    main()
