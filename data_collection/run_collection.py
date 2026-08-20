"""
Main data-collection driver: runs each task in tasks.py through the agent
runner multiple times (agent runs are non-deterministic, so we need several
samples per task to see real variance, not just one lucky/unlucky run) and
appends structured results to a JSONL file.

Usage:
    python run_collection.py                       # all 20 tasks x 4 runs
    python run_collection.py --runs-per-task 2      # quick, cheaper pilot
    python run_collection.py --task-ids t01 t07     # just specific tasks
    python run_collection.py --output data/pilot.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

from agent_runner import run_task
from tasks import TASKS

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "runs.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Collect agent cost data.")
    parser.add_argument("--runs-per-task", type=int, default=4,
                         help="How many times to run each task (default 4).")
    parser.add_argument("--task-ids", nargs="*", default=None,
                         help="Only run these task IDs (default: all tasks).")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                         help="Path to append JSONL results to.")
    args = parser.parse_args()

    tasks = TASKS
    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [t for t in TASKS if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            print(f"Unknown task ids: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    total_runs = len(tasks) * args.runs_per_task
    print(f"Collecting {total_runs} runs ({len(tasks)} tasks x {args.runs_per_task} "
          f"repeats) -> {output_path}")

    completed = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for task in tasks:
            for run_index in range(args.runs_per_task):
                try:
                    result = run_task(client, task)
                except Exception as e:
                    print(f"  [{task['id']} run {run_index}] FAILED: {e}", file=sys.stderr)
                    continue
                result["run_index"] = run_index
                f.write(json.dumps(result) + "\n")
                f.flush()
                completed += 1
                print(
                    f"  [{completed}/{total_runs}] {task['id']} run {run_index}: "
                    f"{result['num_turns']} turns, "
                    f"{result['total_tokens']} tokens "
                    f"(in={result['total_input_tokens']}, out={result['total_output_tokens']}), "
                    f"{result['total_tool_calls']} tool calls, "
                    f"{result['wall_time_seconds']}s"
                )

    print(f"Done. {completed}/{total_runs} runs written to {output_path}")


if __name__ == "__main__":
    main()
