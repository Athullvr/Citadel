"""
A minimal ReAct-style agent runner built directly on the Anthropic Messages API
(manual loop, not the SDK tool runner) so we get full per-turn token-usage
visibility for the cost-prediction dataset.

Each run:
  - sends the task as the first user message
  - loops: call the model -> if it asks for tools, execute them and feed
    results back -> repeat until stop_reason == "end_turn" or a turn cap
    is hit
  - records per-turn token usage and returns a structured result dict

Model: claude-sonnet-5 (explicitly requested by the project spec).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anthropic

from tools import execute_tool, get_tools_for_names

MODEL = "claude-sonnet-5"
MAX_TOKENS_PER_TURN = 4096
MAX_TURNS = 15  # hard cap to bound runaway cost/loops in data collection

SYSTEM_PROMPT = (
    "You are an autonomous agent completing a task on behalf of a user. "
    "You have access to a set of tools (if any are listed below); use them "
    "as needed to complete the task. When you are done, give a final answer "
    "summarizing what you did and the result. Do not ask the user clarifying "
    "questions -- make reasonable assumptions and proceed."
)
logger = logging.getLogger("citadel.runner")


def run_task(client: anthropic.Anthropic, task: dict, telemetry: Any | None = None) -> dict:
    """Run one task through the agent loop once. Returns a result dict with
    full per-turn usage plus totals."""
    tool_names = task["tools"]
    tools = get_tools_for_names(tool_names) if tool_names else None

    messages = [{"role": "user", "content": task["text"]}]
    turns = []
    stop_reason = None
    hit_max_turns = False
    telemetry_execution = None
    if telemetry is not None:
        try:
            telemetry_execution = telemetry.start_execution(task, MODEL)
        except Exception:  # noqa: BLE001 - telemetry must not stop the runner
            logger.warning("Could not initialize Citadel telemetry; the agent run will continue.")
    start = time.monotonic()

    try:
        for turn_index in range(MAX_TURNS):
            kwargs = dict(
                model=MODEL,
                max_tokens=MAX_TOKENS_PER_TURN,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            if tools:
                kwargs["tools"] = tools

            response = client.messages.create(**kwargs)

            usage = response.usage
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            turns.append({
                "turn_index": turn_index,
                "stop_reason": response.stop_reason,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "num_tool_calls": len(tool_use_blocks),
                "tool_names_called": [b.name for b in tool_use_blocks],
            })
            if telemetry_execution is not None:
                try:
                    telemetry.record_response(telemetry_execution, response)
                except Exception:  # noqa: BLE001 - usage reporting is best-effort
                    logger.warning("Could not report one usage event; the agent run will continue.")

            stop_reason = response.stop_reason
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in tool_use_blocks:
                result_text = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            hit_max_turns = True
    except Exception:
        if telemetry_execution is not None:
            try:
                telemetry.complete_execution(telemetry_execution, succeeded=False)
            except Exception:  # noqa: BLE001 - completion reporting is best-effort
                logger.warning("Could not report execution failure to Citadel telemetry.")
        raise

    elapsed = time.monotonic() - start

    total_input = sum(t["input_tokens"] for t in turns)
    total_output = sum(t["output_tokens"] for t in turns)
    total_cache_creation_known = sum(
        t["cache_creation_input_tokens"]
        for t in turns
        if t["cache_creation_input_tokens"] is not None
    )
    total_cache_read_known = sum(
        t["cache_read_input_tokens"] for t in turns if t["cache_read_input_tokens"] is not None
    )
    cache_fields_complete = all(
        t["cache_creation_input_tokens"] is not None and t["cache_read_input_tokens"] is not None
        for t in turns
    )
    total_cache_creation = total_cache_creation_known if all(
        t["cache_creation_input_tokens"] is not None for t in turns
    ) else None
    total_cache_read = total_cache_read_known if all(
        t["cache_read_input_tokens"] is not None for t in turns
    ) else None
    total_tool_calls = sum(t["num_tool_calls"] for t in turns)

    total_processed_input = (
        total_input + total_cache_creation_known + total_cache_read_known
        if cache_fields_complete
        else None
    )
    result = {
        "task_id": task["id"],
        "task_text": task["text"],
        "tools_available": tool_names,
        "category": task["category"],
        "model": MODEL,
        "num_turns": len(turns),
        "hit_max_turns": hit_max_turns,
        "final_stop_reason": stop_reason,
        "total_tool_calls": total_tool_calls,
        # Keep the historical dataset fields stable: Anthropic's input_tokens
        # excludes cache counters, and existing runs used this raw field.
        "total_input_tokens": total_input,
        "total_processed_input_tokens": total_processed_input,
        "total_output_tokens": total_output,
        "total_cache_creation_input_tokens": total_cache_creation,
        "total_cache_read_input_tokens": total_cache_read,
        "total_tokens": total_input + total_output,
        "wall_time_seconds": round(elapsed, 2),
        "turns": turns,
    }
    if telemetry_execution is not None:
        try:
            telemetry.complete_execution(telemetry_execution, succeeded=not hit_max_turns)
        except Exception:  # noqa: BLE001 - completion reporting is best-effort
            logger.warning(
                "Could not complete Citadel telemetry; the agent run result is unchanged."
            )
    return result
