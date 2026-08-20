"""
Feature extraction for the cost predictor.

Deliberately derives every feature from ONLY the raw task text + the list of
tool names available -- the same inputs a real user would give the product
before the agent runs. The hand-labeled `category` field in tasks.py is used
elsewhere for our own analysis/stratification only and is NOT a feature here.
"""

import re

# Words that signal the agent will likely have to plan its own approach
# and/or iterate/retry rather than complete the task in one deterministic pass.
OPEN_ENDED_KEYWORDS = [
    "research", "investigate", "explore", "look into", "keep searching",
    "until", "synthesize", "analysis", "analyze", "revise", "iterate",
    "don't stop", "reliable", "authoritative", "competitive analysis",
    "fill those gaps", "missing", "look up", "figure out",
]

# Words that signal a single, bounded, well-specified action.
NARROW_KEYWORDS = [
    "summarize the following", "calculate", "what is", "list the files",
    "the following document", "read the document at",
]

# Matches things like "3 sources", "5 competitors", "3 different emails" --
# explicit counts imply a known number of repeated sub-actions.
COUNT_PATTERN = re.compile(r"\b(\d+)\s+(sources?|competitors?|emails?|files?|"
                            r"units?|attendees?)\b", re.IGNORECASE)

# Sequencing words that imply multiple sequential steps within one task.
STEP_CONNECTORS = ["then", "after that", "once you", "next,", "followed by"]


def extract_features(task_text: str, tool_names: list[str]) -> dict:
    """Extract a fixed-size numeric feature dict from raw task text + tool list.

    This is the exact function the web UI (Phase 3) will call on user input,
    so it must not depend on anything not available before the agent runs.
    """
    text = task_text.strip()
    text_lower = text.lower()
    words = text.split()

    num_tools = len(tool_names or [])

    open_ended_hits = sum(1 for kw in OPEN_ENDED_KEYWORDS if kw in text_lower)
    narrow_hits = sum(1 for kw in NARROW_KEYWORDS if kw in text_lower)

    count_matches = COUNT_PATTERN.findall(text_lower)
    explicit_counts = [int(m[0]) for m in count_matches]
    max_explicit_count = max(explicit_counts) if explicit_counts else 0
    sum_explicit_counts = sum(explicit_counts)

    step_connector_hits = sum(1 for kw in STEP_CONNECTORS if kw in text_lower)

    # Rough count of imperative "action" verbs at the start of clauses,
    # split on common separators -- a crude proxy for how many distinct
    # sub-tasks are being asked for in one request.
    clauses = re.split(r",|;|\bthen\b|\band then\b", text_lower)
    num_clauses = len([c for c in clauses if c.strip()])

    is_question = text.strip().endswith("?")

    return {
        "text_char_len": len(text),
        "text_word_len": len(words),
        "num_tools": num_tools,
        "open_ended_keyword_hits": open_ended_hits,
        "narrow_keyword_hits": narrow_hits,
        "max_explicit_count": max_explicit_count,
        "sum_explicit_counts": sum_explicit_counts,
        "step_connector_hits": step_connector_hits,
        "num_clauses": num_clauses,
        "is_question": int(is_question),
    }


FEATURE_NAMES = [
    "text_char_len", "text_word_len", "num_tools", "open_ended_keyword_hits",
    "narrow_keyword_hits", "max_explicit_count", "sum_explicit_counts",
    "step_connector_hits", "num_clauses", "is_question",
]
