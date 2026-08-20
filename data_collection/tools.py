"""
Mock tools for the data-collection agent runner.

These are deliberately synthetic (no real network calls) so that:
  - data collection has zero external cost/latency/flakiness beyond the LLM calls
  - tool result sizes are controllable, so we can vary context-accumulation
    pressure across tasks in a reproducible way
  - runs are repeatable (same tool inputs -> same shape of outputs, though we
    inject some randomness to mimic real-world variance)

Each tool function takes a dict of parsed input and returns a string (the
content that goes back to the model as a tool_result).
"""

import random


def _lorem(num_sentences: int) -> str:
    fragments = [
        "The report highlights a moderate increase in quarterly activity.",
        "Several stakeholders raised concerns about data consistency.",
        "Historical trends suggest a seasonal pattern worth investigating.",
        "The methodology section describes a mixed-methods approach.",
        "Analysts disagree on the primary driver of the observed change.",
        "A follow-up study is recommended to confirm these findings.",
        "The dataset excludes entries flagged as low-confidence.",
        "Regional variation appears larger than the national average.",
        "The author notes several limitations in the sampling method.",
        "Comparable figures from last year were not publicly available.",
        "Preliminary results were presented at an internal review.",
        "The summary table aggregates figures from three separate sources.",
        "No statistically significant effect was found in the subgroup analysis.",
        "The vendor's documentation was incomplete for this integration.",
        "Cost estimates vary widely depending on deployment scale.",
    ]
    return " ".join(random.choice(fragments) for _ in range(num_sentences))


def web_search(input: dict) -> str:
    """Simulate a web search: returns 3-5 fake results with title/url/snippet."""
    query = input.get("query", "")
    n = random.randint(3, 5)
    results = []
    for i in range(n):
        results.append(
            f"{i + 1}. \"{query.title()} — Source {i + 1}\"\n"
            f"   url: https://example.com/articles/{abs(hash(query)) % 10000}-{i}\n"
            f"   snippet: {_lorem(random.randint(2, 4))}"
        )
    return "Search results for '{}':\n\n{}".format(query, "\n\n".join(results))


def fetch_url(input: dict) -> str:
    """Simulate fetching a URL: returns a fake page body of moderate length."""
    url = input.get("url", "unknown-url")
    body = _lorem(random.randint(8, 16))
    return f"Fetched content from {url}:\n\n{body}"


def read_document(input: dict) -> str:
    """Simulate reading a local document (longer content, mimics a real file read)."""
    path = input.get("path", "document.txt")
    body = _lorem(random.randint(15, 30))
    return f"Contents of {path}:\n\n{body}"


def list_files(input: dict) -> str:
    """Simulate listing files in a directory."""
    directory = input.get("directory", ".")
    fake_files = [
        "notes.md", "draft_v1.docx", "data.csv", "summary.txt",
        "meeting_minutes.md", "budget.xlsx", "archive/old_report.pdf",
    ]
    n = random.randint(2, len(fake_files))
    chosen = random.sample(fake_files, n)
    return f"Files in {directory}:\n" + "\n".join(f"  {f}" for f in chosen)


def calculator(input: dict) -> str:
    """A real (deterministic) calculator tool — evaluates a simple arithmetic expression."""
    expr = input.get("expression", "")
    try:
        # Restrict eval to arithmetic only.
        allowed = set("0123456789+-*/(). ")
        if not set(expr) <= allowed:
            return "Error: expression contains disallowed characters."
        result = eval(expr, {"__builtins__": {}}, {})
        return f"{expr} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def send_email(input: dict) -> str:
    """Simulate sending an email (no-op, just confirms)."""
    to = input.get("to", "unknown@example.com")
    subject = input.get("subject", "(no subject)")
    return f"Email sent to {to} with subject '{subject}'. (This is a simulated send; no real email was sent.)"


def draft_document(input: dict) -> str:
    """Simulate saving a drafted document — confirms word count."""
    title = input.get("title", "Untitled")
    content = input.get("content", "")
    word_count = len(content.split())
    return f"Draft '{title}' saved ({word_count} words)."


TOOL_IMPLEMENTATIONS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "read_document": read_document,
    "list_files": list_files,
    "calculator": calculator,
    "send_email": send_email,
    "draft_document": draft_document,
}


TOOL_SCHEMAS = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information on a topic. Returns a short list of results with titles, URLs, and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
    "fetch_url": {
        "name": "fetch_url",
        "description": "Fetch the text content of a specific URL (e.g. one returned by web_search).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
            },
            "required": ["url"],
        },
    },
    "read_document": {
        "name": "read_document",
        "description": "Read the contents of a local document/file by path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the document."},
            },
            "required": ["path"],
        },
    },
    "list_files": {
        "name": "list_files",
        "description": "List files in a local directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to list. Defaults to current directory."},
            },
        },
    },
    "calculator": {
        "name": "calculator",
        "description": "Evaluate a simple arithmetic expression (numbers and + - * / ( ) only).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression, e.g. '2 * (3 + 4)'."},
            },
            "required": ["expression"],
        },
    },
    "send_email": {
        "name": "send_email",
        "description": "Send an email to a recipient with a subject and body.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
            },
            "required": ["to", "subject", "body"],
        },
    },
    "draft_document": {
        "name": "draft_document",
        "description": "Save a drafted document (e.g. a report or article) with a title and content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title."},
                "content": {"type": "string", "description": "Full document content."},
            },
            "required": ["title", "content"],
        },
    },
}


def get_tools_for_names(names: list[str]) -> list[dict]:
    """Return the Anthropic tool-definition list for a list of tool names."""
    return [TOOL_SCHEMAS[name] for name in names]


def execute_tool(name: str, input: dict) -> str:
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return f"Error: unknown tool '{name}'."
    return impl(input)
