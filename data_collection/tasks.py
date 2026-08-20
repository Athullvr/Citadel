"""
The 20 representative test tasks used for data collection.

Each task varies along the axes we expect to predict cost from:
  - tool_count: how many tools are available to the agent (0-7)
  - category: a rough hand-labeled complexity bucket, used only for our own
    stratification/analysis -- NOT fed to the model, and not a feature we
    claim to derive automatically (Phase 2 derives features from the raw
    task text + tool list only, the way a real user's input would look).

Categories:
  - single_shot:        one clear, bounded action; little/no iteration
  - narrow_multi_step:  a handful of well-specified sequential steps
  - open_ended:         the task requires the agent to plan its own approach,
                         and/or explicitly implies repetition ("research",
                         "until", "keep trying", "investigate")
"""

TASKS = [
    {
        "id": "t01",
        "text": "Summarize the following document in 2-3 sentences: "
                "'The quarterly report shows a 4% increase in revenue, driven primarily "
                "by the new product line launched in March. Customer churn decreased "
                "slightly. Operating costs rose due to increased hiring in engineering.'",
        "tools": [],
        "category": "single_shot",
    },
    {
        "id": "t02",
        "text": "Read the document at 'reports/q3_summary.txt' and list its 3 key takeaways.",
        "tools": ["read_document"],
        "category": "single_shot",
    },
    {
        "id": "t03",
        "text": "Calculate the total cost of 3 units at $42.50 each, plus a flat $15 shipping fee, "
                "and tell me the final amount.",
        "tools": ["calculator"],
        "category": "single_shot",
    },
    {
        "id": "t04",
        "text": "Research the topic 'renewable energy storage trends' across 3 sources and write "
                "a short summary of what you find.",
        "tools": ["web_search", "fetch_url"],
        "category": "open_ended",
    },
    {
        "id": "t05",
        "text": "Research this topic across 5 sources, synthesize into a report, then draft 3 "
                "follow-up emails to relevant stakeholders based on the findings. Topic: "
                "'the impact of remote work on commercial real estate demand'.",
        "tools": ["web_search", "fetch_url", "draft_document", "send_email"],
        "category": "open_ended",
    },
    {
        "id": "t06",
        "text": "Find all files related to the budget in the 'finance/' directory and summarize "
                "what each one contains.",
        "tools": ["list_files", "read_document"],
        "category": "narrow_multi_step",
    },
    {
        "id": "t07",
        "text": "Look up the current population of France.",
        "tools": ["web_search"],
        "category": "single_shot",
    },
    {
        "id": "t08",
        "text": "Keep searching until you find a reliable, authoritative source confirming the "
                "boiling point of water at high altitude (e.g. 3000m). Don't stop at the first result.",
        "tools": ["web_search", "fetch_url"],
        "category": "open_ended",
    },
    {
        "id": "t09",
        "text": "Draft a short email thanking a colleague named Priya for her help on the last project.",
        "tools": ["send_email"],
        "category": "single_shot",
    },
    {
        "id": "t10",
        "text": "Draft and send 3 different emails: one to the design team about the new mockups, "
                "one to the eng team about the deployment schedule, and one to the exec team with "
                "a status update.",
        "tools": ["send_email"],
        "category": "narrow_multi_step",
    },
    {
        "id": "t11",
        "text": "Read the document at 'data/expenses.txt' and calculate the sum of all the dollar "
                "amounts mentioned in it.",
        "tools": ["read_document", "calculator"],
        "category": "narrow_multi_step",
    },
    {
        "id": "t12",
        "text": "Perform a competitive analysis: research 5 competitors in the project-management "
                "software space, synthesize the findings into a report, and email a summary to the team.",
        "tools": ["web_search", "fetch_url", "draft_document", "send_email"],
        "category": "open_ended",
    },
    {
        "id": "t13",
        "text": "What is 15% of 240?",
        "tools": ["calculator"],
        "category": "single_shot",
    },
    {
        "id": "t14",
        "text": "List the files in the project directory.",
        "tools": ["list_files"],
        "category": "single_shot",
    },
    {
        "id": "t15",
        "text": "Investigate a customer complaint about slow shipping by searching our support "
                "documentation, then draft a response email addressing their concern.",
        "tools": ["web_search", "read_document", "send_email"],
        "category": "narrow_multi_step",
    },
    {
        "id": "t16",
        "text": "Write a report on 'AI adoption in healthcare'. Search for sources, draft an initial "
                "version, then search again for anything you feel is missing, and revise the "
                "report to fill those gaps before finalizing it.",
        "tools": ["web_search", "fetch_url", "draft_document"],
        "category": "open_ended",
    },
    {
        "id": "t17",
        "text": "What is the capital of Australia?",
        "tools": [],
        "category": "single_shot",
    },
    {
        "id": "t18",
        "text": "Research the topic 'electric vehicle battery costs', fetch details from at least 2 "
                "specific sources, calculate the average cost per kWh you find across those sources, "
                "draft a short document summarizing the analysis, and email it to 'team@example.com'.",
        "tools": ["web_search", "fetch_url", "calculator", "draft_document", "send_email"],
        "category": "open_ended",
    },
    {
        "id": "t19",
        "text": "Help plan a small team offsite: search for venue options in the area, calculate a "
                "rough total budget assuming 12 attendees and $85/person for the venue, and draft an "
                "announcement email to the team.",
        "tools": ["web_search", "calculator", "send_email"],
        "category": "narrow_multi_step",
    },
    {
        "id": "t20",
        "text": "Look into whether our product's pricing is competitive in the market.",
        "tools": ["web_search", "fetch_url", "draft_document"],
        "category": "open_ended",
    },
]
