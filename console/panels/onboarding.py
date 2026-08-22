"""A first-run card framed as three questions the console can answer."""

from __future__ import annotations

from typing import Any

from console.core.panels import panel


@panel(
    "first-run",
    title="Start here",
    tab="status",
    order=1,
    description="Three verbs, each tied to the claim it can actually support.",
    agent="built-in",
    size="wide",
    tags=("first run",),
)
def first_run() -> dict[str, Any]:
    return {
        "view": "rows",
        "columns": ["Do", "What it proves"],
        "rows": [
            {
                "cells": [
                    {"value": "Run one rollout", "href": "#run", "mono": False},
                    {"value": "Does the agent act?", "mono": False, "wrap": True},
                ],
            },
            {
                "cells": [
                    {"value": "Compare two runs", "href": "#runs", "mono": False},
                    {"value": "Is the difference real?", "mono": False, "wrap": True},
                ],
            },
            {
                "cells": [
                    {"value": "Open Evidence", "href": "#evidence", "mono": False},
                    {
                        "value": "What do we currently claim?",
                        "mono": False,
                        "wrap": True,
                    },
                ],
            },
        ],
        "count": 3,
        "status": {"label": "first run", "tone": "info"},
        "summary": "Act, compare, then inspect the declared evidence contract.",
    }
