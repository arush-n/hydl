"""Saved setups, grouped by the architecture that uses them.

A configuration that lives only in a launch form is lost when the tab closes,
which makes "the setup we used for the dagger sweep" something you remember or
lose. A profile is a JSON file under `agents/profiles/` naming an agent and its
knobs; drop one in and it shows up here.

`architecture` is free text on purpose. A profile for a Dreamer or a
transformer is worth writing down before the console can run one -- the record
of intent is the useful part, and checking it against a list would mean
maintaining that list forever.
"""

from __future__ import annotations

from typing import Any

from console.core.catalog import profiles
from console.core.panels import panel


@panel("agent-profiles", title="Profiles", tab="run", order=20,
       agent="built-in",
       description="Saved setups from agents/profiles/, grouped by architecture.")
def agent_profiles() -> dict[str, Any]:
    found = profiles.index()
    if not found:
        return {
            "view": "note",
            "text": (
                f"No profiles yet. Drop a JSON file in {profiles.PROFILE_DIR} "
                "with a `title`, an `architecture` and a `run` block, and it "
                "appears here — no restart."
            ),
        }

    rows = []
    for profile in found:
        if profile.get("error"):
            rows.append({"tone": "crit",
                         "cells": [profile["id"], "malformed", profile["error"]]})
            continue
        run = profile.get("suggested_run") or {}
        world = run.get("world", "open_flat")
        where = f"{world}{' ' + run['zone'] if run.get('zone') else ''}"
        role = profile.get("checkpoint_role")
        status = profile.get("selection_status")
        selection = (
            f"{role} / {status.replace('_', ' ')}" if role and status else "-"
        )
        rows.append({
            "tone": "crit" if status == "rejected" else "info",
            "cells": [
                profile.get("display_title") or profile["title"],
                profile["architecture"],
                selection,
                run.get("loadout") or "—",
                where,
                f"{run.get('ticks', '—')} ticks",
            ],
        })

    architectures = sorted({p.get("architecture", "unspecified") for p in found})
    return {
        "view": "rows",
        "columns": [
            "profile", "architecture", "selection", "loadout", "world", "length"
        ],
        "rows": rows,
        "count": f"{len(found)} · {len(architectures)} architecture(s)",
        "note": (
            f"Read from {profiles.PROFILE_DIR}. Architectures present: "
            + ", ".join(architectures)
            + ". A profile for an architecture this console cannot run yet is "
            "still valid — it records intent, not capability."
        ),
    }
