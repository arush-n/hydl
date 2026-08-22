"""The Java side: what would ship, and whether it passes over there.

A policy is trained in JAX and executed by a Java port inside HytaleServer.
Nothing about that is visible from a rollout, so this panel answers the two
questions that decide whether a checkpoint is shippable at all:

* **Does the bundle's contract match the runtime?** A bundle declares its
  observation width, action count and layer widths; `AgentBundle.problems()`
  refuses to arm a runtime that disagrees. A mismatch here is why an agent
  loads and then stands still.
* **Do the Java tests pass?** The port is only worth shipping if it reproduces
  JAX. `RecurrentTest` checks 900 steps of parity, `ActionTest` checks the
  decode, `BundleTest` checks the contract gate.

**Pushing to a live server is deliberately not a button here.** Deploying
writes into a running game server's mod directory, which is outward and not
trivially reversible; the exact command is printed instead so it is run
knowingly. Whether the deployed agent is actually moving is a separate
question, answered by the live-server card on Status -- clean ticks and zero
illegal actions look identical to an NPC standing still, which is a mistake
this project has already made once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from console.core.execution import checks
from console.core.panels import panel

_ROOT = Path(__file__).resolve().parents[2]
JVM_AGENT = _ROOT / "experimental" / "jvm-agent"


@panel("java-port", title="Java port", tab="train", order=20, agent="built-in",
       refresh=15.0,
       description="What would ship to HytaleServer, and whether it passes there.")
def java_port() -> dict[str, Any]:
    facts: list[dict[str, Any]] = []

    # `build/`, not `build/libs/`. This module is not built by Gradle -- it is
    # plain `javac` into `out/` plus a hand-rolled jar step -- so the Gradle
    # convention path is empty and always was. Globbing it reported "not built"
    # while two perfectly good jars sat one directory up, which sent a handoff
    # out telling another lane to build something that already existed.
    jar = sorted(JVM_AGENT.glob("build/*.jar"), key=lambda p: p.stat().st_mtime)
    facts.append({
        "label": "mod jar",
        "value": jar[-1].name if jar else "not built",
        "tone": "pos" if jar else "absent",
    })

    case = JVM_AGENT / "case"
    facts.append({
        "label": "parity fixture",
        "value": "present" if case.is_dir() else "missing",
        "tone": "pos" if case.is_dir() else "neg",
    })

    result = (checks.index()["checks"])
    java = next((c for c in result if c["id"] == "jvm-agent"), None)
    last = (java or {}).get("last")
    if last:
        facts.append({"label": "java tests", "value": last["state"],
                      "tone": "pos" if last["state"] == "passed" else "neg"})
        facts.append({"label": "passed", "value": last["passed"],
                      "tone": "pos" if last["passed"] else "absent"})
        facts.append({"label": "failed", "value": last["failed"],
                      "tone": "neg" if last["failed"] else "pos"})
    else:
        facts.append({"label": "java tests", "value": "never run",
                      "tone": "absent"})

    note = (
        "Run the Java suite from the Tests card; its counts come from JUnit "
        "XML, never the Gradle exit code — BUILD SUCCESSFUL with exit 0 is "
        "also what zero tests executed looks like. "
        "To deploy, export a bundle with `python -m adk.deploy.bundle` and "
        "copy it into the server's `mods/policy-agent/`. That is left as a "
        "command rather than a button because it writes into a running game "
        "server."
    )
    if last and last.get("note"):
        note = f"{last['note']} {note}"

    return {"view": "facts", "facts": facts, "note": note,
            "count": (last or {}).get("state", "")}
