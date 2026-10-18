"""Parity adapter: today's one goal per lane as one request, run through the combiner path (A6).

``docs/AGENT_BRIEF.md`` A6 and the last gate of ``experimental/neural_agent/docs/SYSTEM_ONE.md`` section 9.10: today's
behaviour is the one-request special case. ``ParityActor`` wraps the dispatcher (an ADK ``LowLevelActor``) and, inside
the compiled step, expresses each lane's goal as one request in the active-request table:

    wants[b, s]   = scope s is in the scopes of the lane's skill (``requests.ARGUMENT_SCHEMAS``)
    legal[b, s]   = every present head of scope s has a legal choice under the lane's action mask
                    (a scope without a present head is not legal)
    granted[b, s] = wants[b, s] and all over s' (not wants[b, s'] or legal[b, s'])
    factors[b, h] = dispatcher factor[b, h] where granted[b, scope(h)], else the neutral choice of h

The body is the dispatcher, which runs the lane's proof-of-concept skill; the merge keeps only the
heads of the request's scopes. Identical behaviour therefore also checks the scope assignment: a
skill that moved a head outside its declared scopes would change the action.

CLI (the gate): run the four-arm composition smoke twice, without and with the adapter, and compare
every non-timing column of ``metrics.csv``::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.parity \\
        --arms flat,j0,shuffled,j1 --ticks 64 --batch 8 --episodes 1 --seed 0 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent import requests as protocol
from experimental.neural_agent.skill_agent.contract.surface import (
    SCOPE_NAMES,
    head_layout,
    neutral_factors,
    scope_of,
)

TIMING_COLUMNS: frozenset[str] = frozenset(
    {"wall_seconds", "warmup_seconds", "jev_latency_ms_p50", "jev_latency_ms_p95"}
)
BRIEF_COLUMNS: tuple[str, ...] = (
    "catches",
    "damage_events",
    "goal_switches_per_lane_run",
    "interrupt_onsets",
    "interrupt_answered",
    "interrupt_unanswered",
    "commitment_violations",
)


class ParityActor:
    """The dispatcher behind the one-request combiner path, for the compiled hierarchy step.

    Args:
        actor: The wrapped ADK ``LowLevelActor`` (today's ``SkillDispatcher``).
        skill_ids: The registry's skill ids in catalogue order (``ctx.skill_ids``).
        schemas: Skill id -> ``requests.ArgumentSchema``; its scopes are each skill's request scopes.
        layout: The head layout of the actor's factors (today's ``"gen5"``).
    """

    def __init__(self, actor: Any, skill_ids: Sequence[str], *,
                 schemas: Mapping[str, protocol.ArgumentSchema] = protocol.ARGUMENT_SCHEMAS,
                 layout: str = "gen5") -> None:
        self.actor = actor
        heads = head_layout(layout)
        wants = np.zeros((len(skill_ids), len(SCOPE_NAMES)), dtype=bool)
        for k, skill_id in enumerate(skill_ids):
            for scope in schemas[skill_id].scopes:
                wants[k, SCOPE_NAMES.index(scope)] = True
        self._wants = jnp.asarray(wants)
        self._head_scope = jnp.asarray([SCOPE_NAMES.index(scope_of(head.name)) for head in heads], dtype=jnp.int32)
        self._neutral = jnp.asarray(neutral_factors(layout), dtype=jnp.int32)
        offsets = np.cumsum([0, *(head.size for head in heads)])
        self._slices = tuple((int(offsets[h]), int(offsets[h + 1])) for h in range(len(heads)))
        self._scope_heads = tuple(
            tuple(h for h, head in enumerate(heads) if head.scope == scope) for scope in SCOPE_NAMES
        )

    def initial_state(self, batch: int) -> Any:
        """The wrapped actor's reset state; the request path keeps no state of its own."""

        return self.actor.initial_state(batch)

    def scope_legality(self, action_mask: jax.Array) -> jax.Array:
        """bool[B, S]: every present head of the scope has a legal choice; empty scopes are not legal."""

        head_any = jnp.stack([action_mask[:, low:high].any(axis=1) for low, high in self._slices], axis=1)
        columns = [
            head_any[:, list(indices)].all(axis=1) if indices else jnp.zeros(action_mask.shape[:1], dtype=bool)
            for indices in self._scope_heads
        ]
        return jnp.stack(columns, axis=1)

    def __call__(self, state: Any, belief: Any, goal_payload: Any, actor_input: Any, key: jax.Array, /) -> Any:
        """Run the dispatcher, then merge its factors through the lane's one granted request."""

        decision = self.actor(state, belief, goal_payload, actor_input, key)
        verb = jnp.clip(jnp.asarray(goal_payload.skill_index, dtype=jnp.int32), 0, self._wants.shape[0] - 1)
        wants = self._wants[verb]
        legal = self.scope_legality(jnp.asarray(actor_input.action_mask))
        valid = jnp.all(~wants | legal, axis=1)
        granted = wants & valid[:, None]
        heads = granted[:, self._head_scope]
        factors = jnp.where(heads, decision.action_factors, self._neutral[None, :]).astype(jnp.int32)
        diagnostics = dict(decision.diagnostics or {})
        diagnostics.update(request_valid=valid, request_scopes=granted)
        return decision._replace(action_factors=factors, diagnostics=diagnostics)


def _wrap(arm: Any) -> Any:
    base = arm.make_actor
    return arm._replace(make_actor=lambda ctx: ParityActor(base(ctx), ctx.skill_ids))


def run(argv: Sequence[str], *, adapter: bool) -> int:
    """Run ``eval.run_composition.main(argv)``, with every arm's actor wrapped when ``adapter``."""

    from experimental.neural_agent.eval import run_composition

    if not adapter:
        return run_composition.main(list(argv))
    original = run_composition.get_arm
    run_composition.get_arm = lambda name: _wrap(original(name))
    try:
        return run_composition.main(list(argv))
    finally:
        run_composition.get_arm = original


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    """Arm -> its metrics row."""

    with path.open(newline="") as stream:
        return {row["arm"]: row for row in csv.DictReader(stream)}


def compare(baseline: Mapping[str, Mapping[str, str]], adapted: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Every non-timing cell that differs, as ``"<arm>.<column>: <baseline> != <adapted>"``."""

    differences = []
    for arm in sorted(set(baseline) | set(adapted)):
        left, right = baseline.get(arm, {}), adapted.get(arm, {})
        for column in sorted((set(left) | set(right)) - TIMING_COLUMNS):
            if left.get(column) != right.get(column):
                differences.append(f"{arm}.{column}: {left.get(column)} != {right.get(column)}")
    return differences


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Composition smoke with and without the one-request adapter.")
    parser.add_argument("--out", type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    outputs = {}
    for name, adapter in (("baseline", False), ("adapter", True)):
        directory = args.out / name
        code = run([*rest, "--out", str(directory)], adapter=adapter)
        if code != 0:
            sys.stderr.write(f"the {name} run exited with {code}\n")
            return code
        outputs[name] = read_metrics(directory / "metrics.csv")
    differences = compare(outputs["baseline"], outputs["adapter"])
    lines = ["arm | " + " | ".join(BRIEF_COLUMNS) + " | identical"]
    for arm, row in outputs["baseline"].items():
        same = all(row.get(column) == outputs["adapter"][arm].get(column) for column in row if column not in TIMING_COLUMNS)
        lines.append(f"{arm} | " + " | ".join(f"{row.get(c)} / {outputs['adapter'][arm].get(c)}" for c in BRIEF_COLUMNS)
                     + f" | {'yes' if same else 'NO'}")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"non-timing differences: {len(differences)}\n" + "".join(f"  {d}\n" for d in differences))
    return 0 if not differences else 1


__all__ = ["BRIEF_COLUMNS", "ParityActor", "TIMING_COLUMNS", "compare", "read_metrics", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
