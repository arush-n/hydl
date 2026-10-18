"""Gen-5 overlay: the identity render of the combat simulator.

``OVERLAY``: the modules of ``hytalegym.jax.combat`` plus every ``hytalegym.jax``
module they import, relocated and nothing else: no redirects, no hooks, no
patches, so the per-lane step clock (``nominal_delta_seconds`` 1/30 s,
``loaded_delta_seconds`` 0.045 s, three timing profiles drawn per lane at reset)
is unchanged. Rendering it must reproduce today's gen-5 rollouts bit for bit;
that is the identity gate (``tests/sim/test_render_identity.py``) every renderer
change has to keep green.

``OPTIMIZED``: the identity render plus opt-in optimisation patches, each of
which must keep the identity gate green (bit-exact) before it is offered to a
generation. Not the default; ``render --generation gen5`` renders ``OVERLAY``.
"""

from __future__ import annotations

from dataclasses import replace

from hytalegym.sim.common.overlay import Overlay, Patch

OVERLAY = Overlay(
    generation="gen5",
    profile="hytalegym.sim.gen5.profile:PROFILE",
    modules=("combat/",),
    description="identity: hytalegym.jax.combat and its hytalegym.jax imports, relocated only",
)

_STATUSES = "combat/mechanics/runtime/statuses.py"
_IDENTITY = "the identity gate (tests/sim/test_render_identity.py), bit for bit"

OPTIMIZATIONS = (
    Patch(
        "resistance_walk_bounded_by_cause_depth", _STATUSES,
        "def _resolve_damage_resistance_amount(\n",
        '''def _cause_chain_depth() -> int:
    """Longest ``inherits`` chain of the damage-cause rules that
    ``factory.default_mechanics_rules`` builds ``cause_inherits`` from (one
    table, broadcast to every entity). The parent walk cannot visit further, so
    the remaining ``DAMAGE_COUNT`` iterations are no-ops; a cycle keeps them."""

    from hytalegym.rulesets.damage_causes import load_damage_cause_rules

    parents = {rule.asset_id: rule.inherits for rule in load_damage_cause_rules()}
    deepest = 0
    for cause in parents:
        seen, hops = {cause}, 0
        while parents.get(cause) is not None:
            cause = parents[cause]
            if cause in seen:
                return DAMAGE_COUNT
            seen.add(cause)
            hops += 1
        deepest = max(deepest, hops)
    return min(deepest, DAMAGE_COUNT)


_CAUSE_CHAIN_DEPTH = _cause_chain_depth()


def _resolve_damage_resistance_amount(
''',
        "perf", _IDENTITY,
        "hot spot 1 of the rendered gen-5 profile: fori_loop(0, DAMAGE_COUNT=15) -> 0.5.7 chain depth 1",
    ),
    Patch(
        "resistance_walk_loop_bound", _STATUSES,
        "        0,\n        DAMAGE_COUNT,\n        visit_parent,\n",
        "        0,\n        _CAUSE_CHAIN_DEPTH,\n        visit_parent,\n",
        "perf", _IDENTITY, "uses the bound defined by resistance_walk_bounded_by_cause_depth",
    ),
)

OPTIMIZED = replace(
    OVERLAY,
    patches=OPTIMIZATIONS,
    description="identity render plus bit-exact optimisation patches (opt-in)",
)
