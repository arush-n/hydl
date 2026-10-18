"""Mana pool (0.7.2). Switch ``Constructs.mana_pool``; phase 4b, row 7a (FIDELITY-0.7.2.md section (h)).

0.7.2 ``Server/Entity/Stats/Mana.json`` (``stats/stat:Mana`` in ``sim/gen7/data/stats.json``): initial 100, maximum
100, reset to the maximum, and an Additive +1 regeneration every 0.3 s while Alive. The second regeneration entry is
a Percentage rule for Creative mode and does not apply to the bridge's survival-mode agent. The gen-5 factory
(``jax/combat/mechanics/factory.py``) has a maximum of 0 and +1 per 0.2 s. This construct wraps
``default_mechanics_rules`` for a profile with ``mana_pool`` set and reads its numbers from the content table;
gen 5 returns the original function unchanged.

Differential: the native agent refilled from 0 under ``native_duel`` (live 0.7.2 bridge, the bridge's
``native_agent_mana`` info key) gains +1 per 8.95 world ticks, 0.2982 s, against the 0.3 s bar (5 %);
the rendered engine's ``tick_resources`` from 0 at 1/30 s per tick is the JAX side.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

SWITCH = "mana_pool"
IMPLEMENTED = True


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``mana_pool`` the maximum and the regeneration from the content table."""

    if not profile.constructs.mana_pool:
        return original

    from hytalegym.sim.common.content import load

    stat = load(profile).row("stats", "stat:Mana")
    if float(stat["initial_value"]) != float(stat["max"]):
        raise ValueError("Mana reset_type MaxValue needs initial_value == max; the table disagrees")
    regen = [
        entry
        for entry in stat["regenerating"]
        if entry["regen_type"] == "Additive" and entry["conditions"] == [{"Id": "Alive"}]
    ]
    if len(regen) != 1:
        raise ValueError(f"expected one Additive Alive regeneration for Mana, found {len(regen)}")
    maximum = float(stat["max"])
    amount = float(regen[0]["amount"])
    interval = float(regen[0]["interval"])
    env = original.__globals__
    mana = env["RESOURCE_MANA"]

    @wraps(original)
    def default_mechanics_rules(batch_size: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("mana_maximum", maximum)
        rules = original(batch_size, **kwargs)
        return rules._replace(
            resource_regen_amount=rules.resource_regen_amount.at[..., mana].set(amount),
            resource_regen_interval_seconds=rules.resource_regen_interval_seconds.at[..., mana].set(interval),
        )

    return default_mechanics_rules
