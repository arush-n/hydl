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

Reset (2026-10-22, #107 item 2): ``construct`` sets the factory defaults only. ``arsenal_runtime_config`` builds the rules
from the loadout's ``resource_maximum`` (``arsenal/factory.py`` ``mechanics_rules_for_loadout``), and a weapon profile
has no mana maximum (the iron sword's ``resources`` is signature energy only), so the reset read 0 / 0
for the melee duel and the single-agent env. ``construct_loadout`` is the second hook: it fills the Mana stat on the loadout the config
is built from, where the loadout carries none, so the rules and the reset both read 100 / 100.
"""

from __future__ import annotations

import importlib
import inspect
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


def _rendered_root(original) -> str:
    """The rendered package (``hytalegym.sim.genN.engine``) that ``original`` was rendered into."""

    module = inspect.unwrap(original).__module__
    root, marker, _ = module.partition(".combat.")
    if not marker:
        raise ValueError(f"{module} is not a rendered combat module")
    return root


def _with_mana_stat(loadout: Any, mana: int, maximum: float, initial: float) -> Any:
    """``loadout`` with the Mana stat where it carries none (maximum 0): maximum and initial from the table.

    A loadout that names its own mana maximum keeps its maximum and initial; the stat fills only the empty slots.
    """

    import jax.numpy as jnp

    current_maximum = jnp.asarray(loadout.resource_maximum)
    current_initial = jnp.asarray(loadout.resource_initial)
    empty = current_maximum[..., mana] <= 0.0
    return loadout._replace(
        resource_maximum=current_maximum.at[..., mana].set(
            jnp.where(empty, jnp.float32(maximum), current_maximum[..., mana])
        ),
        resource_initial=current_initial.at[..., mana].set(
            jnp.where(empty, jnp.float32(initial), current_initial[..., mana])
        ),
    )


def construct_loadout(original: Any, profile: Any) -> Any:
    """Hook for ``arsenal_runtime_config``: the Mana stat (maximum and initial, table ``stats/stat:Mana``) on the loadout.

    The config builder reads the input loadout for its resource maxima and initial values, so filling them here reaches
    the rules and the reset with no other change. Gen 5 (switch off) returns ``original`` unchanged.
    """

    if not profile.constructs.mana_pool:
        return original

    from hytalegym.sim.common.content import load

    stat = load(profile).row("stats", "stat:Mana")
    maximum = float(stat["max"])
    initial = float(stat["initial_value"])

    @wraps(original)
    def arsenal_runtime_config(loadout: Any, *args: Any, **kwargs: Any) -> Any:
        contract = importlib.import_module(_rendered_root(original) + ".combat.mechanics.schema.contract")
        return original(_with_mana_stat(loadout, contract.RESOURCE_MANA, maximum, initial), *args, **kwargs)

    return arsenal_runtime_config
