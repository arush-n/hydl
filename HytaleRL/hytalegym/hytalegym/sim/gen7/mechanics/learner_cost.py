"""Gen 7 (row 8f): the learner pays a native NPC instant stat cost once; every other entity pays it twice.

0.7.2's ``InteractionManager.doTickChain`` runs ``simulationTick`` and then ``serverTick`` when the manager has no remote
client, and ``SimpleInstantInteraction`` reaches ``firstRun`` from both, so an NPC's instant ``ChangeStat`` adds twice. NPC
managers are built with a null PlayerRef (``NPCInteractionSystems$AddSimulationManagerSystem``), which is the engine's
``NATIVE_NPC_INSTANT_ADD_STAT_APPLICATIONS = 2``. The bridge leases remote-client mode for every policy-controlled chain
(``NativeEnvironmentSession.bindSyntheticCombatClient``; ``NativeSyntheticCombatClient.bind`` calls
``setHasRemoteClient(true)``), which skips the simulation pass. So the learner's own abilities apply the stat once. Live
(jar 3865F45F, ``thrust_lunge_Kweebec_Razorleaf_jar14_hold30.json``), the iron sword's thrust, authored 4 stamina, takes
10 to 6, where the render took 8. Autonomous NPCs, every entity but ``AGENT_ENTITY``, keep paying twice.

Bytecode and receipt survey: ``tmp/lane_cost_double/FINDINGS.md`` (coordinator, 2026-10-24).
"""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT,
    RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE,
)
from hytalegym.jax.combat.types import AGENT_ENTITY


def learner_single_application(effective_cost, authored_cost, cost_kind):
    """``effective_cost`` with the learner's native NPC instant costs at one application.

    All three arrays share their shape, with the entity axis second ([B, E, ...]). Where the entity is ``AGENT_ENTITY``
    and the kind is a native NPC instant add (kinds 2 and 3), the cost is the authored amount; elsewhere it is unchanged.
    """

    entity = jnp.arange(effective_cost.shape[1]).reshape((1, -1) + (1,) * (effective_cost.ndim - 2))
    native = (cost_kind == RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT) | (
        cost_kind == RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE
    )
    return jnp.where((entity == AGENT_ENTITY) & native, authored_cost, effective_cost)


__all__ = ["learner_single_application"]
