"""NPC set-velocity knockback model. Switch ``Constructs.npc_velocity_model``; P3.

Engine-code change, no table input: 0.5.10 zeroes then adds; 0.7.2
``Role.processSetVelocityInstruction`` applies one set velocity (about 23x for
set-type NPC knockback, TRAINING-0.7.2.md item 2). The bridge clears the
agent's own horizontal motion on both versions, so this reaches other NPCs only.
Differential: NPC-on-NPC knockback displacement on live 0.7.2.
"""

from __future__ import annotations

from typing import Any

SWITCH = "npc_velocity_model"
IMPLEMENTED = False


def apply_set_velocity(state: Any, velocity: Any) -> Any:
    raise NotImplementedError("npc velocity construct: design only (mechanics_072.md)")
