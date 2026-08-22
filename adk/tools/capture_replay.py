"""Capture agent-state replays as JSON for the replay viewer.

    python adk/tools/capture_replay.py [output.json]

Three runs, chosen so the viewer shows a real finding rather than a generic
demo (see adk/CHANGELOG.md, "A bare make_scene duel is blind"):

    blind    -- bare make_scene, uniform_legal      opponent never perceived
    seeing   -- + world provider, uniform_legal     controlled: same policy
    fighting -- + world provider, strike_when_ready combat actually happens

Run from the repository root so ``agent`` and ``hytalegym`` both import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_ROOT, _ROOT / "HytaleRL" / "hytalegym"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import jax  # noqa: E402  -- imports follow the sys.path bootstrap above
import jax.numpy as jnp  # noqa: E402

from adk import AgentKit, AgentSpec  # noqa: E402
from adk.policy.fields import GROUP_FEATURES  # noqa: E402
from adk.probes import strike_when_ready, uniform_legal  # noqa: E402
from adk.probes.policies import HEAD_SPANS  # noqa: E402
from adk.runtime.scenes import OPEN_FLAT_CONTROL_SCENE  # noqa: E402

STEPS = 256
ENVIRONMENT = 0  # which batch row to record

_COMBAT = list(GROUP_FEATURES["combat_f32"])
_VISIBLE = _COMBAT.index("target_visible")
_OBSERVED_DISTANCE = _COMBAT.index("visible_target_planar_distance")
_ATTACK_EXECUTING = _COMBAT.index("agent_attack_executing")
_ATTACK_PROGRESS = _COMBAT.index("target_attack_progress")

#: Target attack phases, in the order the observation lays them out.
TARGET_PHASES = ("idle", "windup", "sweep", "recovery", "cooldown")
_PHASE_COLUMNS = [_COMBAT.index(f"target_phase_{name}") for name in TARGET_PHASES]

DUEL = dict(loadouts=["iron_sword", "iron_sword"], opponents="iron_sword")


def _record(transition):
    observation = transition.next_actor_input.legal_observation
    combat = transition.next_state.environment.runtime.combat
    combat_f32 = observation.base.combat_f32
    return {
        "position": combat.position,
        "yaw": combat.yaw,
        "health": combat.health,
        "grounded": combat.agent_grounded,
        "visible": combat_f32[..., _VISIBLE],
        "observed_distance": combat_f32[..., _OBSERVED_DISTANCE],
        "attack_executing": combat_f32[..., _ATTACK_EXECUTING],
        "target_phase": combat_f32[..., jnp.asarray(_PHASE_COLUMNS)],
        "target_attack_progress": combat_f32[..., _ATTACK_PROGRESS],
        "attack_requested": transition.info.combat_info.attack_requested,
        "attack_accepted": transition.info.combat_info.attack_accepted,
        "damage_dealt": transition.info.arsenal_info.damage_dealt,
        "reward": transition.reward,
        "done": transition.done,
        "action": transition.action_factors,
    }


def capture(kit, label, scene, policy, *, loadout=None):
    """Roll out one scene/policy pair and return a JSON-ready dict.

    ``loadout`` must match the scene's own when the scene is homogeneous --
    ``build()`` rejects a mismatch rather than silently preferring one, so a bow
    scene needs it passed explicitly.
    """

    spec = f"replay/{label}"
    options = {} if loadout is None else {"loadout": loadout}
    kit.register(AgentSpec(name=spec, scene=scene.name, **options))
    handle = kit.make(spec, num_envs=2)
    _final, rec = handle.compile_collector(policy, _record, STEPS)(
        jax.random.key(0)
    )

    position = rec["position"]        # (steps, batch, entity, 3)
    health = rec["health"]            # (steps, batch, entity)
    agent = position[:, ENVIRONMENT, 0, :]
    target = position[:, ENVIRONMENT, 1, :]
    planar = jnp.linalg.norm((target - agent)[:, [0, 2]], axis=-1)
    action = rec["action"]
    action = action[:, ENVIRONMENT, :] if action.ndim == 3 else action[:, ENVIRONMENT]

    def series(values, digits):
        return [round(float(v), digits) for v in values]

    phase = rec["target_phase"][:, ENVIRONMENT, :]

    return {
        "label": label,
        "steps": STEPS,
        "agent_x": series(agent[:, 0], 4),
        "agent_y": series(agent[:, 1], 4),
        "agent_z": series(agent[:, 2], 4),
        "target_x": series(target[:, 0], 4),
        "target_y": series(target[:, 1], 4),
        "target_z": series(target[:, 2], 4),
        "agent_yaw": series(rec["yaw"][:, ENVIRONMENT, 0], 3),
        "target_yaw": series(rec["yaw"][:, ENVIRONMENT, 1], 3),
        "agent_health": series(health[:, ENVIRONMENT, 0], 2),
        "target_health": series(health[:, ENVIRONMENT, 1], 2),
        "true_distance": series(planar, 3),
        "observed_distance": series(rec["observed_distance"][:, ENVIRONMENT], 3),
        "visible": [int(v > 0.5) for v in rec["visible"][:, ENVIRONMENT]],
        "grounded": [int(v) for v in rec["grounded"][:, ENVIRONMENT]],
        "attack_requested": [
            int(v) for v in rec["attack_requested"][:, ENVIRONMENT]
        ],
        "attack_accepted": [
            int(v) for v in rec["attack_accepted"][:, ENVIRONMENT]
        ],
        "attack_executing": [
            int(v > 0.5) for v in rec["attack_executing"][:, ENVIRONMENT]
        ],
        "damage_dealt": series(rec["damage_dealt"][:, ENVIRONMENT], 4),
        "target_phase": [int(row.argmax()) for row in phase],
        "target_attack_progress": series(
            rec["target_attack_progress"][:, ENVIRONMENT], 3
        ),
        "reward": series(rec["reward"][:, ENVIRONMENT], 5),
        "done": [int(v) for v in rec["done"][:, ENVIRONMENT]],
        "action": [[int(x) for x in row] for row in action],
    }


def main() -> None:
    kit = AgentKit()
    bare = kit.make_scene("replay/bare", **DUEL)
    seeing = kit.make_scene(
        "replay/seeing",
        providers=dict(OPEN_FLAT_CONTROL_SCENE.environment_kwargs),
        **DUEL,
    )

    runs = [
        capture(kit, "blind", bare, uniform_legal),
        capture(kit, "seeing", seeing, uniform_legal),
        capture(kit, "fighting", seeing, strike_when_ready()),
    ]

    payload = {
        "steps": STEPS,
        "head_names": list(HEAD_SPANS),
        "target_phases": list(TARGET_PHASES),
        "runs": runs,
    }

    destination = Path(
        sys.argv[1] if len(sys.argv) > 1
        else Path(__file__).with_name("replay_data.json")
    )
    destination.write_text(json.dumps(payload), encoding="utf-8")

    print(f"wrote {destination} ({destination.stat().st_size / 1024:.1f} KiB)")
    for run in runs:
        landed = sum(1 for v in run["damage_dealt"] if v)
        swings = sum(run["attack_requested"])
        print(
            f"  {run['label']:9s} visible {sum(run['visible']):3d}/{STEPS} | "
            f"swings {swings:3d} accepted {sum(run['attack_accepted']):3d} "
            f"landed {landed:2d} | "
            f"reward {sum(run['reward']):.1f} | "
            f"target hp {run['target_health'][0]:.0f}->"
            f"{run['target_health'][-1]:.0f} | "
            f"agent hp {run['agent_health'][0]:.0f}->"
            f"{run['agent_health'][-1]:.0f} | "
            f"y {min(run['agent_y']):.2f}-{max(run['agent_y']):.2f}"
        )


if __name__ == "__main__":
    main()
