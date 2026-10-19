"""look_at's training strategy: DAgger from teacher v2 into look_at's own module, on the shared frozen trunk.

What makes the strategy look_at's (``TRAINING.md`` here):
- the request: the unit direction to the target in the head's frame (``skill.py``, ``spec``). In the body frame a
  held-out look stays outside the trained range for the whole look; in the head's frame it moves back into it as the
  look closes (run C', README 2.16);
- the teacher: the registered v2, the rule on the observed state (``teacher.py``);
- beside it: walk_to, by its own teacher (v3), in half the lanes ("while walking"), with the look placed within 30
  degrees of the walk bearing (``skill.py``, ``beside_walk``), so the module learns to look while the body walks;
- held out: look-only targets 25 to 40 degrees left of the body (``skill.py``, ``in_look_held_out``); a lane while
  walking is held out by its walk target, at bearings 180 to 270 degrees, as in A2;
- sizes: A2's, 6 iterations x 128 lanes x 2,000 Adam steps (``training/skill_dagger.SkillTrainConfig``).

The method is the shared DAgger block (``training/skill_dagger.py``). Training is paused (owner, 2026-10-19): nothing
runs this until it reopens.

Usage: python -m experimental.neural_agent.skill_agent.neural_skills.look_at.train --trunk TRUNK.npz --out DIR
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.look_at import skill as look_skill
from experimental.neural_agent.skill_agent.neural_skills.walk_to import skill as walk_skill
from experimental.neural_agent.skill_agent.training.blocks import broadcast_rows
from experimental.neural_agent.skill_agent.training.skill_dagger import (
    Lanes,
    SkillTrainConfig,
    Strategy,
    make_lanes,
    run_main,
)

KINDS = ("alone", "while walking")


def episodes(key: jax.Array, kind: jax.Array, held_out: bool) -> Lanes:
    """look_at's lanes: a look-only goal, or (``kind`` "while walking") a walk with a look placed beside it."""

    batch = kind.shape[0]
    key_yaw, key_alone, key_walk, key_beside = jax.random.split(key, 4)
    spawn_yaw = jax.random.uniform(key_yaw, (batch,), dtype=jnp.float32, minval=-180.0, maxval=180.0)
    walking = kind == KINDS.index("while walking")
    alone = look_skill.sample(key_alone, batch, spawn_yaw, held_out=held_out)
    walk = walk_skill.sample(key_walk, batch, spawn_yaw, held_out=held_out)
    beside = look_skill.beside_walk(key_beside, walk.target_xz, spawn_yaw)
    look = jax.tree_util.tree_map(lambda near, own: jnp.where(broadcast_rows(walking, near), near, own), beside, alone)
    active = jnp.stack([jnp.ones((batch,), dtype=jnp.bool_), walking], axis=1)
    return make_lanes(kind, (look, walk), active, spawn_yaw)


def strategy(config: SkillTrainConfig = SkillTrainConfig()) -> Strategy:
    return Strategy(spec=look_skill.spec(), beside=(walk_skill.spec(),), kinds=KINDS, episodes=episodes, config=config)


def main(argv: Sequence[str] | None = None) -> int:
    return run_main(strategy, "look_at's training strategy: DAgger from teacher v2 into its own module.", argv)


if __name__ == "__main__":
    raise SystemExit(main())
