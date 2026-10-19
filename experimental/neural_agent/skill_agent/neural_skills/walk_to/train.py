"""walk_to's training strategy: DAgger from teacher v3 into walk_to's own module, on the shared frozen trunk.

What makes the strategy walk_to's (``TRAINING.md`` here):
- the teacher: v3, which stops 0.25 blocks inside the radius (``teacher.py``). Without the margin the student parked
  a few centimetres outside the radius in A2 (README 2.16);
- beside it: look_at, by its own teacher, in half the lanes ("with look_at"), so the module learns to walk while
  another request is active (owner, 2026-10-19: a module reads the other active requests). Those looks are placed
  within 30 degrees of the walk bearing (``look_at/skill.py``, ``beside_walk``);
- held out: walk targets at bearings 180 to 270 degrees from the spawn body (``skill.py``, ``in_rear_left``);
- sizes: A2's, 6 iterations x 128 lanes x 2,000 Adam steps (``training/skill_dagger.SkillTrainConfig``).

The method is the shared DAgger block (``training/skill_dagger.py``). Training is paused (owner, 2026-10-19): nothing
runs this until it reopens.

Usage: python -m experimental.neural_agent.skill_agent.neural_skills.walk_to.train --trunk TRUNK.npz --out DIR
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.look_at import skill as look_skill
from experimental.neural_agent.skill_agent.neural_skills.walk_to import skill as walk_skill
from experimental.neural_agent.skill_agent.training.skill_dagger import (
    Lanes,
    SkillTrainConfig,
    Strategy,
    make_lanes,
    run_main,
)

KINDS = ("alone", "with look_at")


def episodes(key: jax.Array, kind: jax.Array, held_out: bool) -> Lanes:
    """walk_to's lanes: a walk goal in every lane, and a look beside it where ``kind`` is "with look_at"."""

    batch = kind.shape[0]
    key_yaw, key_walk, key_look = jax.random.split(key, 3)
    spawn_yaw = jax.random.uniform(key_yaw, (batch,), dtype=jnp.float32, minval=-180.0, maxval=180.0)
    walk = walk_skill.sample(key_walk, batch, spawn_yaw, held_out=held_out)
    look = look_skill.beside_walk(key_look, walk.target_xz, spawn_yaw)
    active = jnp.stack([jnp.ones((batch,), dtype=jnp.bool_), kind == KINDS.index("with look_at")], axis=1)
    return make_lanes(kind, (walk, look), active, spawn_yaw)


def strategy(config: SkillTrainConfig = SkillTrainConfig()) -> Strategy:
    return Strategy(spec=walk_skill.spec(), beside=(look_skill.spec(),), kinds=KINDS, episodes=episodes, config=config)


def main(argv: Sequence[str] | None = None) -> int:
    return run_main(strategy, "walk_to's training strategy: DAgger from teacher v3 into its own module.", argv)


if __name__ == "__main__":
    raise SystemExit(main())
