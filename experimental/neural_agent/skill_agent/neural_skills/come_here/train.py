"""come_here's training strategy: DAgger from walk_to v3's rule toward entity 1, into come_here's own module, on the shared
frozen trunk (``training/skill_dagger_v3.py``, the V2 trainer).

What makes the strategy come_here's (``TRAINING.md`` here):
- the teacher: walk_to v3's rule aimed at entity 1's current position (``teacher.py``), stop margin 0.25 blocks;
- entity 1 placed per lane (``task.py``): 3 to 15 blocks from the spawn at any bearing; held-out lanes place it at
  bearings 180 to 270 degrees from the spawn body (``skill.py``, ``sample``);
- beside it: look_at, by its own teacher, in half the lanes ("with look_at"), as walk_to's: its looks are placed within
  30 degrees of the bearing to entity 1 (``look_at/skill.py``, ``beside_walk``);
- sizes: A2's, 6 iterations x 128 lanes x 2,000 Adam steps (``training/skill_dagger.SkillTrainConfig``).

It runs through ``training/skill_dagger_v3.py --skill come_here``. This file has no command line of its own: the v1
trainer (``skill_dagger.run_main``) does not place entity 1 per lane or resolve its pointer.

Its smoke and its run wait on the owner's approval (``TRAINING.md`` stages 2 and 3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.come_here import skill as come_skill
from experimental.neural_agent.skill_agent.neural_skills.look_at import skill as look_skill
from experimental.neural_agent.skill_agent.training.skill_dagger import (
    Lanes,
    SkillTrainConfig,
    Strategy,
    make_lanes,
)

KINDS = ("alone", "with look_at")


def episodes(key: jax.Array, kind: jax.Array, held_out: bool) -> Lanes:
    """come_here's lanes: entity 1 placed in every lane, and a look beside it where ``kind`` is "with look_at"."""

    batch = kind.shape[0]
    key_yaw, key_come, key_look = jax.random.split(key, 3)
    spawn_yaw = jax.random.uniform(key_yaw, (batch,), dtype=jnp.float32, minval=-180.0, maxval=180.0)
    come = come_skill.sample(key_come, batch, spawn_yaw, held_out=held_out)
    look = look_skill.beside_walk(key_look, come.target_xz, spawn_yaw)
    active = jnp.stack([jnp.ones((batch,), dtype=jnp.bool_), kind == KINDS.index("with look_at")], axis=1)
    return make_lanes(kind, (come, look), active, spawn_yaw)


def strategy(config: SkillTrainConfig = SkillTrainConfig()) -> Strategy:
    return Strategy(spec=come_skill.spec(), beside=(look_skill.spec(),), kinds=KINDS, episodes=episodes, config=config)
