"""track's training strategy: DAgger from look_at v2 aimed at entity 1, into track's own module, on the shared frozen trunk
(``training/skill_dagger_v3.py``, the V2 trainer).

What makes the strategy track's (``TRAINING.md`` here):
- the teacher: look_at v2 aimed at entity 1's current position, no lead, never done (``teacher.py``); track is a holding
  skill, so its share is the ticks in view within reach, not a first hit;
- entity 1 is a person on a straight path crossing in front of the spawn body, 3 to 10 blocks out, at walk or sneak
  (held out: a speed between them, 2.0 to 2.3 blocks a second); the path is run by the environment's mover
  (``task.py``);
- beside it: walk_to, by its own teacher, in half the lanes ("with walk_to"), toward a goal drawn as walk_to's own
  training goals, so the module tracks while the body walks (``walk_to/skill.py``, ``sample``);
- sizes: A2's, 6 iterations x 128 lanes x 2,000 Adam steps (``training/skill_dagger.SkillTrainConfig``).

It runs through ``training/skill_dagger_v3.py --skill track``. This file has no command line of its own, as come_here's.

Its smoke and its run wait on the owner's approval (``TRAINING.md`` stage 2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.track import skill as track_skill
from experimental.neural_agent.skill_agent.neural_skills.walk_to import skill as walk_skill
from experimental.neural_agent.skill_agent.training.skill_dagger import (
    Lanes,
    SkillTrainConfig,
    Strategy,
    make_lanes,
)

KINDS = ("alone", "with walk_to")


def episodes(key: jax.Array, kind: jax.Array, held_out: bool) -> Lanes:
    """track's lanes: a person's path in every lane, and a walk beside it where ``kind`` is "with walk_to"."""

    batch = kind.shape[0]
    key_yaw, key_track, key_walk = jax.random.split(key, 3)
    spawn_yaw = jax.random.uniform(key_yaw, (batch,), dtype=jnp.float32, minval=-180.0, maxval=180.0)
    track = track_skill.sample(key_track, batch, spawn_yaw, held_out=held_out)
    walk = walk_skill.sample(key_walk, batch, spawn_yaw, held_out=False)
    active = jnp.stack([jnp.ones((batch,), dtype=jnp.bool_), kind == KINDS.index("with walk_to")], axis=1)
    return make_lanes(kind, (track, walk), active, spawn_yaw)


def strategy(config: SkillTrainConfig = SkillTrainConfig()) -> Strategy:
    return Strategy(spec=track_skill.spec(), beside=(walk_skill.spec(),), kinds=KINDS, episodes=episodes, config=config)
