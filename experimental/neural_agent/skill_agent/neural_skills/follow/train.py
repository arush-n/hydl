"""follow's training strategy: DAgger from walk_to v3's rule toward the band's middle behind a person, into follow's own
module, on the shared frozen trunk (``training/skill_dagger_v3.py``, the V2 trainer).

What makes the strategy follow's (``TRAINING.md`` here, and ``README.md``):
- the teacher: walk_to v3's rule aimed 3.5 blocks behind the person, sprinting, never done (``teacher.py``); follow is a
  holding skill, so its share is the ticks inside the band over the horizon, not a first hit;
- the person: a looped random walk, walking (1.5) or sneaking (2.75) blocks a second (held out: 2.0 to 2.3); the learner
  starts 3.5 blocks behind it, facing it (``task.py``); the environment's mover moves the person (the trained skill's
  goals carry it: ``training/skill_dagger_v3.rollout``);
- beside it: track, by its own teacher, in half the lanes ("with track"), so the module follows while watching the
  person. track's goals are sampled as its own strategy samples them; its teacher reads the same person;
- sizes: A2's, 6 iterations x 128 lanes x 2,000 Adam steps (``training/skill_dagger.SkillTrainConfig``).

It runs through ``training/skill_dagger_v3.py --skill follow``. This file has no command line of its own, as come_here's.
Its smoke and its run wait on the owner's approval (``TRAINING.md`` stages 2 and 3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.follow import skill as follow_skill
from experimental.neural_agent.skill_agent.neural_skills.follow.task import FollowTasks
from experimental.neural_agent.skill_agent.neural_skills.track import skill as track_skill
from experimental.neural_agent.skill_agent.training.skill_dagger import (
    Lanes,
    SkillTrainConfig,
    Strategy,
    make_lanes,
)

KINDS = ("alone", "with track")


def episodes(key: jax.Array, kind: jax.Array, held_out: bool) -> Lanes:
    """follow's lanes: a person's path in every lane, and track beside it where ``kind`` is "with track". The learner's spawn
    yaw is the person's (it faces the person), so track's goals are drawn at that yaw."""

    batch = kind.shape[0]
    key_follow, key_track = jax.random.split(key)
    follow = FollowTasks.sample(key_follow, batch, held_out=held_out)
    track = track_skill.sample(key_track, batch, follow.spawn_yaw, held_out=False)
    active = jnp.stack([jnp.ones((batch,), dtype=jnp.bool_), kind == KINDS.index("with track")], axis=1)
    return make_lanes(kind, (follow, track), active, follow.spawn_yaw)


def strategy(config: SkillTrainConfig = SkillTrainConfig()) -> Strategy:
    return Strategy(spec=follow_skill.spec(), beside=(track_skill.spec(),), kinds=KINDS, episodes=episodes, config=config)
