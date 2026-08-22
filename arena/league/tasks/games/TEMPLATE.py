"""COPY THIS FILE to author a new game.

    cp arena/tasks/games/TEMPLATE.py arena/tasks/games/mygame.py

Then edit the marked sections and register it. Registration validates the game
and refuses it if it does not conform, so a mistake here fails immediately
rather than after an hour of training that scored nothing.

Run the gate on its own while iterating::

    from arena.tasks.framework.validate import report
    print(report(MY_GAME, build_scenes=True))

What the gate checks
--------------------
* ``name`` is a bare segment -- scenes are named ``arena/<task>.<difficulty>``.
* every entry in ``heads_exercised`` is a real action head.
* ``loadout`` and every ``opponent_profile`` are real combat profiles.
* the ladder does not consist entirely of armed rungs.
* **the success criterion is executed** against a synthetic rollout and must
  return one **boolean per environment**, shape ``(batch,)``.
* an optional reward contract is executed and must return finite
  ``float32[batch]``.
* with ``build_scenes=True``, every difficulty actually builds.

The two traps the framework closes for you
------------------------------------------
You do not need to remember either of these; ``Task.build_scene`` handles both.

1. A scene that declares opponents without a perception provider reports
   ``target_visible == 0.0`` forever, makes no attack legal, and pays zero
   reward -- which reads exactly like a broken weapon.
2. ``Difficulty.opponent_armed`` has no default. Arena binds either the JAX
   first-legal controller or the explicit inert controller, so the teaching
   behavior cannot change silently.
"""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.criteria import (  # noqa: F401 - the full menu, for editing
    above,
    all_of,
    any_of,
    at_end,
    at_start,
    below,
    col,
    ever_below,
    fraction_of_time,
    maximum,
    mean,
    minimum,
    negate,
    recorded,
    span,
    total,
    within,
)
from arena.tasks.framework.registry import register
from arena.tasks.framework.goals import stay_alive
from arena.tasks.framework.rewards import RewardConfig, reward_for_goal
# from arena.worlds import PublishedWorldSpec, WorldSpec  # choose one below


# ---------------------------------------------------------------------------
# 1. WHAT IS THIS GAME FOR?
#    Say what skill it isolates and why the plain objective fails to teach it.
# ---------------------------------------------------------------------------
NAME = "template"
DESCRIPTION = (
    "REPLACE ME: one sentence on the skill this isolates, and why the plain "
    "fight objective does not teach it."
)

# Which action heads should this game make the agent use? Coverage is checked
# against the real action surface, so a typo fails at registration.
HEADS_EXERCISED = (
    "locomotion_gait_compass",
    # "yaw_delta_bins",
    # "ability_none_plus_slots",
)

# The agent's weapon profile.
LOADOUT = "iron_sword"


# ---------------------------------------------------------------------------
# 2. THE DIFFICULTY LADDER -- easiest first.
#    Declaration order IS the progression: arena.curriculum.ladder.ladder_for
#    walks these in order.
#
#    Open with an unarmed rung. Against a lethal opponent, not-engaging is a
#    cheap local optimum (death -20, damage -1/hit) and winning is rare under
#    exploration (+50 completion) -- a measured 30-update run scored ZERO
#    successes on every policy while driving deaths to 0/16.
# ---------------------------------------------------------------------------
DIFFICULTIES = (
    Difficulty(
        "easy",
        opponent_armed=False,
        opponent_profile="iron_sword",
    ),
    Difficulty(
        "hard",
        opponent_armed=True,
        opponent_profile="iron_sword",
        # Any of the 118 CombatParams fields may be overridden:
        # agent_max_health=80.0,
        # target_max_health=180.0,
        # parameters={"completion_reward": 75.0},
        # sensor_range=48.0,
    ),
)


# ---------------------------------------------------------------------------
# 3. WHAT COUNTS AS SUCCESS?
#    Compose: col(...) -> reducer -> comparison -> combinator.
#    Reducers collapse TICKS ONLY and return one value per environment.
#
#    Patterns:
#      terminal state      above(at_end(col("self_f32", "alive")), 0.5)
#      masked threshold    ever_below(value, 1e-6, available=visibility)
#      sustained           above(mean(col("target_f32", "visible")), 0.9)
#      ever crossed        above(maximum(col("interaction_f32","reach_fraction")), 0.999)
#      combined            all_of(a, b) / any_of(a, b) / negate(a)
#
#    Anything not expressible here can be a plain callable taking the record
#    and returning a (batch,) bool array.
# ---------------------------------------------------------------------------
SUCCESS = stay_alive()

# Success is final evaluation; reward is the per-step training signal. Keep
# native combat reward off unless it is intentionally part of this game.
# Observation-derived rewards must declare availability. Built-in target-health
# rewards already abstain under occlusion and use exact episode-success evidence
# for the defeat bonus; never infer completion from generic `done`.
REWARD = reward_for_goal(
    SUCCESS,
    RewardConfig(native_scale=0.0, state_scale=0.01, step_penalty=0.001),
)


TEMPLATE_TASK = Task(
    name=NAME,
    description=DESCRIPTION,
    heads_exercised=HEADS_EXERCISED,
    loadout=LOADOUT,
    difficulties=DIFFICULTIES,
    success=SUCCESS,
    reward=REWARD,
    # Give this task exact generated terrain when it needs a world:
    # world=WorldSpec(
    #     "plains_pool", "Zone1_Plains1", 570057, pool_seeds=(570061,)
    # ),
    # Or bind a project-authored publication. It may contain many exact seeds;
    # Arena keeps only this deterministic one-to-four-world page resident:
    # world=PublishedWorldSpec(
    #     "my_authored_worlds",
    #     "path/to/environment.publication",
    #     resident_capacity=4,
    #     window_index=0,
    # ),
    # scene_options={"providers": {...}},
)


def install(*, replace: bool = False) -> Task:
    """Register this game. Import-time side effects are deliberately avoided."""

    return register(TEMPLATE_TASK, replace=replace, build_scenes=True)


__all__ = ["TEMPLATE_TASK", "install"]
