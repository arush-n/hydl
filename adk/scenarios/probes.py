"""Single-purpose scripted agents, used to interrogate the environment.

Companion to the scenarios in this package. Those define *scenes with declared
ceilings*; these define the smallest possible *drivers* for them. Neither is an
agent in the RL sense and neither should grow into one.

These are **not** agents in the RL sense and are not meant to become one. Each
one does exactly one thing, so that whatever the environment and the opponent
NPC do in response is attributable to that one thing. A learned policy mixes
every behaviour together and tells you nothing about which part the environment
answered.

The value is in the *differences between rows*. "The agent took 40 damage" means
nothing on its own; "the idle agent took 40 and the guarding agent took 40"
means guarding does not work, and that is a fidelity gap worth a bug.

Every policy respects the published legality mask, and every run also reports
``action_valid`` -- because per-head legality does not imply joint validity
(ability and guard are mutually exclusive while both are masked legal), and a
rejected action must never be counted as a behaviour the environment answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_COMPASS_DIRECTIONS,
    ARSENAL_POLICY_LOCOMOTION_DODGE_START,
    DODGE_ACTION_COUNT,
)

#: Head widths in transport column order, read from the live contract. These
#: were hardcoded, which desynchronised this file the moment movement and dodge
#: merged into ``locomotion_gait_compass`` -- the widths still summed to a
#: plausible mask, so the slicing below silently read the wrong columns.
HEAD_SIZES = tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES)
HEAD_STARTS = np.cumsum((0,) + HEAD_SIZES[:-1])

#: ``locomotion_gait_compass`` packs idle, four gait rings of eight compass
#: directions, and the four dodges into one categorical. Walking a direction is
#: choice ``1..8`` -- the same numbering the retired nine-wide movement head
#: used -- so option values carry over and only the column moved.
WALK_SPAN = (1, ARSENAL_POLICY_COMPASS_DIRECTIONS + 1)
DODGE_SPAN = (
    ARSENAL_POLICY_LOCOMOTION_DODGE_START,
    ARSENAL_POLICY_LOCOMOTION_DODGE_START + DODGE_ACTION_COUNT,
)


def head_names() -> list[str]:
    return list(ARSENAL_POLICY_ACTION_HEAD_NAMES)


def _in_span(options: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    """Legal options narrowed to one slice of a merged head.

    Movement and dodge share a column now, so a bare ``_pick_active`` on that
    column would let a "walk forward" driver answer with a dodge and vice
    versa. Narrowing first keeps each driver doing the one thing it is named
    after, and an empty result still reports as "impossible" rather than
    silently idling.
    """

    low, high = span
    return options[(options >= low) & (options < high)]


def legal_options(mask: np.ndarray, head: int) -> list[np.ndarray]:
    """Per row, the indices this head currently allows."""

    start = HEAD_STARTS[head]
    window = mask[:, start:start + HEAD_SIZES[head]].astype(bool)
    return [np.flatnonzero(window[row]) for row in range(mask.shape[0])]


#: Filled by :func:`_pick`. Counts how often a policy's *intended* choice was
#: actually legal. Without this a scenario that never once did the thing it is
#: named after still produces a full row of numbers, and the row reads as
#: evidence about the behaviour rather than about its absence.
INTENT: dict[str, list[int]] = {"wanted": [0], "granted": [0]}


def reset_intent() -> None:
    INTENT["wanted"][0] = 0
    INTENT["granted"][0] = 0


def _pick(options: np.ndarray, wanted: int, fallback: int = 0,
          count: bool = True) -> int:
    """`wanted` if it is legal, else `fallback` if legal, else neutral."""

    if count:
        INTENT["wanted"][0] += 1
    if options.size == 0:
        return 0
    if wanted in options:
        if count:
            INTENT["granted"][0] += 1
        return int(wanted)
    if fallback in options:
        return int(fallback)
    return int(options[0])


def _pick_active(options: np.ndarray, neutral: int = 0) -> int:
    """Any legal option that is **not** the no-op for this head.

    Hardcoding a specific index (``_pick(options, 1)``) silently degrades to the
    no-op whenever that one index happens to be illegal -- which produced a
    "dodging does not help" row where the agent had in fact never dodged once.
    The behaviour a scenario is named after must either happen or be reported
    as impossible; it must not quietly become idling.
    """

    INTENT["wanted"][0] += 1
    active = options[options != neutral] if options.size else options
    if active.size == 0:
        return int(neutral)
    INTENT["granted"][0] += 1
    return int(active[0])


#: ``(mask, tick, feedback) -> factors``. ``feedback`` is None on the first
#: tick and afterwards carries last tick's privileged CombatInfo. Open-loop
#: policies ignore it; closed-loop ones must not, which is the whole point.
Policy = Callable[[np.ndarray, int, object], np.ndarray]


class Feedback:
    """What a closed-loop probe is allowed to condition on.

    Deliberately built from **privileged** ``CombatInfo`` plus measured health,
    not from the 8271-column observation. These probes ask "does the
    environment respond correctly when an agent behaves this way"; whether the
    *observation* carries the same facts is a separate question, answered by
    the perception-grounding probe rather than conflated with this one.
    """

    __slots__ = ("distance", "agent_health", "agent_hurt", "target_health")

    def __init__(self, distance, agent_health, agent_hurt, target_health):
        self.distance = distance
        self.agent_health = agent_health
        self.agent_hurt = agent_hurt
        self.target_health = target_health


def make_policy(name: str, names: list[str]) -> Policy:
    """Build one scripted behaviour by name.

    Each returns per-head factor indices for the whole batch. They read only the
    mask, never the observation -- these probe the environment's response, not
    perception.
    """

    ability = names.index("ability_none_plus_slots")
    guard = names.index("guard_off_on")
    # Movement and dodge are the same column; they are separated by option
    # span, not by head.
    move = names.index("locomotion_gait_compass")
    yaw = names.index("yaw_delta_bins")
    dodge = move
    jump = names.index("jump_off_on")

    def base(mask: np.ndarray) -> np.ndarray:
        rows = mask.shape[0]
        out = np.zeros((rows, len(HEAD_SIZES)), dtype=np.int32)
        # Yaw and pitch are signed bins whose neutral is the centre, not zero.
        for head in (names.index("yaw_delta_bins"),
                     names.index("pitch_delta_bins")):
            out[:, head] = HEAD_SIZES[head] // 2
        return out

    def idle(mask, tick, fb=None):
        return base(mask)

    def aggressor(mask, tick, fb=None):
        """Use an ability whenever one is legal. Never guards -- guarding and
        using an ability are mutually exclusive, so mixing them would silently
        halve the sample."""
        act = base(mask)
        for row, options in enumerate(legal_options(mask, ability)):
            act[row, ability] = _pick_active(options)
        return act

    def turtle(mask, tick, fb=None):
        """Hold guard. The control for 'does guarding reduce damage taken'."""
        act = base(mask)
        for row, options in enumerate(legal_options(mask, guard)):
            act[row, guard] = _pick_active(options)
        return act

    def charger(mask, tick, fb=None):
        """Walk forward continuously; never attacks."""
        act = base(mask)
        for row, options in enumerate(legal_options(mask, move)):
            act[row, move] = _pick_active(_in_span(options, WALK_SPAN))
        return act

    def kiter(mask, tick, fb=None):
        """Walk away continuously. Tests whether the NPC chases."""
        act = base(mask)
        back = ARSENAL_POLICY_COMPASS_DIRECTIONS // 2
        for row, options in enumerate(legal_options(mask, move)):
            # direction matters here
            act[row, move] = _pick(_in_span(options, WALK_SPAN), back)
        return act

    def spinner(mask, tick, fb=None):
        """Turn continuously. Tests whether facing gates the NPC's attacks."""
        act = base(mask)
        top = HEAD_SIZES[yaw] - 1
        for row, options in enumerate(legal_options(mask, yaw)):
            act[row, yaw] = _pick(options, top, HEAD_SIZES[yaw] // 2)
        return act

    def dodger(mask, tick, fb=None):
        """Dodge on a fixed cadence. Not reactive -- a reactive dodge needs the
        observation, and this file deliberately probes response, not perception.
        """
        act = base(mask)
        if tick % 10 == 0:
            for row, options in enumerate(legal_options(mask, dodge)):
                act[row, dodge] = _pick_active(_in_span(options, DODGE_SPAN))
        return act

    def hopper(mask, tick, fb=None):
        """Jump constantly. Airborne actors get no horizontal translation, so
        this is the control that separates 'cannot move' from 'not moving'."""
        act = base(mask)
        for row, options in enumerate(legal_options(mask, jump)):
            act[row, jump] = _pick_active(options)
        return act


    def greedy(mask, tick, fb=None):
        """Closed loop: attack only inside reach, otherwise close the gap.

        The open-loop `aggressor` fires whenever an ability is legal, which
        conflates "the ability was available" with "the ability could land".
        This one needs the environment to actually *report* distance, so a flat
        row here is a perception finding, not a behaviour one.
        """
        act = base(mask)
        reach = 2.6
        distance = None if fb is None else fb.distance
        if distance is None or distance <= reach:
            for row, options in enumerate(legal_options(mask, ability)):
                act[row, ability] = _pick_active(options)
        else:
            for row, options in enumerate(legal_options(mask, move)):
                act[row, move] = _pick_active(_in_span(options, WALK_SPAN))
        return act

    def skirmisher(mask, tick, fb=None):
        """Attack while healthy, disengage when hurt.

        The only probe that can let health regenerate, because it is the only
        one that stops taking damage on purpose while still having taken some.
        """
        act = base(mask)
        if fb is not None and fb.agent_hurt:
            back = ARSENAL_POLICY_COMPASS_DIRECTIONS // 2
            for row, options in enumerate(legal_options(mask, move)):
                act[row, move] = _pick(_in_span(options, WALK_SPAN), back)
        else:
            for row, options in enumerate(legal_options(mask, ability)):
                act[row, ability] = _pick_active(options)
        return act

    table = {"idle": idle, "aggressor": aggressor, "turtle": turtle,
             "charger": charger, "kiter": kiter, "spinner": spinner,
             "dodger": dodger, "hopper": hopper,
             "greedy": greedy, "skirmisher": skirmisher}
    if name not in table:
        raise KeyError(f"unknown scenario policy {name!r}; have {sorted(table)}")
    return table[name]


SCENARIOS = ("idle", "aggressor", "turtle", "charger", "kiter", "spinner",
             "dodger", "hopper", "greedy", "skirmisher")


@dataclass
class Result:
    name: str
    ticks: int
    rows: int
    damage_dealt: float
    damage_taken: float
    reward: float
    action_valid_rate: float
    target_visible_rate: float
    attack_requested: int
    attack_accepted: int
    distance_start: float
    distance_min: float
    distance_end: float
    target_attacking_rate: float
    dones: int
    #: Fraction of attempts where the scripted behaviour was actually legal. A
    #: row with a low value is evidence about legality, not about behaviour.
    intent_rate: float
    extra: dict = field(default_factory=dict)


def run(name: str, *, ticks: int = 150, batch: int = 4, seed: int = 7,
        loadout: str = "iron_sword", opponent: str = "iron_sword") -> Result:
    """Run one scripted behaviour and report what the environment did back."""

    import jax
    import jax.numpy as jnp
    from hytalegym.jax.combat.arsenal import (
        arsenal_runtime_config,
        hytale_0_5_7_loadouts,
        open_flat_arsenal_world_capabilities,
    )
    from hytalegym.jax.combat.types import (
        AGENT_ENTITY,
        TARGET_ENTITY,
        default_combat_params,
    )
    from hytalegym.jax.training.arsenal import make_arsenal_ppo_environment

    names = head_names()
    policy = make_policy(name, names)

    params = default_combat_params(microticks=1, target_active=True)
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(
        [loadout] * batch, target_profiles=[opponent] * batch))
    env = make_arsenal_ppo_environment(
        params, runtime,
        world_capability_provider=open_flat_arsenal_world_capabilities)
    step = jax.jit(env.step_detailed)

    state, obs, mask = env.reset(jax.random.split(jax.random.key(seed), batch))
    reset_intent()

    def health(s, e):
        return np.asarray(s.runtime.combat.health[:, e])

    agent0, target0 = health(state, AGENT_ENTITY), health(state, TARGET_ENTITY)

    feedback = None
    valid = visible = attacking = 0.0
    requested = accepted = dones = 0
    reward_total = 0.0
    distances = []

    for tick in range(ticks):
        act = jnp.asarray(policy(np.asarray(mask), tick, feedback))
        keys = jax.random.split(jax.random.key(9000 + tick), batch)
        state, obs, reward, done, mask, info = step(state, obs, act, keys)
        ci = info.combat_info
        valid += float(np.asarray(info.action_valid).astype(bool).mean())
        visible += float(np.asarray(ci.target_visible).astype(bool).mean())
        requested += int(np.asarray(ci.attack_requested).astype(bool).sum())
        accepted += int(np.asarray(ci.attack_accepted).astype(bool).sum())
        attacking += float((np.asarray(ci.target_attack_phase) != 0).mean())
        distances.append(float(np.asarray(ci.target_distance).mean()))
        reward_total += float(np.asarray(reward).sum())
        dones += int(np.asarray(done).astype(bool).sum())
        now_agent = health(state, AGENT_ENTITY)
        feedback = Feedback(
            distance=float(np.asarray(ci.target_distance).mean()),
            agent_health=float(now_agent.mean()),
            agent_hurt=bool((agent0 - now_agent).mean() > 0.0),
            target_health=float(np.asarray(ci.target_health).mean()),
        )

    agent1, target1 = health(state, AGENT_ENTITY), health(state, TARGET_ENTITY)
    return Result(
        name=name, ticks=ticks, rows=batch,
        damage_dealt=float((target0 - target1).sum()),
        damage_taken=float((agent0 - agent1).sum()),
        reward=reward_total,
        action_valid_rate=valid / ticks,
        target_visible_rate=visible / ticks,
        attack_requested=requested, attack_accepted=accepted,
        distance_start=distances[0], distance_min=min(distances),
        distance_end=distances[-1],
        target_attacking_rate=attacking / ticks,
        dones=dones,
        intent_rate=(INTENT["granted"][0] / INTENT["wanted"][0]
                     if INTENT["wanted"][0] else float("nan")),
    )
