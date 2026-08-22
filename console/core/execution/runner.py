"""Parameterised rollouts for the debug console.

One function, :func:`run`, turns a plain dict of knobs into a trajectory the
browser can draw. Everything the console exposes as a control maps to one
argument here, so the HTTP layer stays a thin translation and this module can
be driven straight from Python or a test.

Run from the repository root so ``adk`` and ``hytalegym`` both import.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings
from hashlib import sha256
from collections import OrderedDict
from math import isfinite
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
for _entry in (_ROOT, _ROOT / "HytaleRL" / "hytalegym"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from adk import AgentKit, AgentSpec  # noqa: E402
from adk.policy.fields import GROUP_FEATURES  # noqa: E402
from console.core.execution import runners as _runners  # noqa: E402
from adk.probes import strike_when_ready, uniform_legal  # noqa: E402
from adk.probes.policies import (  # noqa: E402
    ACTION_LOGIT_WIDTH,
    HEAD_SPANS,
    PREFERENCE_STRENGTH,
    head_span,
)
from adk.runtime.env_adapter import legal_mask, sample_actions  # noqa: E402
from adk.runtime.scene_builder import (  # noqa: E402
    COMBAT_PARAMETER_NAMES,
    defensive_providers,
    list_loadouts,
)
from hytalegym.jax.combat.types import MAX_MICROTICKS  # noqa: E402
from adk.runtime.scenes import OPEN_FLAT_CONTROL_SCENE  # noqa: E402
from adk.scenarios import shaping  # noqa: E402
from adk.scenarios.minigames import REGISTRY as MINIGAMES  # noqa: E402
from adk.environments.synthetic import PRESETS as SYNTHETIC  # noqa: E402

ENVIRONMENT = 0
BATCH = 2

#: The environment advances 30 ticks per simulated second. Every duration the
#: console reports in milliseconds is derived from this, so a decision period of
#: 1 tick is a 33 ms reaction loop -- far faster than a person can act.
TICKS_PER_SECOND = 30

_COMBAT = list(GROUP_FEATURES["combat_f32"])
_VISIBLE = _COMBAT.index("target_visible")
_OBSERVED = _COMBAT.index("visible_target_planar_distance")
_EXECUTING = _COMBAT.index("agent_attack_executing")
TARGET_PHASES = ("idle", "windup", "sweep", "recovery", "cooldown")
_PHASES = [_COMBAT.index(f"target_phase_{n}") for n in TARGET_PHASES]

#: Policies the console can drive. Each is a factory taking (slot, period).
POLICY_KINDS = (
    "uniform_legal",
    "idle",
    "hold_ability",
    "pulse_ability",
    "strike_when_ready",
)


#: Worlds the console can build.
#:
#: ``open_flat`` is the permissive control the Gym's own readiness gates use --
#: every world query is pinned true, so terrain cannot matter and there is
#: nothing to draw. ``region`` binds real captured geometry, which is the only
#: world where the map shows actual ground.
#:
#: The synthetic arenas are appended from the registry rather than listed, so a
#: new preset cannot be added without the console offering it. They are
#: capability-level: terrain is a function of position answering world queries,
#: not real cells.
#:
#: **Settled 2026-08-09 -- they do reach the observation** (`adk/tests/
#: test_synthetic.py`, 7 passed). This comment previously said reachability was
#: unverified because "the only probe on disk reports 0 of 8271 columns
#: differing from `open_flat`". That 0 is real but is a *spawn-only* artifact:
#: every preset carries a 0.5-unit phase margin, so the shared spawn x=0.500
#: sits outside its bands by construction and a probe that never moves the agent
#: must read 0. With the agent inside a band, measured columns differing:
#:
#:     parkour   4   drop support + drop height, and the two dodge-corridor bits
#:     lava      3   in_fluid, feet_submerged, eyes_submerged
#:     islands   6   both, minus eyes_submerged (it ships `hazard_deep=False`)
#:
#: Caveat that survives, and it is a real one: `fall_depth` reaches the policy
#: as `min(depth / 3.0, 1.0)` and therefore **saturates at 3.0**. PARKOUR (12.0),
#: ISLANDS (10.0) and the `SyntheticWorld` default (8.0) are all above it, so as
#: shipped the depth knob is a boolean and the presets are indistinguishable on
#: it. Still do not read any of these as terrain *fidelity* -- they compose over
#: `open_flat`, so every field a preset does not override reads permissive.
WORLDS = ("open_flat", "region") + tuple(sorted(SYNTHETIC))


@dataclass
class RunSpec:
    """Every knob the console exposes. Defaults reproduce a live duel."""

    loadout: str = "iron_sword"
    opponent: str | None = "iron_sword"
    armed: bool = True
    perceive: bool = True
    #: ``open_flat`` or ``region``. Region is the only world with terrain to
    #: draw, and it is not free: measured on a 48-tick rollout, **56 s flat
    #: against 149 s region**. The scene build itself is only ~10 s of that
    #: (39.5 s against 29.6 s) -- the rest is the collector compiling against a
    #: bound world runtime, so a longer rollout does not cost proportionally
    #: more.
    world: str = "open_flat"
    #: Region only: ``"<region_seed>:<component>"`` from
    #: :mod:`worlds.zones`. A zone is a strongly-connected set of
    #: standable nodes, so the agent cannot spawn in a one-way pocket it can
    #: never leave -- the largest such pocket is only 35.7% of a region.
    #: ``None`` takes the library's default spawn.
    zone: str | None = None
    #: Engine ticks advanced per policy decision, 1-4.
    #:
    #: `combat/env.py`: "Movement intent persists while ``lax.scan`` advances up
    #: to four engine microticks." The scan is bounded at ``MAX_MICROTICKS = 4``
    #: and *masked* (``can_tick = microtick_index < params.microticks``), which
    #: is how one compiled scan serves the whole 1-4 range without recompiling.
    #:
    #: So this is the only real tick-rate knob in the stack -- nothing else
    #: throttles the simulation, and ``TICKS_PER_SECOND`` is a display constant.
    #: Raising it advances more *game* time per env step; the cost is that the
    #: agent acts once per N ticks instead of every tick (33 ms -> 133 ms at 4,
    #: which is closer to human reaction latency, not further from it).
    #:
    #: Must be set here rather than through `parameters`: the scene and the
    #: `AgentSpec` both carry it and `build()` rejects a mismatch.
    microticks: int = 1
    policy: str = "pulse_ability"
    slot: int = 1
    period: int = 12
    ticks: int = 256
    #: Cap on one EPISODE's length in engine ticks (30 ticks = 1 second), or
    #: `None` for the environment's own rule.
    #:
    #: Distinct from `ticks`, which is how long a *rollout* runs: `ticks` is how
    #: much you watch, this is when the world starts over. The environment ends
    #: an episode only on death or a world error, so without a cap an episode
    #: has no upper bound -- measured, a 256-step update completed 0-3 episodes
    #: and `mean_episode_return` reported 0.000 on most of them because it had
    #: nothing to average.
    #:
    #: Published as `truncated`, never as `terminated`, so a learner can tell a
    #: clock expiry from a death and bootstrap accordingly.
    episode_ticks: int | None = 900
    seed: int = 0
    #: Ticks between fresh decisions; the chosen action is HELD in between.
    #: 1 = act every tick (33 ms -- superhuman, fine for training where both
    #: sides run at the same rate, wrong for an inference-time human
    #: comparison). 6-8 ticks is roughly human reaction latency at 30 TPS.
    decision_period: int = 1
    agent_max_health: float = 105.0
    target_max_health: float = 61.0
    target_active: bool = True
    #: A named objective from `adk.scenarios.tasks` — reweights the four native
    #: reward terms. `baseline` leaves the ruleset alone. This changes what the
    #: agent is being *paid* for, so two runs with different tasks are not two
    #: data points on one experiment.
    task: str = "baseline"
    #: Which opponent ability policy drives the target. `None` keeps the Gym's
    #: shipped `first_legal` — which, measured, is the only one of four that
    #: actually lands damage: `highest_slot` and `random_slot` get abilities
    #: *accepted* 16 and 36 times and deal **0.0**. Acceptance is not damage.
    opponent_policy: str | None = None
    #: A named minigame from `adk.scenarios.minigames`, added on top of the
    #: native reward. Where `task` reweights the four reward terms the
    #: environment already computes, this pays for something they cannot express
    #: at all -- distance closed, a route followed, a hit that landed, a guard
    #: that met an actual attack.
    #:
    #: Additive, so the native terms stay intact underneath and a shaped run is
    #: still comparable to an unshaped one on those alone. Deliberately NOT part
    #: of `_scene_digest`: shaping changes what the agent is paid for, not what
    #: it observes or can do, so a checkpoint moves between shaped and unshaped
    #: handles. It IS part of the collector key, which keys on the whole spec.
    minigame: str | None = None
    #: Scales the minigame's own weight. 1.0 leaves it as the game declares it.
    minigame_weight: float = 1.0
    #: Constructor arguments for the minigame -- `goal` for `reach`, `route` for
    #: `checkpoint`, per-game rates for the rest. A game needing one and not
    #: given it is a 400, not a rollout that pays a silent constant.
    minigame_options: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        names = set(list_loadouts())
        if self.loadout not in names:
            raise ValueError(f"unknown loadout {self.loadout!r}")
        if self.opponent not in (None, "", *names):
            raise ValueError(f"unknown opponent {self.opponent!r}")
        if self.policy not in POLICY_KINDS:
            raise ValueError(f"unknown policy {self.policy!r}")
        if self.world not in WORLDS:
            raise ValueError(f"unknown world {self.world!r}; have {list(WORLDS)}")
        if self.zone and self.world != "region":
            raise ValueError("a zone only means something with world='region'")
        if self.zone and self.zone != ROTATE:
            parse_zone(self.zone)
        if not 1 <= self.ticks <= 2048:
            raise ValueError("ticks must be between 1 and 2048")
        if not 0 <= self.slot <= 16:
            raise ValueError("slot must be between 0 and 16")
        if not 1 <= self.period <= 512:
            raise ValueError("period must be between 1 and 512")
        if not 1 <= self.decision_period <= 128:
            raise ValueError("decision_period must be between 1 and 128")
        if self.task not in task_names():
            raise ValueError(
                f"unknown task {self.task!r}; have {sorted(task_names())}")
        if self.opponent_policy and self.opponent_policy not in OPPONENT_POLICIES:
            raise ValueError(
                f"unknown opponent policy {self.opponent_policy!r}; "
                f"have {sorted(OPPONENT_POLICIES)}")
        if not 1 <= self.microticks <= MAX_MICROTICKS:
            raise ValueError(
                f"microticks must be between 1 and {MAX_MICROTICKS}, "
                f"got {self.microticks}")
        if self.minigame is not None and self.minigame not in MINIGAMES:
            raise ValueError(
                f"unknown minigame {self.minigame!r}; have {sorted(MINIGAMES)}")
        if self.minigame is None and self.minigame_options:
            raise ValueError(
                "minigame_options were given without a minigame; they would be "
                "silently ignored")
        unknown = sorted(set(self.parameters) - set(COMBAT_PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"unknown combat parameter(s): {unknown}")
        if "microticks" in self.parameters:
            # It is a CombatParams field, so it passes the check above and then
            # fails much later inside the scene builder. Say so here instead.
            raise ValueError(
                "set microticks as a top-level knob, not through parameters -- "
                "the scene and the AgentSpec both carry it and a mismatch is "
                "rejected at build")


#: Opponent ability policies, by name. The shipped `first_legal` is the default
#: and is `None` here because it is what the Gym binds when nothing is passed.
#:
#: **Acceptance is not damage.** Swept over 300 ticks x batch 4 at standoff 2.0:
#: `first_legal` dealt 120.0, `patient` 110.0, and `random_slot` / `highest_slot`
#: dealt **0.0** while getting 36 and 16 abilities accepted. On `iron_sword`
#: slot 0 is the damaging ability and higher slots are accepted but harmless, so
#: an opponent sweep that gates on acceptance measures nothing.
OPPONENT_POLICIES = ("first_legal", "patient", "highest_slot", "random_slot",
                     "inert")


def task_names() -> tuple[str, ...]:
    from adk.scenarios import tasks

    return tuple(sorted(tasks.TASKS))


def minigame_names() -> tuple[str, ...]:
    from adk.scenarios import minigames

    return tuple(sorted(minigames.REGISTRY))


def minigame_details() -> list[dict[str, Any]]:
    """Each game with what it teaches, what would show it worked, and whether
    it can be launched from the picker at all.

    `needs_options` is read off the constructor rather than hand-listed:
    `reach` takes a goal and `checkpoint` takes a route, and neither has a
    sensible default -- an invented coordinate can sit inside terrain, and the
    agent is then paid for walking into a wall. Offering them as one-click
    options would mean a picker entry that always 400s, so the UI says so
    instead of finding out.
    """

    import inspect

    out = []
    for name in minigame_names():
        game = MINIGAMES[name]
        try:
            required = [
                p.name for p in inspect.signature(game.build).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            required = []
        out.append({
            "name": name,
            "teaches": getattr(game, "teaches", ""),
            "tell": getattr(game, "tell", ""),
            "requires": list(getattr(game, "requires", ())),
            "needs_options": required,
        })
    return out


def _opponent_provider(name: str | None):
    """Resolve an opponent policy name to the provider the scene binds."""

    if not name or name == "first_legal":
        return None                       # the Gym's own default
    if name == "inert":
        from hytalegym.jax.combat.opponents.runtime.policy import (
            inert_opponent_ability_slots,
        )
        return inert_opponent_ability_slots
    from adk.scenarios import npcs

    return getattr(npcs, name)


def parse_zone(value: str) -> tuple[int, int]:
    """``"974971448:42"`` -> ``(974971448, 42)``."""

    try:
        seed, component = value.split(":", 1)
        return int(seed), int(component)
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"zone must look like '<region_seed>:<component>', got {value!r}"
        ) from exc


def _region_bindings(spec: RunSpec, envs: int) -> tuple[dict[str, Any],
                                                        dict[str, Any],
                                                        dict[str, Any]]:
    """Providers and rewritten spawn params for a Region scene.

    Built through `AgentKit` rather than `region_scene` so that **one** collector
    path serves both worlds. Two builders would drift, and the seams are read
    from `REGION_PROVIDER_SEAMS` so the two routes cannot silently bind
    different sets.

    Targeting a zone takes two keys: `selection_key` chooses which of the 16
    train regions loads, `node_seed` chooses the node inside it. Passing only a
    node seed applies it to the *default* region, which lands somewhere
    unrelated to the zone asked for.
    """

    from worlds.zones import seed_for, selection_for, zones as zone_index
    from worlds.region import load_region, region_providers

    kwargs: dict[str, Any] = {}
    wanted = None
    if spec.zone:
        region_seed, component = parse_zone(spec.zone)
        wanted = next(
            (z for z in zone_index(region_seed) if z.component == component), None)
        if wanted is None:
            raise ValueError(
                f"region {region_seed} has no navigable zone {component}")
        selection_key = selection_for(region_seed)
        if selection_key is None:
            raise ValueError(
                f"region {region_seed} is not in the train split, so no "
                "selection_key can reach it")
        node_seed = seed_for(wanted)
        if node_seed is None:
            raise ValueError(
                f"no node_seed lands in zone {component} of region {region_seed}")
        kwargs = {"selection_key": selection_key, "node_seed": node_seed}

    loaded = load_region(
        weapons=(spec.loadout,) * envs, native_evidence=False, **kwargs)

    # Only the fields the fixture actually rewrote. Measured: agent_spawn,
    # floor_y and target_offset, and nothing else.
    from hytalegym.jax.combat import default_combat_params

    base = default_combat_params(microticks=1, target_active=spec.target_active)
    overrides: dict[str, Any] = {}
    for name in base._fields:
        before, after = getattr(base, name), getattr(loaded.params, name)
        if jnp.asarray(before).shape != jnp.asarray(after).shape or not bool(
            jnp.all(jnp.asarray(before) == jnp.asarray(after))
        ):
            overrides[name] = after

    provenance = {
        "region_artifact": loaded.fixture.metadata.get("artifact_seed"),
        "reach": loaded.maximum_distance,
        "evidence": loaded.evidence,
        "zone": spec.zone,
        **kwargs,
    }
    return region_providers(loaded), overrides, provenance


def _pulse(slot: int, period: int):
    offset, _ = head_span("ability_none_plus_slots")

    def policy(step, actor_input, key):
        mask = legal_mask(actor_input)
        choice = jnp.where(jnp.mod(step, period) == 0, slot, 0)
        bias = (
            jax.nn.one_hot(offset + choice, ACTION_LOGIT_WIDTH, dtype=jnp.float32)
            * PREFERENCE_STRENGTH
        )
        return step + 1, sample_actions(jnp.broadcast_to(bias, mask.shape), mask, key)

    policy.initial_carry = jnp.int32(0)
    return policy


def _hold(slot: int):
    offset, _ = head_span("ability_none_plus_slots")

    def policy(actor_input, key):
        mask = legal_mask(actor_input)
        bias = (
            jax.nn.one_hot(offset + slot, ACTION_LOGIT_WIDTH, dtype=jnp.float32)
            * PREFERENCE_STRENGTH
        )
        return sample_actions(jnp.broadcast_to(bias, mask.shape), mask, key)

    return policy


def _idle(actor_input, key):
    mask = legal_mask(actor_input)
    zero = jnp.zeros(mask.shape, dtype=jnp.float32)
    return sample_actions(zero, mask, key)


def _takes_carry(policy) -> bool:
    """Does this policy take ``(carry, actor_input, key)`` or ``(actor_input, key)``?

    Keying on an ``initial_carry`` attribute is NOT enough, and assuming so was
    the same bug twice. Both `uniform_legal` and `strike_when_ready()` take a
    carry, pass it straight back out untouched, and set no attribute -- so an
    attribute test calls them with two arguments and they raise
    ``missing 1 required positional argument`` from inside the scan.

    Arity is the honest test: the attribute says "I keep state", which is a
    different question from "I accept a carry slot".
    """

    import inspect

    if hasattr(policy, "initial_carry"):
        return True
    try:
        return len(inspect.signature(policy).parameters) >= 3
    except (TypeError, ValueError):
        # An un-introspectable callable is assumed to match the collector's own
        # three-argument form, which is what everything in `adk.probes` uses.
        return True


def _throttle(policy, period: int, heads: int):
    """Emit a fresh decision every ``period`` ticks and HOLD it in between.

    The environment steps at 30 TPS. A policy that samples every tick is acting
    on a 33 ms loop, which no person can match -- so an inference-time
    comparison against a human, or any latency-sensitive evaluation, has to
    throttle the decision rate rather than the simulation rate.

    In *training* this is usually left at 1: both fighters run at the same rate,
    so the asymmetry that matters at inference does not exist.

    Wraps any policy, with or without its own carry. The inner policy is
    evaluated every tick and its result discarded when held, because selecting
    with ``jnp.where`` keeps the trace shape constant -- branching on a traced
    step counter would not.
    """

    inner_carry = getattr(policy, "initial_carry", None)
    stateful = _takes_carry(policy)

    def wrapped(carry, actor_input, key):
        step, held, inner = carry
        if stateful:
            inner_next, fresh = policy(inner, actor_input, key)
        else:
            inner_next, fresh = inner, policy(actor_input, key)
        take = jnp.mod(step, period) == 0
        action = jnp.where(take, fresh, held)
        return (step + 1, action, inner_next), action

    # A carry-taking policy that declares no initial value gets a scalar zero:
    # `uniform_legal` and `strike_when_ready()` both ignore the value entirely
    # and return it unchanged, so the slot only has to keep a stable shape.
    zero_inner = inner_carry if inner_carry is not None else jnp.int32(0)
    wrapped.initial_carry = (
        jnp.int32(0),
        jnp.zeros((BATCH, heads), dtype=jnp.int32),
        zero_inner,
    )
    return wrapped


def _carried(policy):
    """Give a stateless policy the carry the collector always passes.

    `collect_transitions` calls `policy_fn(carry, actor_input, key)` -- always
    three arguments. `_idle` and `_hold` were written with two, so selecting
    either raised `takes 2 positional arguments but 3 were given` *inside the
    scan*, i.e. only once the rollout was already running. Both were offered in
    POLICY_KINDS the whole time; the default happens to carry state, which is
    why this survived.
    """

    if _takes_carry(policy):
        return policy

    def wrapped(carry, actor_input, key):
        return carry, policy(actor_input, key)

    wrapped.initial_carry = jnp.int32(0)
    return wrapped


def _build_policy(spec: RunSpec):
    if spec.policy == "uniform_legal":
        policy = uniform_legal
    elif spec.policy == "idle":
        policy = _idle
    elif spec.policy == "strike_when_ready":
        policy = strike_when_ready()
    elif spec.policy == "hold_ability":
        policy = _hold(spec.slot)
    else:
        policy = _pulse(spec.slot, spec.period)

    if spec.decision_period > 1:
        policy = _throttle(policy, spec.decision_period, len(HEAD_SPANS))
    return _carried(policy)


#: Arsenal failure bits, from ``arsenal/contract.py:146-153``. These are the
#: environment saying *why* something did not happen -- the console's single
#: most useful debugging channel, and previously not surfaced at all.
ARSENAL_FAILURES = (
    (1 << 0, "ability_overflow", "more abilities active than the slot budget"),
    (1 << 1, "event_overflow", "more events than the event budget"),
    (1 << 2, "projectile_overflow", "more projectiles than the projectile budget"),
    (1 << 3, "area_overflow", "more area effects than the budget"),
    (1 << 4, "invalid_loadout", "the loadout is not internally consistent"),
    (1 << 5, "invalid_command", "an action was refused as malformed"),
    (1 << 6, "invalid_state", "the arsenal reached a state it cannot represent"),
    (1 << 7, "unsupported_world", "the world lacks something the ability needs"),
)


def _record(transition):
    observation = transition.next_actor_input.legal_observation
    runtime = transition.next_state.environment.runtime
    combat = runtime.combat
    arsenal = runtime.arsenal
    mechanics = runtime.mechanics
    combat_f32 = observation.base.combat_f32
    return {
        "active_slot": arsenal.active_ability_slot,
        "projectiles": transition.info.arsenal_info.projectile_count,
        "arsenal_failures": arsenal.failure_bits,
        "mechanics_failures": mechanics.failure_bits,
        "stamina": mechanics.resources[..., 0],
        "guard_active": mechanics.guard_active,
        "position": combat.position,
        "yaw": combat.yaw,
        "health": combat.health,
        "grounded": combat.agent_grounded,
        "visible": combat_f32[..., _VISIBLE],
        "observed_distance": combat_f32[..., _OBSERVED],
        "attack_executing": combat_f32[..., _EXECUTING],
        "target_phase": combat_f32[..., jnp.asarray(_PHASES)],
        "requested": transition.info.arsenal_info.ability_requested,
        "accepted": transition.info.arsenal_info.ability_accepted,
        "damage_dealt": transition.info.arsenal_info.damage_dealt,
        "reward": transition.reward,
        "done": transition.done,
        "action": transition.action_factors,
    }


def _stage_option_fields(names: Any) -> list[dict[str, Any]]:
    """`{name, default, type}` for each stage option, read from the contract.

    `custom.stage_options` is only a list of NAMES, which leaves a client no
    defaults and no types. The defaults already exist as dataclass fields on the
    stage config, so derive them rather than restating them here, where they
    would go stale the first time the contract moves.

    Additive: `stage_options` keeps its old shape for existing readers.
    """

    import dataclasses

    try:
        from arena.training.contracts.pursuit import PursuitStageConfig
    except Exception:  # noqa: BLE001 - a console without arena still lists stages
        return []
    wanted = set(names or ())
    fields = []
    for option in dataclasses.fields(PursuitStageConfig):
        if wanted and option.name not in wanted:
            continue
        if option.default is dataclasses.MISSING:
            continue
        if not isinstance(option.default, (int, float, bool)):
            continue
        fields.append({
            "name": option.name,
            "default": option.default,
            "type": "bool" if isinstance(option.default, bool) else "number",
        })
    return sorted(fields, key=lambda item: item["name"])


def training_workflows() -> list[dict[str, Any]]:
    """Composable stage contracts shared by Train and Compute clients."""

    from arena.training.contracts.catalog import training_contracts
    from arena.training.runs.pursuit_run import pursuit_launch_options

    catalog = training_contracts()
    pursuit = pursuit_launch_options()
    pursuit.setdefault("custom", {})["stage_option_fields"] = _stage_option_fields(
        pursuit.get("custom", {}).get("stage_options")
    )
    workflows = [pursuit]
    for contract in catalog["contracts"]:
        collector = contract["collector"]
        if collector["stage"] == "pursuit_tracking":
            workflows[0]["contract"] = contract
            workflows[0]["launchable"] = True
            continue
        maximum_ticks = (
            contract["defaults"].get("horizon", {}).get("maximum_ticks", 512)
        )
        workflows.append({
            "schema": "console-training-contract-workflow-v1",
            "stage": contract["contract_id"],
            "title": contract["title"],
            "strategy": "agent_selected_jax_learner",
            "default_preset": "default",
            "presets": {
                "default": {
                    "batch": 256,
                    "updates": 64,
                    "rollout_steps": 128,
                    "evaluation_steps": maximum_ticks,
                    "archive_lanes": 8,
                }
            },
            "custom": {
                "rule": "contract defaults are editable before JIT binding",
                "fixed": {},
                "stage_options": sorted(contract["defaults"]),
                "stage_option_fields": _stage_option_fields(contract["defaults"]),
            },
            "contract": contract,
            "launchable": collector["launchable"],
            "blockers": collector["blockers"],
            "compute": {
                "schema": "console-training-compute-composition-v1",
                "composition": "resolve_static_features_before_one_jax_trace",
                "controls": [],
                "features": contract["action_heads"],
                "extension_rule": "collector binding supplies environment-specific evidence",
            },
        })
    workflows.append(
        {
            "schema": "console-legacy-combat-launch-v1",
            "stage": "combat",
            "strategy": "recurrent_ppo",
            "default_preset": "custom",
            "presets": {"custom": {}},
            "custom": {
                "rule": "environment, task, minigame, reward and PPO fields are explicit"
            },
            "compute": {
                "schema": "console-training-compute-composition-v1",
                "composition": "resolve_static_features_before_one_jax_trace",
                "controls": [
                    {
                        "key": "num_envs",
                        "label": "parallel environments",
                        "kind": "integer",
                        "minimum": 1,
                        "maximum": 512,
                    },
                    {
                        "key": "rollout_steps",
                        "label": "ticks per update",
                        "kind": "integer",
                        "minimum": 1,
                        "maximum": 2048,
                    },
                ],
                "features": [],
                "extension_rule": "environment and strategy contracts add static features before JIT",
            },
        }
    )
    return workflows


def options() -> dict[str, Any]:
    """Everything the UI needs to populate its controls."""

    from arena.training.contracts.catalog import training_contracts

    return {
        "loadouts": list(list_loadouts()),
        "policies": list(POLICY_KINDS),
        "worlds": list(WORLDS),
        "zones": zone_options(),
        "tasks": list(task_names()),
        "minigames": list(minigame_names()),
        "minigame_details": minigame_details(),
        "opponent_policies": list(OPPONENT_POLICIES),
        "heads": [{"name": n, "size": s} for n, (_o, s) in HEAD_SPANS.items()],
        "target_phases": list(TARGET_PHASES),
        "combat_parameters": list(COMBAT_PARAMETER_NAMES),
        "reward_terms": reward_defaults(),
        # Probed once per process. WSL is skipped here: starting a stopped
        # distribution takes tens of seconds and this is the page-load path.
        # The Status tab pays for the full probe.
        "runners": _runners.scan(probe_wsl=False),
        "active_runner": _runners.ACTIVE,
        "training_workflows": training_workflows(),
        "training_contracts": training_contracts(),
        "defaults": asdict(RunSpec()),
    }


#: The environment's entire reward, in four numbers (`combat/types.py:182`):
#:
#:     target_damage * target_damage_reward_scale
#:   + agent_damage  * agent_damage_reward_scale
#:   + completion_reward   when the target dies
#:   + death_reward        when the agent dies
#:
#: They are `CombatParams` fields, so they travel as `parameters` overrides and
#: need no scene change. A named `task` is a preset over exactly these four.
REWARD_TERMS = (
    ("target_damage_reward_scale", "pay per point of damage dealt"),
    ("agent_damage_reward_scale", "pay per point of damage taken (negative)"),
    ("completion_reward", "one-off, when the target dies"),
    ("death_reward", "one-off, when the agent dies (negative)"),
)


def reward_defaults() -> list[dict[str, Any]]:
    """The four reward weights with the values the environment actually uses.

    Read from `default_combat_params` rather than written down here. A constant
    copied into the console drifts the moment the environment retunes, and a UI
    that shows a stale default is worse than one that shows none: the number
    looks authoritative and silently is not.
    """

    try:
        from hytalegym.jax.combat import default_combat_params

        base = default_combat_params(microticks=1, target_active=True)
    except Exception:  # noqa: BLE001 - the picker must never break the console
        return []
    out = []
    for name, meaning in REWARD_TERMS:
        value = getattr(base, name, None)
        if value is None:
            continue
        out.append({"name": name, "meaning": meaning, "default": float(value)})
    return out


#: How many zones to offer per axis. The library holds 182 navigable zones;
#: a dropdown of 182 is a table with extra steps, so the picker offers a few
#: per measured axis. They are drawn with `spread` rather than `select`: the
#: top-N of one ranking are near-duplicates of each other, so the old 3-per-axis
#: list was twelve extremes and no middle. Spanning each axis instead covers far
#: more of the corpus for the same dropdown length.
ZONES_PER_AXIS = 5
#: All five axes `zones.select` measures. `enclosed` -- share of nodes under
#: cover -- was missing, so cave and overhang arenas were in the corpus and
#: unreachable from the console entirely.
ZONE_AXES = ("open", "vertical", "materials", "area", "enclosed")


def zone_options() -> list[dict[str, Any]]:
    """A short, curated zone list for the picker.

    Every entry is a strongly-connected component of a region's traversal
    graph, so an agent spawned in one can reach the whole arena -- no one-way
    drop can strand it. Ordered so the flattest arena comes first, because that
    is the closest real-terrain analogue of the flat control and therefore the
    honest place to start comparing.
    """

    try:
        from worlds.zones import library, selection_for, spread
    except Exception:  # noqa: BLE001 - the picker must never break the console
        return []

    try:
        found = library()
    except Exception:  # noqa: BLE001 - a missing corpus is not a crash
        return []

    # Only the 16 train regions can be selected -- the loader ranks the train
    # split and nothing else is a candidate. Offering a heldout zone would put
    # an option in the menu that fails the moment it is chosen; measured, 3 of
    # the first 10 ranked zones were heldout.
    reachable = {z.seed for z in found if selection_for(z.seed) is not None}
    found = [z for z in found if z.seed in reachable]

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for axis in ZONE_AXES:
        for rank, zone in enumerate(spread(found, by=axis,
                                           count=ZONES_PER_AXIS)):
            key = f"{zone.seed}:{zone.component}"
            if key in seen:
                continue
            seen.add(key)
            span = zone.span
            out.append({
                "id": key,
                "axis": axis,
                # The rank says where on the axis this sits, so two entries
                # under one axis are distinguishable in the menu rather than
                # reading as duplicates.
                "rank": rank,
                "label": (
                    f"{zone.character()} · {axis}"
                    f"{'' if rank == 0 else f' #{rank + 1}'} · "
                    f"{zone.spawnable} spawns · y-span {span[1]:.0f} · "
                    f"{zone.materials} materials"
                ),
                "character": zone.character(),
                "spawnable": zone.spawnable,
                "materials": zone.materials,
                "y_span": round(float(span[1]), 1),
            })
    return out


def _scene_for(spec: RunSpec, envs: int = BATCH):
    """Build the agent handle one configuration describes.

    Extracted so a training job and a rollout construct the *same* scene from
    the same knobs. Two builders would drift, and a training run that trained
    against a subtly different world than the rollout used to inspect it is the
    kind of mismatch nothing downstream can detect.

    ``envs`` is the only thing a training job varies. `BATCH` is two because the
    console *draws* one environment and needs a second only to keep the batch
    axis honest; PPO's own default is 256, and training at two collects 512
    steps per update, which is noise. The scene is the same scene either way, so
    the width deliberately stays out of `_scene_digest` -- what changes is how
    many copies of it run at once.

    A checkpoint still remembers the width it trained at: `_validate_config`
    treats `num_envs` as a static field and compares it against `built.batch`,
    so weights trained at 64 load only into a handle built at 64. That is why
    the value is recorded on the artifact rather than left implicit.
    """

    spec.validate()

    providers: dict[str, Any] = {}
    parameters = dict(spec.parameters)
    region_note: dict[str, Any] = {}

    if spec.world == "region":
        providers, overrides, region_note = _region_bindings(spec, envs)
        # The fixture's spawn wins over anything the caller set, because a
        # Region spawn that is not a traversal node is not standable ground.
        parameters.update(overrides)
    elif spec.world in SYNTHETIC:
        # A synthetic arena composes *over* the permissive control: it starts
        # from `open_flat` and overrides only the handful of fields it models,
        # so combat stays reachable and the difference is attributable to
        # terrain rather than to a denied line of sight.
        providers.update(OPEN_FLAT_CONTROL_SCENE.environment_kwargs)
        providers["world_capability_provider"] = SYNTHETIC[spec.world].provider()
    elif spec.perceive:
        providers.update(OPEN_FLAT_CONTROL_SCENE.environment_kwargs)

    if spec.armed and spec.opponent:
        providers = defensive_providers(providers)

    chosen = _opponent_provider(spec.opponent_policy)
    if chosen is not None and spec.opponent:
        # `opponent_ability_provider` is a declared scene seam, so a custom
        # opponent is a first-class scene property rather than a wrapper.
        providers = {**providers, "opponent_ability_provider": chosen}

    parameters.setdefault("agent_max_health", spec.agent_max_health)
    parameters.setdefault("target_max_health", spec.target_max_health)

    # A task reweights the four native reward terms, which are CombatParams
    # floats -- so it lands as parameter overrides and needs no scene change.
    if spec.task and spec.task != "baseline":
        from adk.scenarios import tasks
        from hytalegym.jax.combat import default_combat_params

        base = default_combat_params(microticks=1,
                                     target_active=spec.target_active)
        shaped = tasks.TASKS[spec.task].params(base)
        for name in base._fields:
            before, after = getattr(base, name), getattr(shaped, name)
            if not bool(jnp.all(jnp.asarray(before) == jnp.asarray(after))):
                parameters.setdefault(name, float(after))

    global _LAST_CACHE_HIT
    digest = _scene_digest(spec, parameters)
    key = (digest, envs)
    with _BUILD_LOCK:
        cached = _SCENE_CACHE.get(key)
        if cached is not None:
            # Same configuration, same handle -- so JAX's own compilation cache
            # hits too and the rollout starts immediately.
            _SCENE_CACHE.move_to_end(key)
            _LAST_CACHE_HIT = True
            return cached, region_note
        _LAST_CACHE_HIT = False

        kit = AgentKit(compilation_cache=os.environ.get("HYTALERL_JAX_CACHE"))
        name = f"console/{digest}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scene = kit.make_scene(
                name,
                loadouts=[spec.loadout] * envs,
                opponents=([spec.opponent] * envs) if spec.opponent else None,
                providers=providers,
                parameters=parameters,
                microticks=spec.microticks,
                target_active=spec.target_active,
            )
        # The same value on both, deliberately: `build()` rejects a scene and an
        # AgentSpec that disagree rather than silently preferring one.
        kit.register(AgentSpec(name=name, scene=scene.name,
                               loadout=spec.loadout,
                               microticks=spec.microticks))
        handle = kit.make(name, num_envs=envs)

        _SCENE_CACHE[key] = handle
        while len(_SCENE_CACHE) > SCENE_CACHE_LIMIT:
            _SCENE_CACHE.popitem(last=False)
        return handle, region_note


#: Built handles, keyed by scene digest and env width. A Region scene costs
#: ~40 s to build and the collector another ~60 s to compile, so re-running the
#: same configuration used to pay both again for nothing -- every rollout was a
#: cold start. JAX caches its own compilations per process, but only if the
#: traced callable is the same object, which a rebuilt handle is not.
#:
#: Deliberately tiny. Each entry pins device buffers for a whole environment,
#: so this trades memory for latency and two configurations in flight is the
#: most that is worth holding.
_SCENE_CACHE: "OrderedDict[tuple[str, int], Any]" = OrderedDict()
#: Jitted collectors, keyed by scene *and* the policy and length they were
#: traced for. Held separately because the same scene serves many rollout
#: lengths, and a longer one must not evict the scene itself.
_COLLECTOR_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
SCENE_CACHE_LIMIT = 2

#: Serialises scene building, collector compilation and rollout execution.
#:
#: FastAPI runs sync route handlers on a threadpool, so two browser tabs -- or a
#: rollout and a training job's scene build -- reach this module at the same
#: time. Two problems, neither of which announces itself: the two caches above
#: are plain `OrderedDict`s mutated check-then-set, and `_LAST_CACHE_HIT` is a
#: module global read back after the build, so a concurrent request silently
#: reports the other one's answer.
#:
#: The heavier reason is memory. Each entry pins device buffers for a whole
#: environment and a Region build peaks well above its steady state; two
#: compiling at once is how this runs out of memory rather than getting faster.
#: `checks.py` and `jobs.py` already refuse to run two at a time -- this is the
#: same rule for the path that costs the most.
#:
#: Reentrant because `run` holds it across its own call to `_scene_for`.
_BUILD_LOCK = threading.RLock()

#: Set by `_scene_for` so a result can say whether it paid the build.
_LAST_CACHE_HIT = False


def _resource_sample() -> dict[str, Any]:
    """Process and device memory right now, for a before/after difference."""

    sample: dict[str, Any] = {"cpu_seconds": time.process_time()}
    try:
        import resource  # noqa: F401 - POSIX only

        sample["rss_mb"] = None
    except ImportError:
        sample["rss_mb"] = None
    try:
        import psutil  # type: ignore

        sample["rss_mb"] = round(psutil.Process().memory_info().rss / 2 ** 20, 1)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    try:
        for device in jax.devices():
            stats = getattr(device, "memory_stats", None)
            if stats:
                raw = stats() or {}
                sample[f"device{device.id}_mb"] = round(
                    int(raw.get("bytes_in_use", 0)) / 2 ** 20, 1)
    except Exception:  # noqa: BLE001 - CPU backend exposes no stats
        pass
    return sample


def _resource_cost(before: dict[str, Any]) -> dict[str, Any]:
    """What changed between the start of a run and now.

    Device memory is a *delta*, so it answers "what did this run add", not
    "what is loaded". On the CPU backend there are no device stats at all and
    the field is simply absent -- reporting 0 MB of VRAM would read as "this
    run used no GPU memory" rather than "there is no GPU".
    """

    after = _resource_sample()
    cost: dict[str, Any] = {
        "cpu_seconds": round(after["cpu_seconds"] - before["cpu_seconds"], 2),
        "scene_cache_hit": _LAST_CACHE_HIT,
    }
    if after.get("rss_mb") is not None:
        cost["rss_mb"] = after["rss_mb"]
        if before.get("rss_mb") is not None:
            cost["rss_delta_mb"] = round(after["rss_mb"] - before["rss_mb"], 1)
    for key, value in after.items():
        if key.startswith("device") and key in before:
            cost[f"{key}_delta"] = round(value - before[key], 1)
            cost[key] = value
    return cost


def cache_state() -> dict[str, Any]:
    """What the scene cache is holding, for the console to show."""

    return {
        "entries": [{"digest": digest, "num_envs": envs}
                    for digest, envs in _SCENE_CACHE],
        "collectors": len(_COLLECTOR_CACHE),
        "limit": SCENE_CACHE_LIMIT,
        "last_was_hit": _LAST_CACHE_HIT,
    }


def clear_scene_cache() -> int:
    """Drop every cached handle and collector. Returns how many were held."""

    held = len(_SCENE_CACHE) + len(_COLLECTOR_CACHE)
    _SCENE_CACHE.clear()
    _COLLECTOR_CACHE.clear()
    return held


def _scene_digest(spec: RunSpec, parameters: dict[str, Any]) -> str:
    """A name that is the same in every process for the same configuration.

    This was `abs(hash(...))` over a tuple of strings. Python salts string
    hashing per process, so the identical configuration was named differently on
    every launch -- measured: the same tuple hashed to 6870068102786984762,
    3618970641710170465 and 3763111922062303777 in three consecutive
    interpreters.

    That was invisible while scenes were per-request and nothing outlived the
    process. Checkpoints changed it: the name goes into the `AgentSpec`, its
    digest is stamped into the checkpoint, and `load_checkpoint` compares it --
    so a run could not reload its own weights in a later process, failing with
    "AgentSpec digest differs" for a configuration that had not changed at all.
    """

    return sha256(json.dumps({
        "loadout": spec.loadout,
        "opponent": spec.opponent,
        "armed": spec.armed,
        "perceive": spec.perceive,
        "target_active": spec.target_active,
        # `world` and `zone` are part of the scene's identity, not of how it is
        # run. Leaving them out gave a flat scene and a Region scene the same
        # name, so a checkpoint trained on real terrain would load into
        # `open_flat` -- where every world query is pinned true -- and the
        # AgentSpec digest check would raise nothing.
        "world": spec.world,
        "zone": spec.zone,
        # Engine ticks per decision changes the dynamics the weights were
        # trained against, so it is part of what the scene IS -- not of how it
        # is run. Two handles differing only here must not share a cache entry.
        "microticks": spec.microticks,
        # A different opponent policy is a different world to fight in, so it
        # must change the scene identity -- otherwise the cache would hand back
        # a handle built against the previous one.
        "opponent_policy": spec.opponent_policy,
        "parameters": parameters,
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


#: `zone` value meaning "pick one from the library, keyed on the seed".
ROTATE = "rotate"


def rotated(spec: RunSpec) -> RunSpec:
    """Resolve `zone="rotate"` to a concrete zone, keyed on the seed.

    One run trains on ONE terrain, and that is a property of the loader rather
    than a choice: `load_region` takes a scalar `selection_key` and a scalar
    `node_seed`, so every row of the batch is a copy of the same zone entered at
    the same spawn node. Variety within a run is not expressible today. Variety
    ACROSS runs is, and the seed is the knob people already sweep -- so
    rotating on it turns a seed sweep into a terrain sweep for free.

    Resolved at launch rather than inside `_scene_for`, because the concrete
    zone has to reach `_scene_digest`: two seeds that rotate onto different
    terrain must not share one cached handle, and an artifact that recorded
    "rotate" could never say afterwards which place the policy trained in.
    """

    if spec.zone != ROTATE:
        return spec
    import dataclasses

    options = zone_options()
    if not options:
        raise ValueError(
            "no zones are available to rotate through; the region library is "
            "missing or nothing in it is in the train split")
    chosen = options[spec.seed % len(options)]
    return dataclasses.replace(spec, zone=chosen["id"])


def _bounded(handle, spec: RunSpec):
    """Cap episode length on `handle`, if the spec asked for one.

    Same placement argument as `_shaped`: applied to the handle the cache
    returned rather than to the cached entry, so a later run with a different
    cap -- or none -- does not inherit this one's.

    Out of `_scene_digest` on purpose. The cap does not change the observation
    or action contract, so a checkpoint trained under one still loads into a
    handle built without it; what it changes is how long a run of the scene
    lasts. The collector key is derived from the whole spec separately, so two
    caps still compile their own collectors.
    """

    if not spec.episode_ticks:
        return handle
    return handle.with_episode_limit(int(spec.episode_ticks))


def _shaped(handle, spec: RunSpec):
    """Attach the spec's minigame to `handle`, if it named one.

    Applied to the handle the cache returned, not to the cached entry itself.
    `with_shaping` is frozen and returns a copy, so the cached handle stays
    unshaped and a later run that names no minigame -- or a different one --
    gets the native reward rather than this one's leftovers.

    Kept out of `_scene_for` deliberately. That function builds and caches a
    *scene*; a reward term is not part of what the scene is, and folding it in
    would either pollute the cache key or hand a shaped handle to a caller that
    asked for a baseline.
    """

    if not spec.minigame:
        return handle
    game = MINIGAMES[spec.minigame]
    try:
        term = shaping.resolve(game, **dict(spec.minigame_options))
    except TypeError as exc:
        # A game needing a constructor argument it was not given. Say which,
        # rather than failing later inside a traced rollout where the message
        # is about tracers.
        raise ValueError(
            f"minigame {spec.minigame!r} needs different options "
            f"({exc}); pass them as minigame_options") from exc
    return handle.with_shaping(shaping.weighted(term, spec.minigame_weight))


def _rollout(spec: RunSpec):
    """Build or reuse the scene and collector, then execute one rollout.

    Split out of `run` only so the whole build-compile-execute sequence sits
    under one `with` rather than being indented into it -- everything below the
    return value is pure post-processing of arrays already on the host and does
    not need to hold the lock.
    """

    # Before `_scene_for`, so the concrete zone reaches the digest and the
    # cache rather than the literal "rotate".
    spec = rotated(spec)
    handle, region_note = _scene_for(spec)
    handle = _bounded(_shaped(handle, spec), spec)

    # The collector is jitted per (scene, policy, length). Rebuilding it hands
    # JAX a new callable, so its own compilation cache misses and the trace is
    # paid again -- which was most of what a "warm" run still cost: 60.3 s cold
    # against 26.2 s with only the scene cached.
    # Key on the WHOLE spec except the seed, which is passed at call time and
    # does not change the trace. Keying on a hand-picked subset is how a
    # different task -- which only shows up as reward parameters -- would have
    # silently reused the previous task's compiled collector.
    collector_key = sha256(json.dumps(
        {k: v for k, v in asdict(spec).items() if k != "seed"},
        sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    collector = _COLLECTOR_CACHE.get(collector_key)
    if collector is None:
        policy = _build_policy(spec)
        kwargs = {}
        if hasattr(policy, "initial_carry"):
            kwargs["initial_carry"] = policy.initial_carry
        collector = handle.compile_collector(policy, _record, spec.ticks, **kwargs)
        _COLLECTOR_CACHE[collector_key] = collector
        while len(_COLLECTOR_CACHE) > SCENE_CACHE_LIMIT:
            _COLLECTOR_CACHE.popitem(last=False)
    else:
        _COLLECTOR_CACHE.move_to_end(collector_key)
    _final, rec = collector(jax.random.key(spec.seed))
    return region_note, rec


def run(spec: RunSpec) -> dict[str, Any]:
    """Roll out one configuration and return a JSON-ready trajectory."""

    spec.validate()
    started = time.perf_counter()
    before = _resource_sample()

    # Held across the build, the compile AND the rollout. Two of these running
    # at once share two mutable caches and one global, and peak device memory
    # is set by how many compile simultaneously rather than by any one of them.
    with _BUILD_LOCK:
        region_note, rec = _rollout(spec)

    e = ENVIRONMENT
    position = rec["position"]
    health = rec["health"]
    agent = position[:, e, 0, :]
    target = position[:, e, 1, :]
    planar = jnp.linalg.norm((target - agent)[:, [0, 2]], axis=-1)
    action = rec["action"]
    action = action[:, e, :] if action.ndim == 3 else action[:, e]

    # Non-finite values are counted per series and emitted as null.
    #
    # Starlette renders responses with `allow_nan=False`, so ONE NaN anywhere in
    # this payload raised "Out of range float values are not JSON compliant" at
    # render time -- an opaque HTTP 500 after the entire rollout had already
    # been paid for, naming neither the column nor the tick. `ints()` was worse
    # and failed earlier still, with `int(nan)` raising ValueError.
    #
    # A NaN here is normally THE finding rather than an inconvenience: a single
    # NaN column disables an NPC permanently, and nothing in the environment
    # raises when it happens. So it is recorded, surfaced in the summary and
    # warned about, and the run still returns something to look at.
    nonfinite: dict[str, int] = {}

    def series(values, digits=4):
        return [None if not isfinite(float(v)) else round(float(v), digits)
                for v in values]

    def ints(values):
        return [None if not isfinite(float(v)) else int(v) for v in values]

    slot = rec["active_slot"][:, e, 0]
    prior = [-1, *[int(v) for v in slot[:-1]]]
    starts = [int(int(v) >= 0 and p < 0) for v, p in zip(slot, prior)]
    accepted = rec["accepted"][:, e, 0]
    requested = rec["requested"][:, e, 0]
    damage = rec["damage_dealt"][:, e]
    reward = rec["reward"][:, e]
    visible = [int(v > 0.5) for v in rec["visible"][:, e]]
    phase = rec["target_phase"][:, e, :]

    trajectory = {
        "agent_x": series(agent[:, 0]),
        "agent_y": series(agent[:, 1]),
        "agent_z": series(agent[:, 2]),
        "target_x": series(target[:, 0]),
        "target_y": series(target[:, 1]),
        "target_z": series(target[:, 2]),
        "agent_yaw": series(rec["yaw"][:, e, 0], 3),
        "target_yaw": series(rec["yaw"][:, e, 1], 3),
        "agent_health": series(health[:, e, 0], 2),
        "target_health": series(health[:, e, 1], 2),
        "true_distance": series(planar, 3),
        "observed_distance": series(rec["observed_distance"][:, e], 3),
        "visible": visible,
        "grounded": ints(rec["grounded"][:, e]),
        "requested": ints(requested),
        "accepted": ints(accepted),
        "attack_executing": [int(v > 0.5) for v in rec["attack_executing"][:, e]],
        "damage_dealt": series(damage),
        "target_phase": [int(row.argmax()) for row in phase],
        "reward": series(reward, 5),
        "done": ints(rec["done"][:, e]),
        "action": [[int(x) for x in row] for row in action],
        "active_slot": ints(slot),
        "ability_start": starts,
        "projectiles": ints(rec["projectiles"][:, e]),
        "stamina": series(rec["stamina"][:, e, 0], 2),
        "guard_active": ints(rec["guard_active"][:, e, 0]),
    }

    # Stop at the end of the first episode -- BEFORE anything is counted.
    #
    # The environment does not auto-reset: after `done` the agent is a corpse at
    # zero health and the flag repeats every tick. Measured once at ~700 of 900
    # ticks spent that way. Worse for the map, positions jump when the episode
    # rolls, which looks exactly like the agent teleporting.
    #
    # This used to run at the very END of this function, after the summary was
    # already computed -- so every headline number (damage, reward_total,
    # accepted, hit_rate, stamina_floor) was aggregated over ticks that were
    # then dropped from the trajectory the browser draws. The panel and the
    # chart under it disagreed, and the summary was the one that was wrong.
    ended = next((i for i, v in enumerate(trajectory["done"]) if v), None)
    episode_ticks = spec.ticks
    if ended is not None and ended + 1 < spec.ticks:
        episode_ticks = ended + 1
        for key, values in trajectory.items():
            if isinstance(values, list) and len(values) == spec.ticks:
                trajectory[key] = values[:episode_ticks]

    for _name, _values in trajectory.items():
        _bad = sum(1 for v in _values if v is None)
        if _bad:
            nonfinite[_name] = _bad

    def finite(name):
        """One series with its nulls removed, for reductions.

        Every reduction below has to go through this: `sum` and `min` raise on
        a None, so an un-guarded NaN would trade the render-time 500 for a
        summary-time one.
        """

        return [v for v in trajectory[name] if v is not None]

    # -- events: the discrete things a debugger actually looks for ------------
    events = []

    def note(tick, kind, detail, severity="info"):
        events.append(
            {"tick": int(tick), "kind": kind, "detail": detail, "severity": severity}
        )

    for i, v in enumerate(trajectory["ability_start"]):
        if v:
            note(i, "ability_start", f"slot {trajectory['active_slot'][i]}")
    for i, v in enumerate(trajectory["damage_dealt"]):
        if v is not None and v > 0:
            note(i, "hit", f"{v:.1f} damage dealt", "good")
    taken = trajectory["agent_health"]
    for i in range(1, len(taken)):
        if taken[i] is None or taken[i - 1] is None:
            continue
        drop = taken[i - 1] - taken[i]
        if drop > 0:
            note(i, "damage_taken", f"-{drop:.1f} hp", "warn")
    for i, v in enumerate(trajectory["done"]):
        if v:
            note(i, "episode_end", "done flag raised", "crit")
            break
    seen_bits = 0
    # Sliced like everything else: a failure bit first raised by the corpse is
    # not a failure of the fight being watched.
    for i, bits in enumerate(ints(rec["arsenal_failures"][:episode_ticks, e])):
        fresh = bits & ~seen_bits
        if fresh:
            for mask, name, why in ARSENAL_FAILURES:
                if fresh & mask:
                    note(i, f"failure:{name}", why, "crit")
            seen_bits |= bits
    events.sort(key=lambda ev: ev["tick"])

    def drop(name):
        """First minus last of a health series, or None if it is all null."""

        values = finite(name)
        return round(values[0] - values[-1], 2) if values else None

    landed = sum(1 for v in finite("damage_dealt") if v > 0)
    accepted_total = sum(finite("accepted"))
    summary = {
        # Read back out of `trajectory`, never off the local `visible` list --
        # the local is the untruncated one and slipped past the first pass of
        # this fix, still counting corpse ticks while every neighbour did not.
        "visible_steps": sum(trajectory["visible"]),
        "requested": sum(finite("requested")),
        "accepted": accepted_total,
        "landed": landed,
        "damage_dealt": drop("target_health"),
        "damage_taken": drop("agent_health"),
        "reward_total": round(sum(finite("reward")), 3),
        # What the agent was PAID for, alongside what it did. `reward_total` is
        # not comparable across these two -- a shaped run and a baseline are
        # different objectives, not two samples of one -- so the reader needs
        # both named next to the number they change.
        "task": spec.task,
        "minigame": spec.minigame,
        "minigame_weight": spec.minigame_weight if spec.minigame else None,
        "shaped": bool(spec.minigame),
        "hit_rate": round(landed / max(1, accepted_total), 4),
        "wall_seconds": round(time.perf_counter() - started, 2),
        # Ticks are the unit the agent lives in. It decides every N ticks and
        # the engine advances in ticks; wall time only says how fast we got
        # through them, and is a pure win when it goes down.
        "engine_ticks": episode_ticks * spec.microticks,
        "ticks_per_wall_second": round(
            episode_ticks * spec.microticks
            / max(1e-9, time.perf_counter() - started), 1),
        "microticks": spec.microticks,
        "decision_period": spec.decision_period,
        #: How many engine ticks pass between two fresh decisions. This is the
        #: number that describes the agent -- `microticks` advances the engine
        #: inside one decision, `decision_period` repeats a decision across
        #: several. Both are ticks; neither is a duration.
        "decision_every_ticks": spec.decision_period * spec.microticks,
        # Wall-clock equivalents, kept because a REAL server ticks at a fixed
        # rate and a deployed agent has to answer within one. Nothing in
        # training should be tuned against these -- they describe production.
        "ticks_per_second": TICKS_PER_SECOND,
        "decision_interval_ms": round(
            spec.decision_period * spec.microticks / TICKS_PER_SECOND * 1000, 1
        ),
        # Derived from the ticks actually shown, not the ticks requested -- a
        # run that ended at 96 of 256 did not simulate 8.5 seconds.
        "simulated_seconds": round(
            episode_ticks * spec.microticks / TICKS_PER_SECOND, 2),
        "ability_starts": sum(trajectory["ability_start"]),
        "projectiles_max": max(finite("projectiles"), default=0),
        "stamina_floor": min(finite("stamina"), default=0.0),
        "failures": sorted({e["kind"] for e in events if e["kind"].startswith("failure:")}),
    }
    if nonfinite:
        summary["nonfinite"] = nonfinite

    # The console's whole reason for existing: say plainly when a run is
    # degenerate, instead of leaving a flat zero to be mistaken for a result.
    warnings_out = []
    if nonfinite:
        worst = ", ".join(f"{k} ({v})" for k, v in sorted(
            nonfinite.items(), key=lambda kv: -kv[1])[:4])
        warnings_out.append(
            f"NON-FINITE VALUES in {len(nonfinite)} series: {worst}. This is "
            "the finding, not a display glitch -- a NaN carried into the "
            "policy state disables an NPC permanently and nothing in the "
            "environment raises when it happens. Those ticks are null here, "
            "and every number above skips them."
        )
    if summary["visible_steps"] == 0:
        warnings_out.append(
            "the agent never perceived the opponent - every target column is "
            "masked, not zero. Enable 'perceive'."
        )
    if summary["accepted"] == 0:
        warnings_out.append(
            "no ability was ever accepted, so nothing this run says about "
            "damage is meaningful."
        )
    if summary["damage_taken"] == 0 and spec.opponent:
        warnings_out.append(
            "the agent took no damage - agent_damage and death are dead this "
            "run, so guard and dodge have no signal."
        )
    # There used to be a warning here on every `decision_period == 1` run,
    # telling the reader to raise it to 6-8 ticks for "human reaction latency".
    # That fired on every default run and was wrong to fire: acting once per
    # tick is CORRECT for training, and milliseconds are a production concern
    # -- a deployed agent must answer inside a real server's tick, but nothing
    # in training should be tuned against a human's reaction time. The agent is
    # tick-quantised, so the honest report is `decision_every_ticks` in the
    # summary, not a warning about a defect that is not one.
    if summary["landed"] == 0 and summary["accepted"] > 0:
        warnings_out.append(
            f"{summary['accepted']} abilities accepted and none landed. "
            "Measured hit rate is 0-2 per few hundred steps, so treat a single "
            "run as noise and sweep the period or seed."
        )

    terrain = None
    if spec.world == "region" and region_note.get("region_artifact") is not None:
        from console.core.worlds import terrain as terrain_module

        # `patch` only wants the extent, so each axis is filtered on its own.
        # A null here would reach `min()` and raise "'<' not supported between
        # NoneType and float" -- turning a NaN in the trajectory into a crash
        # in the map instead of the warning above.
        xs = finite("agent_x") + finite("target_x")
        ys = finite("agent_y") + finite("target_y")
        zs = finite("agent_z") + finite("target_z")
        if xs and ys and zs:
            terrain = terrain_module.patch(
                int(region_note["region_artifact"]), xs, ys, zs)

    summary["episode_ticks"] = episode_ticks
    # What this run cost. Wall clock alone hides the difference between a cold
    # build and a cache hit, and hides device memory entirely -- which is the
    # number that decides how many environments fit at once.
    summary["resources"] = _resource_cost(before)
    if episode_ticks != spec.ticks:
        summary["truncated_after_done"] = spec.ticks - episode_ticks

    return {
        "spec": asdict(spec),
        # The number of steps actually shown. Past the first `done` the agent is
        # a corpse and positions jump on the episode roll, so those ticks are
        # dropped rather than drawn as movement.
        "steps": episode_ticks,
        "head_names": list(HEAD_SPANS),
        "target_phases": list(TARGET_PHASES),
        "summary": summary,
        "warnings": warnings_out,
        "events": events,
        "trajectory": trajectory,
        "region": region_note or None,
        "terrain": terrain,
    }
