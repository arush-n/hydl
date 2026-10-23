"""How an agent feels: its emotions, its mood, and its feelings toward the entities it knows (``docs/VIRTUAL_HUMAN.md``;
owner, 2026-10-21: "there likely should be some emotion system as well like fear, anger").

Three timescales, all continuous, all in the neural layer's carry and read by every layer:

- **emotions** (seconds): fear, anger, startle, valence (joy or distress), and coping, the agent's sense of control
  over the threat, which decides whether a threat becomes fear (low control) or anger (high control) (Lazarus 1991;
  Lerner and Keltner 2001);
- **mood** (minutes): pleasure, arousal and dominance (PAD; Mehrabian 1996), pulled toward the active emotions and
  back to a baseline set by the persona (the ALMA layering: personality, mood, emotion; Gebhard 2005);
- **feelings toward an entity** (many minutes): hostility, fear, trust and anger, keyed by its id (GAMYGDALA's
  per-target relations; Popescu et al. 2014).

**Appraisal** (v0, hand-coded so the constants can be certified; a learned threat predictor can replace the
likelihood later): each threat's intensity is its likelihood times its cost (EMA; Marsella and Gratch 2009), and the
threats combine as ``T = 1 - prod(1 - theta)``:

- a hostile entity: ``sigmoid(2 attacking + closing - 2 (d / reach - 1))``, with the cost of one expected hit
  (``EXPECTED_HIT`` of my maximum health) over my health;
- a hostile projectile on course (closest approach within ``ON_COURSE_BLOCKS``): ``exp(-t / 0.7 s)`` times the same
  cost; a hostile hazard: ``0.5 exp(-d / 2)``.

Damage taken is my own health's drop (the simulator does not yet say who hit me or from where: S11). It is
attributed to a visible hostile within reach in its sweep, or else in its recovery (the simulator's sword lands as
the attacker enters recovery, ``docs/PERCEPTION.md`` section 10.5), or else counts as unattributed (and, with no
hostile in view at all, as unseen, which doubles fear and startles).

**Startle** (15.5; 20.2 and 20.6) is the reflex to damage that lands now, times four factors from the startle literature
and the persona's gain ``g_S = exp(negative_emotionality / 2)``. Its impulse is ``I_S = min(DAMAGE_GAIN taken P E H U D,
1)``, where:

- ``P = 1 + fear``: fear-potentiated startle, so a threat the agent has been watching raises its startle through alarm
  (20.6; Grillon et al. 1991). ``fear`` is last tick's;
- ``E = 1 - PREPULSE_INHIBITION cue``: expectation, prepulse inhibition (about 65% in healthy people, 20.6). ``cue``
  is 1 when the last warning onset came ``PREPULSE_MIN_TICKS`` to ``PREPULSE_MAX_TICKS`` ticks (30 to 500 ms) ago. A
  warning is a visible hostile in its windup within ``THREAT_REACH_BLOCKS``, or a hostile projectile on course; its
  onset is the first tick it shows. A threat seen for longer does not inhibit: it raises alarm, which potentiates;
- ``H = 1 / (1 + n)``: habituation to repeats, ``n`` the recent impulses, recovering with ``TAU_HABITUATION`` [E] (20.6:
  repeats habituate within a session; no source gives the recovery time in humans);
- ``U = 1 + unseen``: 2 when no hostile is in view, as before;
- ``D = 1 + d``: darkness, ``d = 1 - light`` of the nearest perceived cell (0 when no light is known) [E]. Darkness
  raises startle through affect, not attention (Grillon et al. 1997, 20.6).

The calm no longer makes it stronger: that proxy (the agent is not expecting the hit) is what ``E`` models (15.5). A
respawn (16.2) sets ``fear`` and ``S`` to 0 at the start of its tick, and nothing else.

**Dynamics** (leaky integrators, fixed shapes, no branches; ``lam(tau) = exp(-dt / tau)``)::

    c <- c + (c* - c)(1 - lam(TAU_COPING)),   c* = sigmoid(1 + 2 (h - 1/2) + risk_taking - 1.5 n_attacking + p)
    F <- F + (F* - F)(1 - lam(tau_F)) + g_F I_F (1 - F),   F* = T (1 - c)(1 + MU max(0, -mood_P))
    a0 <- a0 + (A* - a0)(1 - lam(TAU_ANGER)) + g_A I_0 (1 - a0),   A* = T c / 2
    a_j <- lam(TAU_ANGER) a_j + g_A K harm_j (1 - a_j),            anger = 1 - (1 - a0) prod_j (1 - a_j)
    S <- lam(TAU_STARTLE) S + g_S I_S (1 - S),   I_S as above (P E H U D)
    n <- lam(TAU_HABITUATION) n + I_S
    V <- clip(lam(TAU_VALENCE) V + VALENCE_GAIN (g+ dealt - g- taken), -1, 1)
    M <- M + (1 - lam(TAU_MOOD))(M0 - M) + MOOD_PULL dt sum_e w_e (P_e - M)

with ``tau_F`` = ``TAU_FEAR_RISE`` while fear rises, else ``TAU_FEAR_DECAY exp(negative_emotionality / 2)``; the
gains ``g = exp(trait)`` (negative emotionality for fear and startle; negative emotionality and negative reciprocity for
anger, angry hostility being a facet of neuroticism (NEO-PI-R; recalled); positive and negative emotionality for
valence); ``P_e`` the PAD points of fear (-0.64, 0.60, -0.43), anger (-0.51, 0.59, 0.25),
joy (0.4, 0.2, 0.1) and sadness (-0.6, -0.4, -0.5) (Gebhard 2005); and ``M0`` Mehrabian's mapping from the persona's
Big Five view. Feelings toward entity j decay with their own constants::

    hostility_j <- lam(TAU_HOSTILITY) hostility_j + g_A K harm_j (1 - hostility_j)
    fear_j      <- max(lam(TAU_FEAR_MEMORY) fear_j, F [j is the top threat])
    trust_j     <- clip(lam(TAU_TRUST) trust_j - 2 K harm_j, -1, 1)

Every time constant and gain is an estimate (marked [E] in ``docs/VIRTUAL_HUMAN.md``) to be certified by the step
and monotonicity tests, then fitted to behaviour.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.contract.surface import ATTACK_PHASES
from experimental.neural_agent.skill_agent.state.senses import (
    APPROACH_DISTANCE_BLOCKS,
    APPROACH_HORIZON_SECONDS,
    Perceived,
)
from experimental.neural_agent.skill_agent.state.frames import POSITION_SCALE_BLOCKS
from experimental.neural_agent.skill_agent.state.persona import big_five, trait

DT_SECONDS: float = 0.045
"""One gen-7 simulator tick."""
TAU_FEAR_RISE, TAU_FEAR_DECAY = 0.3, 6.0
TAU_ANGER, TAU_STARTLE, TAU_VALENCE, TAU_COPING, TAU_MOOD = 12.0, 1.2, 8.0, 2.0, 180.0
TAU_HOSTILITY, TAU_FEAR_MEMORY, TAU_TRUST = 600.0, 900.0, 1800.0
"""Time constants in seconds [E]: fear rises fast and fades in seconds, anger outlasts it, startle is brief, mood takes
minutes (WASABI; Left 4 Dead's director; Verduyn and Lavrijsen 2015), feelings toward an entity last many minutes."""
DAMAGE_GAIN: float = 4.0
"""Impulse per unit of my maximum health lost (a hit taking 10% gives an impulse of 0.4) [E]."""
PREPULSE_INHIBITION: float = 0.65
"""How much a warning lowers a startle that comes 30 to 500 ms after it (20.6: about 65% in healthy people)."""
PREPULSE_MIN_TICKS, PREPULSE_MAX_TICKS = 1, 11
"""That window in ticks: 30 to 500 ms at 0.045 s a tick (20.6; leads of 2 to 4.5 s facilitate instead)."""
TAU_HABITUATION: float = 120.0
"""Seconds for a startle's habituation to recover [E] (20.6: no source gives the recovery time in humans)."""
NO_WARNING_TICKS: int = 1_000_000
"""``since_warning`` before any warning, and its ceiling: the count saturates there."""
EXPECTED_HIT: float = 0.1
"""The cost of one expected hit, as a fraction of my maximum health [E]."""
THREAT_REACH_BLOCKS: float = 3.75
"""The reach an entity's attack is assumed to have (its own reach is not published): the longest measured melee hit,
the iron sword's and the mace's 3.75 blocks centre to centre (``runtime/scripted_bodies.py``, ``MELEE_REACH_BLOCKS``,
measured 2026-10-20 on gen 5 and gen 7). It was an estimate of 3.0 until the encoding audit found sword hits landing
from 3.48 blocks unattributed (``docs/PERCEPTION.md`` section 10.5)."""
ON_COURSE_BLOCKS: float = 1.5
"""A projectile whose closest approach passes nearer than this is on course [E]."""
MOOD_CONGRUENCE: float = 0.3
"""How much a bad mood raises fear's target (kept low: a mood loop gain of 1.2 oscillates, Eldar and Niv 2015)."""
MOOD_PULL: float = 0.03
"""Per second, how strongly the active emotions pull the mood [E]."""
VALENCE_GAIN: float = 5.0
"""Valence per unit of health dealt or taken (a hit taking 10% of my health: -0.5) [E]."""
ENTITY_SLOTS: int = 16
"""Entities whose feelings are kept (the least recently seen gives way)."""
_PAD = jnp.asarray([[-0.64, 0.60, -0.43], [-0.51, 0.59, 0.25], [0.4, 0.2, 0.1], [-0.6, -0.4, -0.5]], jnp.float32)
"""The PAD points of fear, anger, joy and sadness (Gebhard 2005)."""
_E = {name: index for index, name in enumerate(v3.EntityTokenV3.FEATURES)}
_P = {name: index for index, name in enumerate(v3.ProjectileTokenV3.FEATURES)}
_H = {name: index for index, name in enumerate(v3.HazardTokenV3.FEATURES)}
_S = {name: index for index, name in enumerate(v3.SelfTokenV3.FEATURES)}
_G = {name: index for index, name in enumerate(v3.GeometryTokenV3.FEATURES)}


class AffectState(NamedTuple):
    """The emotional state of B agents and their feelings toward up to K entities each."""

    fear: jax.Array  # float32[B]
    anger_unattributed: jax.Array  # float32[B]: anger at no one in particular (a0)
    startle: jax.Array  # float32[B]
    valence: jax.Array  # float32[B]
    coping: jax.Array  # float32[B]
    mood: jax.Array  # float32[B, 3]: pleasure, arousal, dominance
    health: jax.Array  # float32[B]: my health fraction last tick (-1 before the first)
    entity_id: jax.Array  # int32[B, K]: -1 for an empty slot
    feelings: jax.Array  # float32[B, K, 4]: hostility, fear, trust, anger (``tokens_v3.FEELINGS``)
    entity_health: jax.Array  # float32[B, K]: its health fraction when last perceived
    last_seen: jax.Array  # int32[B, K]: the tick it was last perceived
    now: jax.Array  # int32[B]
    attributed: jax.Array  # int32[B]: the entity id this tick's damage taken was credited to; -1 none
    warned: jax.Array  # bool[B]: a warning was in view last tick (20.2)
    since_warning: jax.Array  # int32[B]: ticks since the last warning onset (as of last tick); saturates
    recent: jax.Array  # float32[B]: the recent startle impulses, recovering with TAU_HABITUATION (habituation's n)


def mood_baseline(persona: jax.Array) -> jax.Array:
    """float32[B, 3]: the mood the persona returns to (Mehrabian 1996, on the persona's Big Five view, stability being
    the opposite of neuroticism), clipped to [-1, 1]."""

    five = big_five(persona)
    stability = -five["neuroticism"]
    pleasure = 0.59 * five["agreeableness"] + 0.19 * stability + 0.21 * five["extraversion"]
    arousal = -0.57 * stability + 0.30 * five["agreeableness"] + 0.15 * five["openness"]
    dominance = (0.60 * five["extraversion"] - 0.32 * five["agreeableness"] + 0.25 * five["openness"]
                 + 0.17 * five["conscientiousness"])
    return jnp.clip(jnp.stack([pleasure, arousal, dominance], axis=-1), -1.0, 1.0)


def initial_affect(persona: jax.Array) -> AffectState:
    """Calm, at the persona's baseline mood, with no feelings toward anyone."""

    batch = persona.shape[0]
    zero = jnp.zeros((batch,), jnp.float32)
    return AffectState(zero, zero, zero, zero, jnp.full((batch,), 0.5, jnp.float32), mood_baseline(persona),
                       jnp.full((batch,), -1.0, jnp.float32), jnp.full((batch, ENTITY_SLOTS), -1, jnp.int32),
                       jnp.zeros((batch, ENTITY_SLOTS, len(v3.FEELINGS)), jnp.float32),
                       jnp.zeros((batch, ENTITY_SLOTS), jnp.float32), jnp.zeros((batch, ENTITY_SLOTS), jnp.int32),
                       jnp.zeros((batch,), jnp.int32), jnp.full((batch,), -1, jnp.int32),
                       jnp.zeros((batch,), jnp.bool_), jnp.full((batch,), NO_WARNING_TICKS, jnp.int32),
                       jnp.zeros((batch,), jnp.float32))


def _lam(tau: jax.Array | float, dt: float) -> jax.Array:
    return jnp.exp(-dt / jnp.asarray(tau, jnp.float32))


def _rows(perceived: Perceived, name: str, width: int) -> tuple[jax.Array, jax.Array]:
    found = perceived.rows.get(name)
    if found is None:
        batch = perceived.pose.body_yaw.shape[0]
        return jnp.zeros((batch, 0, width), jnp.float32), jnp.zeros((batch, 0), jnp.bool_)
    return found


def _distance(rows: jax.Array) -> jax.Array:
    return jnp.linalg.norm(rows[..., :3], axis=-1) * POSITION_SCALE_BLOCKS


def _slots(affect: AffectState, ids: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each perceived entity's slot int32[B, E] (a known id keeps its slot; a new one takes an empty slot, else the
    least recently seen), whether that slot is new to it bool[B, E], and the table's ids after."""

    def one(table, last_seen, candidate_ids, candidate_valid, now):
        def write(state, candidate):
            table, last_seen = state
            ident, ok = candidate
            same = (table == ident) & (table >= 0)
            known = same.any()
            score = jnp.where(table < 0, -1, last_seen)
            slot = jnp.where(known, jnp.argmax(same), jnp.argmin(score))
            table = jnp.where(ok, table.at[slot].set(ident), table)
            last_seen = jnp.where(ok, last_seen.at[slot].set(now), last_seen)
            return (table, last_seen), (slot, ok & ~known)

        (table, _), (slot, fresh) = jax.lax.scan(write, (table, last_seen), (candidate_ids, candidate_valid))
        return slot, fresh, table

    return jax.vmap(one)(affect.entity_id, affect.last_seen, ids, valid, affect.now)


def _darkness(perceived: Perceived) -> jax.Array:
    """float32[B]: ``d = 1 - light`` of the nearest perceived cell, light the larger of its sky and block light (the
    ``light_*`` columns, 0 to 1); 0 with no cell, or with no light known (the factor ``1 + d`` is then 1)."""

    cells, valid = _rows(perceived, "geometry", len(v3.GeometryTokenV3.FEATURES))  # float32[B, G, F], bool[B, G]
    batch, count = valid.shape
    if count == 0:  # the shapes are static: no geometry rows this tick
        return jnp.zeros((batch,), jnp.float32)
    nearest = jnp.argmin(jnp.where(valid, _distance(cells), jnp.inf), axis=1)
    cell = jnp.take_along_axis(cells, nearest[:, None, None], axis=1)[:, 0]  # float32[B, F]
    sky_valid, block_valid = cell[:, _G["light_sky_valid"]], cell[:, _G["light_block_light_rgb_valid"]]
    block = cell[:, _G["light_block_light_red"]:_G["light_block_light_blue"] + 1].max(axis=1)
    light = jnp.maximum(sky_valid * cell[:, _G["light_sky_light"]], block_valid * block)
    known = jnp.any(valid, axis=1) & (jnp.maximum(sky_valid, block_valid) > 0.5)
    return jnp.where(known, 1.0 - light, 0.0)


def update(affect: AffectState, perceived: Perceived, persona: jax.Array, reset: jax.Array | None = None,
           dt: float = DT_SECONDS, respawn: jax.Array | None = None) -> AffectState:
    """The emotional state after this tick's perception (module docstring). Lanes where ``reset`` bool[B] is True start
    from :func:`initial_affect` first. Lanes where ``respawn`` bool[B] is True (16.2) start their alarm and startle
    over at the start of the tick, and keep everything else."""

    if reset is not None:
        fresh = initial_affect(persona)
        affect = jax.tree_util.tree_map(
            lambda new, old: jnp.where(reset.reshape(reset.shape + (1,) * (old.ndim - 1)), new, old), fresh, affect)
    if respawn is not None:
        affect = affect._replace(fear=jnp.where(respawn, 0.0, affect.fear),
                                 startle=jnp.where(respawn, 0.0, affect.startle))
    batch = persona.shape[0]
    now = affect.now + 1
    self_rows, _ = _rows(perceived, "self", len(v3.SelfTokenV3.FEATURES))
    health = self_rows[:, 0, _S["health_fraction"]]
    taken = jnp.where(affect.health >= 0.0, jnp.maximum(affect.health - health, 0.0), 0.0)
    cost = jnp.minimum(EXPECTED_HIT / jnp.maximum(health, 0.05), 1.0)[:, None]

    entities, entity_valid = _rows(perceived, "entity", len(v3.EntityTokenV3.FEATURES))
    count = entities.shape[1]
    ids = perceived.entity_ids[:, :count]
    hostile = jnp.maximum(entities[..., _E["attitude_hostile"]], entities[..., _E["hostile"]]) * entity_valid
    distance = _distance(entities)
    attacking = entities[..., _E[f"phase_{ATTACK_PHASES[1]}"]] + entities[..., _E[f"phase_{ATTACK_PHASES[2]}"]]
    direction = entities[..., :3] / jnp.maximum(jnp.linalg.norm(entities[..., :3], axis=-1, keepdims=True), 1e-6)
    closing = -jnp.sum(direction * entities[..., _E["v_forward"]:_E["v_forward"] + 3], axis=-1)
    likelihood = jax.nn.sigmoid(2.0 * attacking + closing - 2.0 * (distance / THREAT_REACH_BLOCKS - 1.0))
    entity_threat = hostile * likelihood * cost

    projectiles, projectile_valid = _rows(perceived, "projectile", len(v3.ProjectileTokenV3.FEATURES))
    on_course = projectiles[..., _P["approach_distance"]] * APPROACH_DISTANCE_BLOCKS < ON_COURSE_BLOCKS
    seconds = projectiles[..., _P["approach_time"]] * APPROACH_HORIZON_SECONDS
    projectile_threat = (projectile_valid * on_course * projectiles[..., _P["hostile"]] * jnp.exp(-seconds / 0.7)
                         * cost)
    hazards, hazard_valid = _rows(perceived, "hazard", len(v3.HazardTokenV3.FEATURES))
    hazard_threat = hazard_valid * hazards[..., _H["hostile"]] * 0.5 * jnp.exp(-_distance(hazards) / 2.0)
    every = jnp.concatenate([entity_threat, projectile_threat, hazard_threat], axis=1)
    threat = 1.0 - jnp.prod(1.0 - jnp.clip(every, 0.0, 1.0), axis=1)

    sweep = (entities[..., _E[f"phase_{ATTACK_PHASES[2]}"]] > 0.5).astype(jnp.float32)
    recovery = (entities[..., _E[f"phase_{ATTACK_PHASES[3]}"]] > 0.5).astype(jnp.float32)
    swinging = hostile * (2.0 * sweep + recovery) * (distance <= THREAT_REACH_BLOCKS)  # a sweep outranks a recovery
    attributed = jnp.any(swinging > 0, axis=1) & (taken > 0)
    attacker = jnp.argmax(swinging, axis=1)
    harm = jnp.where(attributed[:, None], jax.nn.one_hot(attacker, count) * taken[:, None], 0.0)  # [B, E]
    unseen = (taken > 0) & ~jnp.any(hostile > 0, axis=1)
    attackers_near = jnp.sum(hostile * (attacking > 0.5) * (distance <= 2.0 * THREAT_REACH_BLOCKS), axis=1)
    top = jnp.argmax(entity_threat, axis=1)
    has_top = jnp.any(entity_threat > 0, axis=1)
    their_health = jnp.take_along_axis(entities[..., _E["health_fraction"]], top[:, None], axis=1)[:, 0]
    power = jnp.where(has_top, health - their_health, 0.0)

    risk = trait(persona, "risk_taking")
    negative, positive = trait(persona, "negative_emotionality"), trait(persona, "positive_emotionality")
    coping_target = jax.nn.sigmoid(1.0 + 2.0 * (health - 0.5) + risk - 1.5 * attackers_near + power)
    coping = affect.coping + (coping_target - affect.coping) * (1.0 - _lam(TAU_COPING, dt))

    fear_target = jnp.clip(threat * (1.0 - coping) * (1.0 + MOOD_CONGRUENCE * jnp.maximum(-affect.mood[:, 0], 0.0)),
                           0.0, 1.0)
    tau_fear = jnp.where(fear_target > affect.fear, TAU_FEAR_RISE, TAU_FEAR_DECAY * jnp.exp(0.5 * negative))
    fear_impulse = DAMAGE_GAIN * taken * (1.0 + unseen)
    fear = affect.fear + (fear_target - affect.fear) * (1.0 - _lam(tau_fear, dt))
    fear = jnp.clip(fear + jnp.exp(negative) * fear_impulse * (1.0 - fear), 0.0, 1.0)

    anger_gain = jnp.exp(0.5 * negative + 0.5 * trait(persona, "negative_reciprocity"))
    anger_target = threat * coping / 2.0
    unattributed = DAMAGE_GAIN * taken * (~attributed) * 0.3
    anger0 = affect.anger_unattributed + (anger_target - affect.anger_unattributed) * (1.0 - _lam(TAU_ANGER, dt))
    anger0 = jnp.clip(anger0 + anger_gain * unattributed * (1.0 - anger0), 0.0, 1.0)

    # The warning (20.2): a visible hostile in its windup within reach, or a hostile projectile on course. Its cue is
    # the age of the last onset, counted before this tick's onset, so a warning on the same tick as the hit inhibits
    # nothing.
    windup = entities[..., _E[f"phase_{ATTACK_PHASES[1]}"]] > 0.5
    warning = jnp.any((hostile > 0) & (entities[..., _E["visible"]] > 0.5) & windup
                      & (distance <= THREAT_REACH_BLOCKS), axis=1)
    warning = warning | jnp.any(projectile_valid & on_course & (projectiles[..., _P["hostile"]] > 0), axis=1)
    since = jnp.minimum(affect.since_warning + 1, NO_WARNING_TICKS)  # ticks since the last onset, this tick included
    cue = (since >= PREPULSE_MIN_TICKS) & (since <= PREPULSE_MAX_TICKS)
    since_warning = jnp.where(warning & ~affect.warned, 0, since)
    # The startle (15.5; module docstring): the damage now, times fear potentiation, expectation (prepulse inhibition),
    # habituation, the unseen doubling and darkness. The calm no longer raises it.
    expectation = 1.0 - PREPULSE_INHIBITION * cue.astype(jnp.float32)
    startle_impulse = jnp.minimum(DAMAGE_GAIN * taken * (1.0 + affect.fear) * expectation / (1.0 + affect.recent)
                                  * (1.0 + unseen.astype(jnp.float32)) * (1.0 + _darkness(perceived)), 1.0)
    startle = (_lam(TAU_STARTLE, dt) * affect.startle
               + jnp.exp(0.5 * negative) * startle_impulse * (1.0 - affect.startle))
    recent = _lam(TAU_HABITUATION, dt) * affect.recent + startle_impulse

    slot, fresh, table = _slots(affect, ids, entity_valid)
    rows_index = jnp.arange(batch)[:, None]
    previous = affect.entity_health[rows_index, slot]
    now_health = entities[..., _E["health_fraction"]]
    dealt = jnp.sum(jnp.where(entity_valid & ~fresh & (hostile > 0), jnp.maximum(previous - now_health, 0.0), 0.0),
                    axis=1)
    valence = jnp.clip(_lam(TAU_VALENCE, dt) * affect.valence
                       + VALENCE_GAIN * (jnp.exp(positive) * dealt - jnp.exp(negative) * taken), -1.0, 1.0)

    feelings = jnp.where((table != affect.entity_id)[..., None], 0.0, affect.feelings)  # a slot given a new entity
    harm_slot = jnp.zeros((batch, ENTITY_SLOTS)).at[rows_index, slot].add(jnp.where(entity_valid, harm, 0.0))
    impulse = anger_gain[:, None] * DAMAGE_GAIN * harm_slot
    hostility = _lam(TAU_HOSTILITY, dt) * feelings[..., 0]
    hostility = hostility + impulse * (1.0 - hostility)
    top_slot = jnp.where(has_top, jnp.take_along_axis(slot, top[:, None], axis=1)[:, 0], -1)
    fear_memory = jnp.maximum(_lam(TAU_FEAR_MEMORY, dt) * feelings[..., 1],
                              jnp.where(jnp.arange(ENTITY_SLOTS)[None, :] == top_slot[:, None], fear[:, None], 0.0))
    trust = jnp.clip(_lam(TAU_TRUST, dt) * feelings[..., 2] - 2.0 * DAMAGE_GAIN * harm_slot, -1.0, 1.0)
    anger_j = _lam(TAU_ANGER, dt) * feelings[..., 3]
    anger_j = anger_j + impulse * (1.0 - anger_j)
    feelings = jnp.clip(jnp.stack([hostility, fear_memory, trust, anger_j], axis=-1), -1.0, 1.0)
    occupied = (table >= 0)[..., None]
    feelings = jnp.where(occupied, feelings, 0.0)
    entity_health = affect.entity_health.at[rows_index, slot].set(
        jnp.where(entity_valid, now_health, affect.entity_health[rows_index, slot]))

    anger = 1.0 - (1.0 - anger0) * jnp.prod(1.0 - feelings[..., 3], axis=1)
    weights = jnp.stack([fear, anger, jnp.maximum(valence, 0.0), jnp.maximum(-valence, 0.0)], axis=-1)
    pull = jnp.einsum("be,bek->bk", weights, _PAD[None] - affect.mood[:, None, :])
    mood = affect.mood + (mood_baseline(persona) - affect.mood) * (1.0 - _lam(TAU_MOOD, dt)) + MOOD_PULL * dt * pull
    credited = jnp.where(attributed, jnp.take_along_axis(ids, attacker[:, None], axis=1)[:, 0], -1).astype(jnp.int32)
    return AffectState(fear, anger0, startle, valence, coping, jnp.clip(mood, -1.0, 1.0), health, table, feelings,
                       entity_health, affect.last_seen.at[rows_index, slot].set(
                           jnp.where(entity_valid, now[:, None], affect.last_seen[rows_index, slot])), now, credited,
                       warning, since_warning, recent)


def anger(affect: AffectState) -> jax.Array:
    """float32[B]: anger at anyone, ``1 - (1 - a0) prod_j (1 - a_j)``."""

    return 1.0 - (1.0 - affect.anger_unattributed) * jnp.prod(1.0 - affect.feelings[..., 3], axis=1)


def affect_features(affect: AffectState) -> jax.Array:
    """float32[B, len(AFFECT_FEATURES)]: the ``affect`` token's features (all but ``marked``)."""

    return jnp.concatenate([jnp.stack([affect.fear, anger(affect), affect.startle, affect.valence, affect.coping],
                                      axis=-1), affect.mood], axis=-1)


def feelings_for(affect: AffectState, ids: jax.Array) -> jax.Array:
    """float32[B, N, 4]: the feelings toward each entity id int32[B, N] (zeros for one not kept)."""

    match = (affect.entity_id[:, None, :] == ids[..., None]) & (ids[..., None] >= 0)
    return jnp.einsum("bnk,bkf->bnf", match.astype(jnp.float32), affect.feelings)


__all__ = ["AffectState", "DT_SECONDS", "ENTITY_SLOTS", "affect_features", "anger", "feelings_for",
           "initial_affect", "mood_baseline", "update"]
