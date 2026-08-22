"""Compiled ability legality, start, cooldown, and timed-event scheduling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    ABILITY_CHARGE_CAPACITY,
    ARSENAL_FAILURE_INVALID_COMMAND,
    ARSENAL_FAILURE_EVENT_OVERFLOW,
    ARSENAL_FAILURE_INVALID_LOADOUT,
    ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    EF_SELECTOR_RUNTIME_SECONDS,
    EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA,
    FIRED_EVENT_CAPACITY,
    INTERACTION_QUEUE_DELAY_TICKS,
    NATIVE_NPC_INSTANT_ADD_STAT_APPLICATIONS,
    REQUIRE_CLEAR_FORCE_PATH,
    REQUIRE_CLEAR_PROJECTILE_FLIGHT,
    REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE,
    REQUIRE_ENTITY_ONLY_AREA,
    REQUIRE_LINE_OF_SIGHT,
    REQUIRE_STATIC_AREA_PLACEMENT,
    REQUIRE_WORLD_PROJECTILE_COLLISION,
    RESOURCE_COST_NONE,
    RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT,
    RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE,
    RESOURCE_COST_SINGLE_APPLICATION,
    RESOURCE_COST_SINGLE_APPLICATION_STAMINA_BREAK_IMMUNE,
)
from hytalegym.jax.combat.arsenal.programs.scheduling import (
    SCHEDULER_CLOCK_EPSILON_SECONDS,
    advance_scheduler_clocks,
    event_clock_crossed,
    event_dispatch_delay_ticks,
    event_uses_progressive_selector,
    scheduler_clock,
    scheduler_clock_reached,
)
from hytalegym.jax.combat.arsenal.programs.outer_roots import (
    condition_internal_legality,
    policy_ability_mask,
    project_legal_mask,
    resolve_requested_slot,
)
from hytalegym.jax.combat.arsenal.programs.interaction_rules import (
    active_operation_interrupted_by,
)
from hytalegym.jax.combat.arsenal.programs.continuation import (
    CONTINUATION_NONE,
    advance_continuation_clock,
    continuation_resolution,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    AbilityLoadout,
    ArsenalState,
    ArsenalWorldCapabilities,
    FiredEvents,
)
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsRules,
    CombatMechanicsState,
    RESOURCE_STAMINA,
    STAMINA_BROKEN_REGEN_DELAY_SECONDS,
    STATUS_FLAG_DISABLE_ABILITIES,
    status_modifiers,
)


def ability_legal_mask(
    arsenal: ArsenalState,
    mechanics: CombatMechanicsState,
    loadout: AbilityLoadout,
    world: ArsenalWorldCapabilities,
    alive: jax.Array,
) -> jax.Array:
    """Return legality for every entity and ability without side effects."""

    root_admission = _ability_root_admission_mask(
        arsenal,
        mechanics,
        loadout,
        alive,
    )
    base = condition_internal_legality(
        loadout,
        mechanics,
        _internal_ability_base_mask(
            arsenal,
            mechanics,
            loadout,
            world,
            alive,
        ),
    )
    interaction_available = _ability_interaction_available(
        arsenal,
        loadout,
    )
    return project_legal_mask(
        loadout,
        base & interaction_available,
        base,
        interaction_available,
        root_admission,
    )


def _internal_ability_legal_mask(
    arsenal: ArsenalState,
    mechanics: CombatMechanicsState,
    loadout: AbilityLoadout,
    world: ArsenalWorldCapabilities,
    alive: jax.Array,
) -> jax.Array:
    """Return child-program legality before an outer root selects a child."""

    return _internal_ability_base_mask(
        arsenal,
        mechanics,
        loadout,
        world,
        alive,
    ) & _ability_interaction_available(arsenal, loadout)


def _internal_ability_base_mask(
    arsenal: ArsenalState,
    mechanics: CombatMechanicsState,
    loadout: AbilityLoadout,
    world: ArsenalWorldCapabilities,
    alive: jax.Array,
) -> jax.Array:
    """Return program-local legality before active-chain arbitration."""

    resource_enough = jnp.all(
        mechanics.resources[:, :, None, :] + jnp.float32(1.0e-6)
        >= loadout.ability_resource_minimum,
        axis=3,
    )
    # A policy-selected child behind an authored outer root is admitted by
    # that root.  Its private resource row is populated later by the root's
    # inventory/nock phase and must not make the public control disappear.
    resource_enough |= loadout.ability_outer_root_selector
    requirements = _requirements_available(
        loadout.ability_requirements,
        world,
    )
    return (
        _ability_root_admission_mask(
            arsenal,
            mechanics,
            loadout,
            alive,
        )
        & resource_enough
        & requirements
    )


def _ability_root_admission_mask(
    arsenal: ArsenalState,
    mechanics: CombatMechanicsState,
    loadout: AbilityLoadout,
    alive: jax.Array,
) -> jax.Array:
    """Return root-local gates before a selected child is evaluated."""

    flags, _ = status_modifiers(mechanics.statuses)
    status_allows = (flags & jnp.uint32(STATUS_FLAG_DISABLE_ABILITIES)) == 0
    guard_fork = loadout.ability_guard_fork_type >= 0
    guard_allows = jnp.where(
        guard_fork,
        mechanics.guard_held[:, :, None],
        ~mechanics.guard_held[:, :, None],
    )
    return (
        loadout.ability_mask
        & loadout.equipped[:, :, None]
        & ~loadout.overflow[:, :, None]
        & (arsenal.ability_cooldown_seconds <= 0.0)
        & (arsenal.ability_charge_count > 0)
        & alive[:, :, None]
        & status_allows[:, :, None]
        # Ordinary roots are refused while Guard is admitted. An authored
        # Guard fork is the inverse: it is legal only while its parent is
        # already held. The runtime retains that held level for the fork edge,
        # so the policy still emits exactly one standard-input choice.
        & guard_allows
        & (arsenal.failure_bits == 0)[:, None, None]
        & (mechanics.failure_bits == 0)[:, None, None]
    )


def _ability_interaction_available(
    arsenal: ArsenalState,
    loadout: AbilityLoadout,
) -> jax.Array:
    """Apply current-operation ``InterruptedBy`` rules per incoming root."""

    active = arsenal.active_ability_slot >= 0
    return (~active[..., None]) | active_operation_interrupted_by(
        arsenal,
        loadout,
        loadout.ability_interaction_type,
    )


def ability_lifecycle_admitted(
    arsenal: ArsenalState,
    loadout: AbilityLoadout,
) -> jax.Array:
    """Return programs admitted past Hytale's queued-root boundary.

    ``active_ability_slot`` is also the internal pending-program handle: JAX
    needs it immediately so the child scheduler can traverse the same queued
    interaction graph as the server.  Native actor evidence, however, does
    not publish a root as active until ``InteractionManager`` drains
    ``chainStartQueue`` on the following tick.  The per-ability scheduler
    prelude delays child execution (not admission), so add it back before
    testing the universal queue boundary.
    """

    ability_capacity = _runtime_ability_capacity(arsenal, loadout)
    active = arsenal.active_ability_slot >= 0
    slot = jnp.clip(
        arsenal.active_ability_slot,
        0,
        ability_capacity - 1,
    )
    prelude_ticks = _gather_ability(
        loadout.ability_scheduler_prelude_ticks,
        slot,
    )
    lifecycle_tick = arsenal.ability_scheduler_tick + prelude_ticks
    return active & (lifecycle_tick > jnp.int32(INTERACTION_QUEUE_DELAY_TICKS))


def ability_lifecycle_legality_view(
    arsenal: ArsenalState,
    loadout: AbilityLoadout,
) -> ArsenalState:
    """Expose only roots which native has admitted to its active-chain map.

    Hytale's ``InteractionManager.applyRules`` evaluates active and forked
    chains, but not roots still waiting in ``chainStartQueue``. JAX retains a
    queued root in ``active_ability_slot`` immediately because its bounded
    scheduler needs a pending-program handle. Feeding that private handle to
    :func:`ability_legal_mask` closes the runtime one tick earlier than native
    and disagrees with the actor-visible policy mask.

    This projection changes only the state used for admission. The scheduler
    continues to own the original pending handle and its child-event clocks.
    """

    admitted = ability_lifecycle_admitted(arsenal, loadout)
    return arsenal._replace(
        active_ability_slot=jnp.where(
            admitted,
            arsenal.active_ability_slot,
            jnp.int32(-1),
        ),
        active_ability_root_slot=jnp.where(
            admitted,
            arsenal.active_ability_root_slot,
            jnp.int32(-1),
        ),
    )


def start_abilities(
    arsenal: ArsenalState,
    mechanics: CombatMechanicsState,
    requested_slot: jax.Array,
    loadout: AbilityLoadout,
    world: ArsenalWorldCapabilities,
    alive: jax.Array,
    rules: CombatMechanicsRules,
    admission_mechanics: CombatMechanicsState | None = None,
) -> tuple[ArsenalState, CombatMechanicsState, jax.Array, jax.Array]:
    """Start legal programs; queued resource spends occur on their native tick.

    ``mechanics`` is the post-command resource state which the new program
    inherits. ``admission_mechanics`` may supply the pre-command interaction
    view used by native ``applyRules``. This distinction matters when guard
    rises or falls on the same control tick as an ability: the existing guard
    chain blocks through its release tick, while a newly queued guard is not
    yet part of the admitted-chain set.
    """

    ability_capacity = _runtime_ability_capacity(arsenal, loadout)
    requested = requested_slot >= 0
    invalid_slot = requested & (
        (requested_slot < 0) | (requested_slot >= ability_capacity)
    )
    bits = _set_failure(
        arsenal.failure_bits,
        jnp.any(invalid_slot, axis=1),
        ARSENAL_FAILURE_INVALID_COMMAND,
    )
    bits = _set_failure(
        bits,
        jnp.any(loadout.overflow, axis=1),
        ARSENAL_FAILURE_INVALID_LOADOUT,
    )
    original_slot = jnp.clip(requested_slot, 0, ability_capacity - 1)
    profile_present = _gather_ability(
        policy_ability_mask(loadout),
        original_slot,
    )
    legality_arsenal = ability_lifecycle_legality_view(arsenal, loadout)
    legality_mechanics = (
        mechanics if admission_mechanics is None else admission_mechanics
    )
    internal_base = _internal_ability_base_mask(
        legality_arsenal,
        legality_mechanics,
        loadout,
        world,
        alive,
    )
    internal_base = condition_internal_legality(
        loadout,
        legality_mechanics,
        internal_base,
    )
    root_admission = _ability_root_admission_mask(
        legality_arsenal,
        legality_mechanics,
        loadout,
        alive,
    )
    slot, selectable = resolve_requested_slot(
        loadout,
        internal_base,
        requested_slot,
    )
    requirement_available = _gather_ability(
        _requirements_evidence_available(
            loadout.ability_requirements,
            world,
        ),
        slot,
    )
    unsupported = requested & profile_present & ~requirement_available
    bits = _set_failure(
        bits,
        jnp.any(unsupported, axis=1),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    selected_program_legal = _gather_ability(
        internal_base,
        slot,
    )
    incoming_type = _gather_ability(
        loadout.ability_interaction_type,
        original_slot,
    )
    interaction_available = (
        legality_arsenal.active_ability_slot < 0
    ) | active_operation_interrupted_by(
        legality_arsenal,
        loadout,
        incoming_type,
    )
    requested_root_legal = _gather_ability(
        root_admission,
        original_slot,
    )
    legal = selected_program_legal & requested_root_legal & interaction_available
    row_valid = bits == jnp.uint32(0)
    provisionally_legal = requested & selectable & legal & row_valid[:, None]
    # ``applyRules`` deliberately ignores roots still waiting in
    # ``InteractionManager.chainStartQueue``, so the preceding learner mask
    # advertises this narrow frame.  The current BA53 live discriminator shows
    # the bridge then refusing a repeated request as ``native_combat_busy``
    # while the original queued root starts.  Preserve that first bounded
    # handle and reject the duplicate without rewinding its scheduler.
    pending = (arsenal.active_ability_slot >= 0) & ~ability_lifecycle_admitted(
        arsenal, loadout
    )
    accepted = provisionally_legal & ~pending
    # Cooldown, charges, and root-level timing belong to the policy-requested
    # root.  ``slot`` may be a server-selected child (Crossbow standard,
    # reload, or Big Arrow) and owns only execution events/resources.
    admission_slot = original_slot
    delay_value = _gather_ability(
        loadout.ability_stamina_regen_delay_seconds,
        admission_slot,
    )
    delay_start_tick = _gather_ability(
        loadout.ability_stamina_regen_delay_start_tick,
        admission_slot,
    )
    delay = mechanics.stamina_regen_delay_seconds
    delay = jnp.where(
        accepted & (delay_value < 0.0) & (delay_start_tick == 0),
        jnp.minimum(delay, delay_value),
        delay,
    )
    mechanics_candidate = mechanics._replace(
        stamina_regen_delay_seconds=delay,
    )
    cooldown = _gather_ability(
        loadout.ability_cooldown_seconds,
        admission_slot,
    )
    cooldowns = _set_ability(
        arsenal.ability_cooldown_seconds,
        admission_slot,
        jnp.where(
            accepted,
            cooldown,
            _gather_ability(
                arsenal.ability_cooldown_seconds,
                admission_slot,
            ),
        ),
    )
    current_charge_count = _gather_ability(
        arsenal.ability_charge_count,
        admission_slot,
    )
    charge_counts = _set_ability(
        arsenal.ability_charge_count,
        admission_slot,
        jnp.where(
            accepted,
            jnp.maximum(jnp.int32(0), current_charge_count - jnp.int32(1)),
            current_charge_count,
        ),
    )
    current_charge_timer = _gather_ability(
        arsenal.ability_charge_timer_seconds,
        admission_slot,
    )
    interrupt_recharge = _gather_ability(
        loadout.ability_interrupt_recharge,
        admission_slot,
    )
    charge_timers = _set_ability(
        arsenal.ability_charge_timer_seconds,
        admission_slot,
        jnp.where(
            accepted & interrupt_recharge,
            jnp.float32(0.0),
            current_charge_timer,
        ),
    )
    scheduler_prelude_ticks = _gather_ability(
        loadout.ability_scheduler_prelude_ticks,
        slot,
    )
    arsenal_candidate = arsenal._replace(
        active_ability_slot=jnp.where(
            accepted,
            slot,
            arsenal.active_ability_slot,
        ),
        active_ability_root_slot=jnp.where(
            accepted,
            admission_slot,
            arsenal.active_ability_root_slot,
        ),
        ability_elapsed_seconds=jnp.where(
            accepted,
            jnp.float32(0.0),
            arsenal.ability_elapsed_seconds,
        ),
        ability_scheduler_tick=jnp.where(
            accepted,
            -scheduler_prelude_ticks,
            arsenal.ability_scheduler_tick,
        ),
        ability_scheduler_clock_seconds=jnp.where(
            accepted[..., None],
            jnp.float32(0.0),
            arsenal.ability_scheduler_clock_seconds,
        ),
        ability_selector_hit_bits=jnp.where(
            accepted,
            jnp.uint32(0),
            arsenal.ability_selector_hit_bits,
        ),
        ability_cooldown_seconds=cooldowns,
        ability_charge_count=charge_counts,
        ability_charge_timer_seconds=charge_timers,
        failure_bits=bits,
    )
    return (
        _select_tree(row_valid, arsenal_candidate, arsenal)._replace(failure_bits=bits),
        _select_tree(row_valid, mechanics_candidate, mechanics),
        requested & row_valid[:, None],
        accepted,
    )


def spend_due_ability_resources(
    arsenal: ArsenalState,
    mechanics: CombatMechanicsState,
    loadout: AbilityLoadout,
    rules: CombatMechanicsRules,
    dt_seconds: jax.Array,
    player_backed_actor_mask: jax.Array | bool = False,
) -> CombatMechanicsState:
    """Apply costs when their authored interaction-graph clock is crossed."""

    ability_capacity = _runtime_ability_capacity(arsenal, loadout)
    batch, entity_count = arsenal.active_ability_slot.shape
    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)
    if dt.ndim == 0:
        dt = jnp.broadcast_to(dt, (batch, entity_count))
    elif dt.shape == (batch,):
        dt = jnp.broadcast_to(dt[:, None], (batch, entity_count))
    if dt.shape != (batch, entity_count):
        raise ValueError("dt_seconds must be scalar, [batch], or [batch, entity]")
    active = arsenal.active_ability_slot >= 0
    slot = jnp.clip(
        arsenal.active_ability_slot,
        0,
        ability_capacity - 1,
    )
    after_clocks = advance_scheduler_clocks(
        active,
        arsenal.ability_scheduler_tick,
        arsenal.ability_scheduler_clock_seconds,
        dt,
    )
    commit_time = _gather_ability(
        loadout.ability_resource_commit_time_seconds,
        slot,
    )
    commit_flags = _gather_ability(
        loadout.ability_resource_commit_flags,
        slot,
    )
    commit_delay = event_dispatch_delay_ticks(
        commit_time,
        commit_flags,
        player_backed_actor_mask,
    )
    prior_clock = scheduler_clock(
        arsenal.ability_scheduler_clock_seconds,
        commit_delay,
    )
    after_clock = scheduler_clock(after_clocks, commit_delay)
    cost_kind = _gather_ability(loadout.ability_resource_cost_kind, slot)
    crossed_commit = event_clock_crossed(
        prior_clock,
        after_clock,
        commit_time,
    )
    due = active[..., None] & (cost_kind != RESOURCE_COST_NONE) & crossed_commit
    cost = effective_resource_cost(
        _gather_ability(loadout.ability_resource_cost, slot),
        cost_kind,
    )
    resources = jnp.clip(
        mechanics.resources - jnp.where(due, cost, jnp.float32(0.0)),
        rules.resource_minimum,
        rules.resource_maximum,
    )
    stamina = resources[..., RESOURCE_STAMINA]
    spent_stamina = due[..., RESOURCE_STAMINA] & (cost[..., RESOURCE_STAMINA] > 0.0)
    stamina_break_immune = (
        cost_kind[..., RESOURCE_STAMINA]
        == RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE
    ) | (
        cost_kind[..., RESOURCE_STAMINA]
        == RESOURCE_COST_SINGLE_APPLICATION_STAMINA_BREAK_IMMUNE
    )
    newly_broken = spent_stamina & ~stamina_break_immune & (stamina <= 0.0)
    broken = mechanics.stamina_broken | newly_broken
    delay = jnp.where(
        newly_broken,
        jnp.minimum(
            mechanics.stamina_regen_delay_seconds,
            jnp.float32(STAMINA_BROKEN_REGEN_DELAY_SECONDS),
        ),
        mechanics.stamina_regen_delay_seconds,
    )
    return mechanics._replace(
        resources=resources,
        stamina_broken=broken,
        guard_held=mechanics.guard_held & ~broken,
        guard_active=mechanics.guard_active & ~broken,
        guard_windup_elapsed_seconds=jnp.where(
            broken,
            jnp.float32(0.0),
            mechanics.guard_windup_elapsed_seconds,
        ),
        stamina_regen_delay_seconds=delay,
    )


def apply_ability_resource_phases(
    arsenal: ArsenalState,
    mechanics: CombatMechanicsState,
    loadout: AbilityLoadout,
) -> CombatMechanicsState:
    """Apply asset-authored outer-root resource values by lifecycle tick.

    The phase bank is fully data-driven over ``[batch, entity, ability,
    resource]``.  It models public roots which temporarily set a resource
    while a selected child is pending (for example a nocked arrow) without
    turning that transient server value into a permanent admission token.
    Durations are already integer 30 TPS boundaries; no continuous rate is
    rounded here.
    """

    shape = mechanics.resources.shape
    if loadout.ability_resource_phase_mask.shape[:2] != shape[:2] or (
        loadout.ability_resource_phase_mask.shape[3] != shape[2]
    ):
        raise ValueError(
            "ability resource phases must share mechanics batch/entity/resource axes"
        )
    active = arsenal.active_ability_slot >= 0
    ability_capacity = loadout.ability_mask.shape[2]
    slot = jnp.clip(
        arsenal.active_ability_slot,
        jnp.int32(0),
        jnp.int32(ability_capacity - 1),
    )

    configured = jnp.any(loadout.ability_resource_phase_mask, axis=2)
    first_slot = jnp.argmax(
        loadout.ability_resource_phase_mask.astype(jnp.int32),
        axis=2,
    )
    inactive = jnp.take_along_axis(
        loadout.ability_resource_phase_inactive_value,
        first_slot[:, :, None, :],
        axis=2,
    )[:, :, 0, :]
    resources = jnp.where(configured, inactive, mechanics.resources)

    phase_mask = _gather_ability(
        loadout.ability_resource_phase_mask,
        slot,
    )
    phase_value = _gather_ability(
        loadout.ability_resource_phase_value,
        slot,
    )
    start_tick = _gather_ability(
        loadout.ability_resource_phase_start_tick,
        slot,
    )
    end_tick = _gather_ability(
        loadout.ability_resource_phase_end_tick,
        slot,
    )
    prelude = _gather_ability(
        loadout.ability_scheduler_prelude_ticks,
        slot,
    )
    lifecycle_tick = arsenal.ability_scheduler_tick + prelude
    in_phase = (
        active[..., None]
        & phase_mask
        & (lifecycle_tick[..., None] >= start_tick)
        & (lifecycle_tick[..., None] < end_tick)
    )
    resources = jnp.where(in_phase, phase_value, resources)
    return mechanics._replace(resources=resources)


def apply_ability_stamina_regen_delay_phases(
    prior: ArsenalState,
    mechanics: CombatMechanicsState,
    loadout: AbilityLoadout,
) -> CombatMechanicsState:
    """Apply timed ChangeStat(Set) nodes on the fixed lifecycle clock.

    Ordinary abilities retain admission-time delay assignment through a zero
    start tick. Some public roots place the Set node later in their graph and
    explicitly clear it from a final child's ``Next`` node. Those boundaries
    are represented here without keeping a weapon-specific runtime branch.

    ``prior`` is the active state at the start of the fixed interaction tick.
    This matters when an item node fails during the same tick: a preceding Set
    node still executes, while its later clear node remains unreachable.
    """

    active = prior.active_ability_slot >= 0
    ability_capacity = loadout.ability_mask.shape[2]
    slot = jnp.clip(
        prior.active_ability_slot,
        jnp.int32(0),
        jnp.int32(ability_capacity - 1),
    )
    delay_value = _gather_ability(
        loadout.ability_stamina_regen_delay_seconds,
        slot,
    )
    start_tick = _gather_ability(
        loadout.ability_stamina_regen_delay_start_tick,
        slot,
    )
    end_tick = _gather_ability(
        loadout.ability_stamina_regen_delay_end_tick,
        slot,
    )
    prelude = _gather_ability(
        loadout.ability_scheduler_prelude_ticks,
        slot,
    )
    lifecycle_prior = prior.ability_scheduler_tick + prelude
    lifecycle_after = lifecycle_prior + jnp.int32(1)
    start_due = (
        active
        & (delay_value < 0.0)
        & (start_tick > 0)
        & (lifecycle_prior < start_tick)
        & (lifecycle_after >= start_tick)
    )
    end_due = (
        active
        & (end_tick >= 0)
        & (lifecycle_prior < end_tick)
        & (lifecycle_after >= end_tick)
    )
    delay = jnp.where(
        start_due,
        jnp.minimum(mechanics.stamina_regen_delay_seconds, delay_value),
        mechanics.stamina_regen_delay_seconds,
    )
    # The later Set(0) is ordered after the initial Set if malformed data ever
    # puts both on one tick; validation otherwise requires end > start.
    delay = jnp.where(end_due, jnp.float32(0.0), delay)
    return mechanics._replace(stamina_regen_delay_seconds=delay)


def effective_resource_cost(
    authored_cost: jax.Array,
    cost_kind: jax.Array,
) -> jax.Array:
    """Resolve authored resource costs for the native NPC execution path."""

    applications = jnp.where(
        (
            (cost_kind == RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT)
            | (
                cost_kind
                == RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE
            )
        ),
        jnp.int32(NATIVE_NPC_INSTANT_ADD_STAT_APPLICATIONS),
        jnp.where(
            (cost_kind == RESOURCE_COST_SINGLE_APPLICATION)
            | (cost_kind == RESOURCE_COST_SINGLE_APPLICATION_STAMINA_BREAK_IMMUNE),
            jnp.int32(1),
            jnp.int32(0),
        ),
    )
    return authored_cost * applications.astype(authored_cost.dtype)


def tick_ability_programs(
    state: ArsenalState,
    loadout: AbilityLoadout,
    dt_seconds: jax.Array,
    cooldown_dt_seconds: jax.Array | None = None,
    player_backed_actor_mask: jax.Array | bool = False,
    entity_grounded: jax.Array | None = None,
    entity_collided: jax.Array | None = None,
) -> tuple[ArsenalState, FiredEvents]:
    """Advance fixed-tick interactions and actual-delta cooldown clocks.

    Hytale's ``InteractionManager`` advances operation timestamps with the
    World's fixed ``tickStepNanos``. Its ``CooldownHandler`` independently
    consumes the engine-supplied float ``dt``. ``cooldown_dt_seconds``
    preserves that native two-clock boundary during measured-delta replay.
    ``player_backed_actor_mask`` adds only the client-sourced selector
    request/response boundaries; ordinary role-only callers retain the former
    scheduler.

    ``entity_grounded``/``entity_collided`` carry the previous tick's per-entity
    motion for C4 Continuation. Omitting them means no awaited condition can
    ever fire, so a continuation root would hold until its authored ``RunTime``;
    that is the correct reading for callers with no world attached, and is inert
    for every ability that authors no continuation.
    """

    ability_capacity = _runtime_ability_capacity(state, loadout)
    batch, entity_count = state.active_ability_slot.shape
    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)
    if dt.ndim == 0:
        dt_entity = jnp.broadcast_to(dt, (batch, entity_count))
    elif dt.shape == (batch,):
        dt_entity = jnp.broadcast_to(dt[:, None], (batch, entity_count))
    else:
        dt_entity = dt
    if dt_entity.shape != (batch, entity_count):
        raise ValueError("dt_seconds must be scalar, [batch], or [batch, entity]")
    cooldown_dt = (
        dt_entity
        if cooldown_dt_seconds is None
        else jnp.asarray(cooldown_dt_seconds, dtype=jnp.float32)
    )
    if cooldown_dt.ndim == 0:
        cooldown_dt = jnp.broadcast_to(
            cooldown_dt,
            (batch, entity_count),
        )
    elif cooldown_dt.shape == (batch,):
        cooldown_dt = jnp.broadcast_to(
            cooldown_dt[:, None],
            (batch, entity_count),
        )
    if cooldown_dt.shape != (batch, entity_count):
        raise ValueError(
            "cooldown_dt_seconds must be scalar, [batch], or [batch, entity]"
        )
    player_backed = jnp.asarray(player_backed_actor_mask, dtype=jnp.bool_)
    if player_backed.shape == ():
        player_backed = jnp.broadcast_to(
            player_backed,
            (batch, entity_count),
        )
    if player_backed.shape != (batch, entity_count):
        raise ValueError(
            "player_backed_actor_mask must be scalar or have shape [batch, entity]"
        )
    active = state.active_ability_slot >= 0
    slot = jnp.clip(state.active_ability_slot, 0, ability_capacity - 1)
    scheduler_tick = state.ability_scheduler_tick
    scheduler_prelude_ticks = _gather_ability(
        loadout.ability_scheduler_prelude_ticks,
        slot,
    )
    lifecycle_tick = scheduler_tick + scheduler_prelude_ticks
    lifecycle_after = state.ability_elapsed_seconds + jnp.where(
        active & (lifecycle_tick >= jnp.int32(INTERACTION_QUEUE_DELAY_TICKS)),
        dt_entity,
        jnp.float32(0.0),
    )
    prior_clocks = state.ability_scheduler_clock_seconds
    after_clocks = advance_scheduler_clocks(
        active,
        scheduler_tick,
        prior_clocks,
        dt_entity,
    )
    (
        event_mask,
        event_time,
        event_kind,
        event_flags,
        payload_index,
        local_event_index,
    ) = _active_event_schedule(loadout, slot)
    schedule_event_flags = event_flags
    event_delay = event_dispatch_delay_ticks(
        event_time,
        event_flags,
        player_backed,
    )
    event_prior = scheduler_clock(prior_clocks, event_delay)
    event_after = scheduler_clock(after_clocks, event_delay)
    event_f32_wide = _gather_event_payload(
        loadout,
        loadout.event_f32,
        slot,
        payload_index,
    )
    selector_runtime = event_f32_wide[..., EF_SELECTOR_RUNTIME_SECONDS]
    progressive_selector_wide = event_uses_progressive_selector(event_flags)
    selector_end = event_time + selector_runtime
    selector_progress_denominator = jnp.maximum(
        selector_runtime,
        jnp.float32(1.0e-9),
    )
    selector_previous_progress_wide = jnp.clip(
        (event_prior - event_time) / selector_progress_denominator,
        jnp.float32(0.0),
        jnp.float32(1.0),
    )
    selector_current_progress_wide = jnp.clip(
        (event_after - event_time) / selector_progress_denominator,
        jnp.float32(0.0),
        jnp.float32(1.0),
    )
    selector_event_bit = jnp.left_shift(
        jnp.uint32(1),
        local_event_index.astype(jnp.uint32),
    )
    selector_not_hit = (
        state.ability_selector_hit_bits[..., None] & selector_event_bit
    ) == jnp.uint32(0)
    selector_active = (
        progressive_selector_wide
        & (selector_runtime > jnp.float32(0.0))
        & (event_after > event_time)
        & (event_prior < selector_end)
        & selector_not_hit
    )
    one_shot_event = ~progressive_selector_wide & event_clock_crossed(
        event_prior, event_after, event_time
    )
    fires_wide = (
        active[:, :, None]
        & event_mask
        & (one_shot_event | selector_active)
        & (state.failure_bits == 0)[:, None, None]
    )
    event_overflow = jnp.sum(fires_wide, axis=2) > FIRED_EVENT_CAPACITY
    bits = _set_failure(
        state.failure_bits,
        jnp.any(event_overflow, axis=1),
        ARSENAL_FAILURE_EVENT_OVERFLOW,
    )
    (
        fires,
        compact_event_index,
        selected,
    ) = _compact_fired_events(fires_wide, payload_index)
    compact_local_event_index = _compact_fired_values(
        selected,
        local_event_index,
    ).astype(jnp.int32)
    selector_previous_progress = _compact_fired_values(
        selected,
        selector_previous_progress_wide,
    )
    selector_current_progress = _compact_fired_values(
        selected,
        selector_current_progress_wide,
    )
    event_kind = _gather_event_payload(
        loadout,
        loadout.event_kind,
        slot,
        compact_event_index,
    )
    event_f32 = _gather_event_payload(
        loadout,
        loadout.event_f32,
        slot,
        compact_event_index,
    )
    event_i32 = _gather_event_payload(
        loadout,
        loadout.event_i32,
        slot,
        compact_event_index,
    )
    event_flags = _gather_event_payload(
        loadout,
        loadout.event_flags,
        slot,
        compact_event_index,
    )
    progressive_selector = event_uses_progressive_selector(event_flags)
    ability_requirements = _gather_ability(
        loadout.ability_requirements,
        slot,
    )
    event_requirements = jnp.where(
        fires,
        ability_requirements[..., None],
        jnp.uint32(0),
    )
    fires &= (bits == 0)[:, None, None]
    duration = _gather_ability(loadout.ability_duration_seconds, slot)
    resource_cost_kind = _gather_ability(
        loadout.ability_resource_cost_kind,
        slot,
    )
    resource_commit_time = _gather_ability(
        loadout.ability_resource_commit_time_seconds,
        slot,
    )
    has_resource_cost = resource_cost_kind != RESOURCE_COST_NONE
    duration = jnp.maximum(
        duration,
        jnp.max(
            jnp.where(
                has_resource_cost,
                resource_commit_time,
                jnp.float32(0.0),
            ),
            axis=2,
        ),
    )
    # Ability duration belongs to the selected interaction root. Child clocks
    # do not extend an already-complete root merely because they started later,
    # but the root also cannot disappear before its authored child work has
    # received an executable scheduler sample. This distinction is observable:
    # Iron Sword's selector fires well before the root ends, whereas the
    # Root/Stoneskin selector is the terminal child of a short parent root.
    # Keep the rule graph-shaped and data-driven rather than branching on a
    # weapon or event kind.
    root_delay = jnp.full_like(
        slot,
        jnp.int32(INTERACTION_QUEUE_DELAY_TICKS),
    )
    root_prior = scheduler_clock(prior_clocks, root_delay)
    root_after = scheduler_clock(after_clocks, root_delay)
    terminal_equality = jnp.any(
        event_mask
        & (
            (
                schedule_event_flags
                & jnp.uint32(EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA)
            )
            != 0
        )
        & (
            jnp.abs(event_time - duration[..., None]) <= SCHEDULER_CLOCK_EPSILON_SECONDS
        ),
        axis=2,
    )
    # A typed terminal Instant Projectile is the final child of its Serial
    # root. When its timestamp equals the root duration it fires and completes
    # the root on the same scheduler sample; ordinary roots retain the prior-
    # clock completion rule.
    root_completion_clock = jnp.where(
        terminal_equality,
        root_after,
        root_prior,
    )
    event_terminal_time = jnp.where(
        progressive_selector_wide,
        selector_end,
        event_time,
    )
    event_work_complete = jnp.all(
        ~event_mask
        | scheduler_clock_reached(
            event_after,
            event_terminal_time,
        ),
        axis=2,
    )
    resource_commit_flags = _gather_ability(
        loadout.ability_resource_commit_flags,
        slot,
    )
    resource_commit_delay = event_dispatch_delay_ticks(
        resource_commit_time,
        resource_commit_flags,
        player_backed,
    )
    resource_after = scheduler_clock(
        after_clocks,
        resource_commit_delay,
    )
    resource_work_complete = jnp.all(
        ~has_resource_cost
        | scheduler_clock_reached(
            resource_after,
            resource_commit_time,
        ),
        axis=2,
    )
    stamina_delay_end_tick = _gather_ability(
        loadout.ability_stamina_regen_delay_end_tick,
        slot,
    )
    stamina_delay_work_complete = (stamina_delay_end_tick < 0) | (
        scheduler_tick + scheduler_prelude_ticks + jnp.int32(1)
        >= stamina_delay_end_tick
    )
    finished = (
        active
        & (root_completion_clock + SCHEDULER_CLOCK_EPSILON_SECONDS >= duration)
        & event_work_complete
        & resource_work_complete
        # A terminal ChangeStat node is part of the interaction graph, not a
        # detached timer.  Keep the root alive through that fixed boundary so
        # the phase executor can observe and apply its final Set in the same
        # tick.  Failed item roots are already removed before this function
        # and therefore cannot incorrectly reach their missing Next node.
        & stamina_delay_work_complete
    )
    # C4 Continuation. An authored WaitForGround/WaitForCollision root does not
    # end when its own clock does: it parks until the world reports the awaited
    # condition, then runs its authored child. Gating `finished` here is the
    # whole hold -- every other clock above has already completed, which is
    # exactly the moment native hands control to the wait.
    #
    # `ability_continuation_mode` is 0 for every ability currently authored, so
    # `parked` is uniformly False and this stage is a no-op until the three
    # deferred world-conditional roots land.
    continuation_mode = _gather_ability(loadout.ability_continuation_mode, slot)
    continuation_waited = advance_continuation_clock(
        state.ability_continuation_seconds,
        (continuation_mode != jnp.int32(CONTINUATION_NONE)) & finished,
        dt_entity,
    )
    continuation_resume, continuation_slot, continuation_expired = (
        continuation_resolution(
            mode=continuation_mode,
            ground_slot=_gather_ability(
                loadout.ability_continuation_ground_slot,
                slot,
            ),
            collision_slot=_gather_ability(
                loadout.ability_continuation_collision_slot,
                slot,
            ),
            ground_check_delay_seconds=_gather_ability(
                loadout.ability_continuation_ground_check_delay_seconds,
                slot,
            ),
            run_time_seconds=_gather_ability(
                loadout.ability_continuation_run_time_seconds,
                slot,
            ),
            waited_seconds=continuation_waited,
            grounded=(
                jnp.zeros_like(active)
                if entity_grounded is None
                else jnp.broadcast_to(
                    jnp.asarray(entity_grounded, jnp.bool_),
                    active.shape,
                )
            ),
            collided=(
                jnp.zeros_like(active)
                if entity_collided is None
                else jnp.broadcast_to(
                    jnp.asarray(entity_collided, jnp.bool_),
                    active.shape,
                )
            ),
        )
    )
    waiting = (
        (continuation_mode != jnp.int32(CONTINUATION_NONE))
        & finished
        & ~continuation_resume
        & ~continuation_expired
    )
    finished = finished & ~waiting
    continuation_waited = jnp.where(
        waiting,
        continuation_waited,
        jnp.zeros_like(continuation_waited),
    )
    continuation_slot = jnp.where(waiting, jnp.int32(-1), continuation_slot)
    cooldowns = jnp.maximum(
        jnp.float32(0.0),
        state.ability_cooldown_seconds - cooldown_dt[..., None],
    )
    charge_capacity = loadout.ability_charge_capacity
    charge_count = state.ability_charge_count
    recharging = charge_count < charge_capacity
    charge_index = jnp.clip(
        charge_count,
        jnp.int32(0),
        jnp.int32(ABILITY_CHARGE_CAPACITY - 1),
    )
    charge_time = jnp.take_along_axis(
        loadout.ability_charge_times_seconds,
        charge_index[..., None],
        axis=3,
    )[..., 0]
    charge_timer = state.ability_charge_timer_seconds + jnp.where(
        recharging,
        cooldown_dt[..., None],
        jnp.float32(0.0),
    )
    charge_replenished = recharging & (charge_timer >= charge_time)
    charge_count = jnp.where(
        charge_replenished,
        charge_count + jnp.int32(1),
        charge_count,
    )
    charge_timer = jnp.where(
        charge_replenished,
        jnp.float32(0.0),
        charge_timer,
    )
    candidate = state._replace(
        active_ability_slot=jnp.where(
            finished,
            jnp.int32(-1),
            state.active_ability_slot,
        ),
        active_ability_root_slot=jnp.where(
            finished,
            jnp.int32(-1),
            state.active_ability_root_slot,
        ),
        ability_elapsed_seconds=jnp.where(
            finished | ~active,
            jnp.float32(0.0),
            lifecycle_after,
        ),
        ability_continuation_slot=continuation_slot,
        ability_continuation_seconds=continuation_waited,
        ability_scheduler_tick=jnp.where(
            finished | ~active,
            jnp.int32(0),
            scheduler_tick + jnp.int32(1),
        ),
        ability_scheduler_clock_seconds=jnp.where(
            (finished | ~active)[..., None],
            jnp.float32(0.0),
            after_clocks,
        ),
        ability_selector_hit_bits=jnp.where(
            finished | ~active,
            jnp.uint32(0),
            state.ability_selector_hit_bits,
        ),
        ability_cooldown_seconds=cooldowns,
        ability_charge_count=charge_count,
        ability_charge_timer_seconds=charge_timer,
        failure_bits=bits,
    )
    source = jnp.broadcast_to(
        jnp.arange(entity_count, dtype=jnp.int32)[None, :, None],
        fires.shape,
    )
    return candidate, FiredEvents(
        requested=fires,
        source_entity_id=source,
        ability_event_index=jnp.where(
            fires,
            compact_local_event_index,
            jnp.int32(-1),
        ),
        selector_previous_progress=jnp.where(
            fires & progressive_selector,
            selector_previous_progress,
            jnp.float32(0.0),
        ),
        selector_current_progress=jnp.where(
            fires & progressive_selector,
            selector_current_progress,
            jnp.float32(0.0),
        ),
        kind=event_kind,
        f32=event_f32,
        i32=event_i32,
        flags=event_flags,
        requirements=event_requirements,
    )


def _active_event_schedule(
    loadout: AbilityLoadout,
    slot: jax.Array,
) -> tuple[jax.Array, ...]:
    if loadout.event_mask.ndim == 4:
        event_count = loadout.event_mask.shape[3]
        payload_index = jnp.broadcast_to(
            jnp.arange(event_count, dtype=jnp.int32),
            slot.shape + (event_count,),
        )
        return (
            _gather_ability(loadout.event_mask, slot),
            _gather_ability(loadout.event_time_seconds, slot),
            _gather_ability(loadout.event_kind, slot),
            _gather_ability(loadout.event_flags, slot),
            payload_index,
            payload_index,
        )

    start = _gather_ability(loadout.ability_event_start, slot)
    count = _gather_ability(loadout.ability_event_count, slot)
    local_mask = _gather_ability(loadout.ability_event_mask, slot)
    event_index = jnp.arange(
        local_mask.shape[2],
        dtype=jnp.int32,
    )
    active_range = event_index < count[..., None]
    packed_index = jnp.clip(
        start[..., None] + event_index,
        jnp.int32(0),
        jnp.int32(loadout.event_mask.shape[2] - 1),
    )
    return (
        _gather_packed_events(loadout.event_mask, packed_index)
        & local_mask
        & active_range,
        _gather_packed_events(
            loadout.event_time_seconds,
            packed_index,
        ),
        _gather_packed_events(loadout.event_kind, packed_index),
        _gather_packed_events(loadout.event_flags, packed_index),
        packed_index,
        jnp.broadcast_to(
            event_index,
            slot.shape + (local_mask.shape[2],),
        ),
    )


def _compact_fired_events(
    requested: jax.Array,
    payload_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rank = jnp.cumsum(requested, axis=2) - jnp.int32(1)
    output_slot = jnp.arange(
        FIRED_EVENT_CAPACITY,
        dtype=jnp.int32,
    )
    selected = requested[..., None] & (rank[..., None] == output_slot)
    compact_requested = jnp.any(selected, axis=2)
    compact_index = jnp.sum(
        jnp.where(
            selected,
            payload_index[..., None],
            jnp.int32(0),
        ),
        axis=2,
    )
    return compact_requested, compact_index, selected


def _compact_fired_values(
    selected: jax.Array,
    values: jax.Array,
) -> jax.Array:
    return jnp.sum(
        jnp.where(
            selected,
            values[..., None],
            jnp.zeros((), dtype=values.dtype),
        ),
        axis=2,
    )


def _gather_event_payload(
    loadout: AbilityLoadout,
    values: jax.Array,
    ability_slot: jax.Array,
    event_index: jax.Array,
) -> jax.Array:
    if loadout.event_mask.ndim == 4:
        event_capacity = values.shape[3]
        flattened = values.reshape(
            values.shape[:2] + (values.shape[2] * event_capacity,) + values.shape[4:]
        )
        absolute_index = ability_slot[..., None] * event_capacity + event_index
        return _gather_packed_events(flattened, absolute_index)
    return _gather_packed_events(values, event_index)


def _gather_packed_events(
    values: jax.Array,
    index: jax.Array,
) -> jax.Array:
    trailing = values.shape[3:]
    expanded = index.reshape(index.shape + (1,) * len(trailing))
    expanded = jnp.broadcast_to(
        expanded,
        index.shape + trailing,
    )
    return jnp.take_along_axis(values, expanded, axis=2)


def _requirements_available(
    requirements: jax.Array,
    world: ArsenalWorldCapabilities,
) -> jax.Array:
    """Return whether each authored world predicate is currently satisfied."""

    def available(bit: int, value: jax.Array) -> jax.Array:
        return ((requirements & jnp.uint32(bit)) == 0) | value[:, :, None]

    force_capacity = requirements.shape[2]
    force_available = (
        world.clear_force_path[:, :, :force_capacity]
        | world.applied_force_collision_available[:, :, :force_capacity]
    )

    finite_center = jnp.all(jnp.isfinite(world.area_center), axis=2)
    return (
        available(
            REQUIRE_CLEAR_PROJECTILE_FLIGHT,
            world.clear_projectile_flight
            & world.muzzle_valid
            & jnp.all(jnp.isfinite(world.muzzle_position), axis=2)
            & jnp.isfinite(world.muzzle_yaw_degrees)
            & jnp.isfinite(world.muzzle_pitch_degrees),
        )
        & (
            ((requirements & jnp.uint32(REQUIRE_CLEAR_FORCE_PATH)) == 0)
            | force_available
        )
        & available(REQUIRE_ENTITY_ONLY_AREA, world.entity_only_area)
        & available(
            REQUIRE_STATIC_AREA_PLACEMENT,
            world.static_area_placement & finite_center,
        )
        & available(
            REQUIRE_LINE_OF_SIGHT,
            world.selector_line_of_sight_valid,
        )
        & available(
            REQUIRE_WORLD_PROJECTILE_COLLISION,
            world.projectile_world_collision_available,
        )
        & available(
            REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE,
            world.deployable_intended_graph_available,
        )
    )


def _requirements_evidence_available(
    requirements: jax.Array,
    world: ArsenalWorldCapabilities,
) -> jax.Array:
    """Return whether required world evidence is known, independent of outcome.

    A certified obstruction is ordinary gameplay.  In particular,
    ``clear_projectile_flight == False`` is a known negative result when the
    projectile-contact query is available.  It must not set
    ``ARSENAL_FAILURE_UNSUPPORTED_WORLD``: the policy mask may avoid a
    currently blocked shot, while a path obstructed after a delayed ability
    starts resolves as an ordinary terrain contact.
    """

    projectile_transform_available = (
        world.muzzle_valid
        & jnp.all(jnp.isfinite(world.muzzle_position), axis=2)
        & jnp.isfinite(world.muzzle_yaw_degrees)
        & jnp.isfinite(world.muzzle_pitch_degrees)
    )
    projectile_query_available = projectile_transform_available & (
        world.clear_projectile_flight | world.projectile_world_collision_available
    )
    clear_projectile_required = (
        requirements & jnp.uint32(REQUIRE_CLEAR_PROJECTILE_FLIGHT)
    ) != 0
    world_collision_required = (
        requirements & jnp.uint32(REQUIRE_WORLD_PROJECTILE_COLLISION)
    ) != 0
    contact_chain_required = (
        requirements & jnp.uint32(REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE)
    ) != 0
    projectile_evidence_available = (
        ((~clear_projectile_required) | projectile_query_available[:, :, None])
        & (
            (~world_collision_required)
            | (
                projectile_transform_available
                & world.projectile_world_collision_available
            )[:, :, None]
        )
        & (
            (~contact_chain_required)
            | world.deployable_intended_graph_available[:, :, None]
        )
    )

    # The remaining predicates do not yet expose distinct availability and
    # outcome channels.  Retain their fail-closed behaviour until their public
    # World contracts publish that distinction.
    non_projectile_requirements = requirements & ~jnp.uint32(
        REQUIRE_CLEAR_PROJECTILE_FLIGHT
        | REQUIRE_WORLD_PROJECTILE_COLLISION
        | REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE
    )
    return projectile_evidence_available & _requirements_available(
        non_projectile_requirements,
        world,
    )


def _gather_ability(values: jax.Array, slot: jax.Array) -> jax.Array:
    trailing = values.shape[3:]
    index = slot[:, :, None].reshape(slot.shape + (1,) * (values.ndim - 2))
    index = jnp.broadcast_to(index, slot.shape + (1,) + trailing)
    return jnp.take_along_axis(values, index, axis=2)[:, :, 0]


def _set_ability(
    values: jax.Array,
    slot: jax.Array,
    replacement: jax.Array,
) -> jax.Array:
    one_hot = jax.nn.one_hot(
        slot,
        values.shape[2],
        dtype=jnp.bool_,
    )
    return jnp.where(one_hot, replacement[:, :, None], values)


def _set_failure(bits: jax.Array, mask: jax.Array, code: int) -> jax.Array:
    return jnp.where(mask, bits | jnp.uint32(code), bits)


def _runtime_ability_capacity(
    arsenal: ArsenalState,
    loadout: AbilityLoadout,
) -> int:
    capacity = loadout.ability_mask.shape[2]
    state_capacity = arsenal.ability_cooldown_seconds.shape[2]
    if state_capacity != capacity:
        raise ValueError(
            "arsenal state/loadout ability capacity mismatch: "
            f"{state_capacity} != {capacity}"
        )
    if arsenal.ability_charge_count.shape != arsenal.ability_cooldown_seconds.shape:
        raise ValueError("arsenal charge-count shape must match cooldown shape")
    if (
        arsenal.ability_charge_timer_seconds.shape
        != arsenal.ability_cooldown_seconds.shape
    ):
        raise ValueError("arsenal charge-timer shape must match cooldown shape")
    return capacity


def _select_tree(mask: jax.Array, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
