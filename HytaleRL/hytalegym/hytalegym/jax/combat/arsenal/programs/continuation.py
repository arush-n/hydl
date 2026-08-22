"""C4 Continuation: park an ability until the world lets it resume.

An authored ``ApplyForce`` node may carry ``WaitForGround`` / ``WaitForCollision``
instead of ending when its own clock does. The ability launches the actor, then
waits; when the awaited condition fires the authored ``Next`` chain runs, which
is what actually lands the hit. All three deferred world-conditional roots --
Mace Groundslam, Battleaxe Downstrike and Daggers Pounce -- are gated on this
one mechanism, so the law lives here once rather than per weapon.

Two authored guards keep the wait honest:

``GroundCheckDelay``
    The launch leaves the actor standing for a tick or two, so an immediate
    ground test would resume the continuation before the actor ever left the
    floor. Pounce authors 0.3 s; Downstrike authors none.

``RunTime``
    A ceiling on the wait. Pounce authors 10.0 s. Without it an actor that never
    lands -- launched over a void, or wedged -- would hold its ability open for
    the rest of the episode.

The condition inputs are read from the *previous* tick's per-entity motion, so a
resume trails the physical landing by one tick. That lag is deliberate: the
motion integrator that decides `grounded` runs after ability execution within a
tick, and reading it forward would make the result depend on execution order.
"""

from __future__ import annotations

import jax.numpy as jnp


#: ``ability_continuation_mode`` bits. Zero means the ability is not a
#: continuation and finishes on its own clock.
CONTINUATION_NONE = 0
CONTINUATION_WAIT_FOR_GROUND = 1 << 0
CONTINUATION_WAIT_FOR_COLLISION = 1 << 1

#: Radius of the ``WaitForCollision`` probe, in blocks. **Hardcoded in the
#: server**, not authored: ``ApplyForceInteraction`` carries a private
#: ``SPATIAL_STRUCTURE_RADIUS = 1.5F`` and ``simulateTick0`` passes the literal
#: ``1.5`` to ``spatialStructure.collect``. The authored ``RaycastDistance`` and
#: ``RaycastHeightOffset`` fields look like they should decide this and do not:
#: they are only copied onto the outgoing client packet. Pounce authors 3.0 and
#: 0.5 for those, so a build that used them would test twice the real radius.
CONTINUATION_CONTACT_RADIUS_BLOCKS = 1.5


def continuation_contact(positions, present, *, radius=CONTINUATION_CONTACT_RADIUS_BLOCKS):
    """Whether each actor has another entity inside the C4 contact probe.

    Mirrors::

        spatialStructure.collect(transform.getPosition(), 1.5, entities);
        collided = checkCollision && waitForCollision && entities.size() > 1;

    ``positions`` is ``[B, N, 3]`` and ``present`` is ``[B, N]``. Returns
    ``[B, N]``.

    Three details the Java carries that a naive "is the target near me" test
    would miss:

    * **The probe is centred on ``TransformComponent.getPosition()``** -- the
      entity origin, its feet. It adds no eye height and applies no yaw-rotated
      offset, unlike ``AOECircleSelector``, which does both.
    * **The collect returns the acting entity itself**, which is why the server
      compares ``size() > 1`` rather than ``> 0``. Counting only *other*
      entities, as here, is the same predicate with the self term removed --
      but a build that reuses a raw spatial count must keep the ``> 1``.
    * **It is a sphere.** ``KDTree.collect`` tests squared distance on all three
      axes, so an opponent directly below a leaping actor counts.

    The structure queried is ``getNetworkSendableSpatialResourceType()`` -- one
    structure, unlike ``Selector.selectNearbyEntities`` which unions the player,
    entity and item structures. Items on the ground therefore do not end a
    Pounce, so ``present`` should carry actors, not dropped items.
    """

    position = jnp.asarray(positions, dtype=jnp.float32)
    alive = jnp.asarray(present, dtype=jnp.bool_)
    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError("positions must have shape (B, N, 3)")
    if alive.shape != position.shape[:2]:
        raise ValueError("present must have shape (B, N)")

    offset = position[:, :, None, :] - position[:, None, :, :]
    distance_squared = jnp.sum(offset * offset, axis=-1)
    entity_count = position.shape[1]
    not_self = ~jnp.eye(entity_count, dtype=jnp.bool_)[None, :, :]
    inside = distance_squared <= jnp.float32(radius) * jnp.float32(radius)
    # Both ends must be real: a dead or absent row is not in the structure, and
    # an absent actor cannot be the one doing the colliding.
    pair = inside & not_self & alive[:, None, :] & alive[:, :, None]
    return jnp.any(pair, axis=-1)


def continuation_resolution(
    *,
    mode,
    ground_slot,
    collision_slot,
    ground_check_delay_seconds,
    run_time_seconds,
    waited_seconds,
    grounded,
    collided,
    collision_check_delay_seconds=0.0,
    timer_slot=-1,
):
    """Decide what one waiting ability does this tick.

    Mirrors ``ApplyForceInteraction.simulateTick0`` in the decompiled server,
    which is a single if/else chain resolving **at most one exit per tick** in a
    fixed order::

        if      onGround      -> context.jump(label(1))    # ground child
        else if collided      -> context.jump(label(2))    # collision child
        else if timerFinished -> context.jump(label(0))    # timer child

    Returns ``(resume, resume_slot, expired)``:

    * ``resume`` -- an exit fired and the authored child named by ``resume_slot``
      should run. ``resume_slot`` is -1 when the author wants the ability to
      simply end.
    * ``expired`` -- the timer exit fired. It is reported separately because it
      is the one exit that may legitimately have no child.

    **Ground beats collision.** An actor that lands on a surface next to an
    opponent is both grounded and in contact on the same tick; the server takes
    the ground branch and never evaluates the collision one. Pounce points its
    ground exit at Sweep and its collision exit at Stab, so getting this
    backwards picks the wrong attack on exactly the tick a pouncing dagger hits
    most often.
    """

    mode = jnp.asarray(mode, dtype=jnp.int32)
    waited = jnp.asarray(waited_seconds, dtype=jnp.float32)
    delay = jnp.asarray(ground_check_delay_seconds, dtype=jnp.float32)
    collision_delay = jnp.asarray(collision_check_delay_seconds, dtype=jnp.float32)
    run_time = jnp.asarray(run_time_seconds, dtype=jnp.float32)

    wants_ground = (mode & jnp.int32(CONTINUATION_WAIT_FOR_GROUND)) != 0
    wants_collision = (mode & jnp.int32(CONTINUATION_WAIT_FOR_COLLISION)) != 0

    # `GroundCheckDelay` and `CollisionCheckDelay` are *separate* authored
    # fields (`ApplyForceInteraction.java:128` vs the ground one). Pounce
    # authors 0.3 for ground and leaves collision at the codec default of 0.0,
    # so reusing the ground delay here would blind the collision exit for its
    # first 0.3 s -- the exact window a pounce crosses the gap in.
    ground_ready = wants_ground & (waited >= delay) & jnp.asarray(grounded, jnp.bool_)
    collision_ready = (
        wants_collision
        & (waited >= collision_delay)
        & jnp.asarray(collided, jnp.bool_)
    )

    # `instantlyComplete = runTime <= 0 && !waitForGround && !waitForCollision`,
    # and `timerFinished = instantlyComplete || (runTime > 0 && time >= runTime)`.
    # Note the strict `> 0`: an ability that authors `RunTime: 0` *and* a wait
    # never times out on the server, because neither disjunct can be true. The
    # earlier `run_time >= 0` reading expired such an ability on its first tick,
    # which is the opposite behaviour. A negative RunTime is "no ceiling".
    waits = wants_ground | wants_collision
    instantly_complete = (run_time <= jnp.float32(0.0)) & ~waits
    timer_finished = instantly_complete | (
        (run_time > jnp.float32(0.0)) & (waited >= run_time)
    )
    timer_ready = (mode != jnp.int32(CONTINUATION_NONE)) & timer_finished

    # One exit per tick, in the authored order.
    fired = ground_ready | collision_ready | timer_ready
    resume_slot = jnp.where(
        ground_ready,
        jnp.asarray(ground_slot, dtype=jnp.int32),
        jnp.where(
            collision_ready,
            jnp.asarray(collision_slot, dtype=jnp.int32),
            jnp.asarray(timer_slot, dtype=jnp.int32),
        ),
    )
    resume_slot = jnp.where(fired, resume_slot, jnp.int32(-1))
    # `resume` means "a child runs"; the timer exit only resumes when the author
    # gave label(0) a child. Pounce does -- it points its timer at the same
    # Sweep node as its ground exit -- but an ability may leave it empty.
    resume = fired & (resume_slot >= jnp.int32(0))
    expired = timer_ready & ~ground_ready & ~collision_ready
    return resume, resume_slot, expired


def advance_continuation_clock(waited_seconds, waiting, dt_seconds):
    """Accumulate the wait clock for the lanes that are still parked."""

    waited = jnp.asarray(waited_seconds, dtype=jnp.float32)
    return jnp.where(
        jnp.asarray(waiting, jnp.bool_),
        waited + jnp.asarray(dt_seconds, dtype=jnp.float32),
        jnp.zeros_like(waited),
    )
