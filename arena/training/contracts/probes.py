"""Deterministic policy-comparison probes for the arena skill contracts.

    python -m arena.training.contracts.probes

Each probe enumerates candidate policies -- including the degenerate ones the
reward law is supposed to defeat -- runs them over a full horizon against a
fixed scripted opponent, and reports cumulative return. The intended policy
must finish first, and the probe fails loudly if it does not.

This exists because a single-tick sign check cannot catch a law that is
exploitable *in aggregate*. ``punish_window`` passed every per-tick check and
still scored "attack whenever the opponent is not swinging" at more than twice
the intended policy, because it credited damage in the long neutral region as
well as the short recovery window. Only summing a whole episode revealed it.

Run this after changing any weight. The policies here are deliberately dumb and
deterministic: no learning, no sampling, so a failure is always a statement
about the reward law rather than about optimisation.
"""

import jax.numpy as jnp

from arena.training import contracts as C

HORIZON = 512
ATTACK_PERIOD = 32     # opponent attacks every 32 ticks
THREAT_WINDOW = 6      # threat is telegraphed for 6 ticks before it resolves
DAMAGE = 4.0


def lane_f32(value):
    return jnp.float32(value).reshape(1)


def lane_bool(value):
    return jnp.bool_(value).reshape(1)


def lane_i32(value):
    return jnp.int32(value).reshape(1)


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def report(rows, winner_key):
    width = max(len(name) for name, _ in rows)
    ranked = sorted(rows, key=lambda kv: kv[1], reverse=True)
    for name, total in ranked:
        mark = "  <-- intended" if name == winner_key else ""
        print(f"  {name:<{width}}  return {total:+10.3f}{mark}")
    top = ranked[0][0]
    verdict = "PASS" if top == winner_key else f"FAIL (winner was {top!r})"
    print(f"  => {verdict}")
    return top == winner_key


# --------------------------------------------------------------------------
def probe_guard_duel():
    banner("guard_duel -- can 'hold guard forever' beat timed guarding?")
    cfg = C.GuardDuelConfig()

    def run(guard_policy):
        total, streak = 0.0, lane_i32(0)
        for t in range(HORIZON):
            phase = t % ATTACK_PERIOD
            resolves = phase == 0 and t > 0
            threat = 0 < (ATTACK_PERIOD - phase) <= THREAT_WINDOW
            guard = guard_policy(t, threat, resolves)
            dealt = DAMAGE if (resolves and not guard) else 0.0
            out = C.guard_duel_signals(
                cfg,
                attack_requested=lane_bool(resolves),
                attack_accepted=lane_bool(resolves),
                damage_dealt=lane_f32(dealt),
                incoming_attack_active=lane_bool(threat),
                incoming_attack_resolved=lane_bool(resolves),
                guard_active=lane_bool(guard),
                guard_streak=streak,
                valid=lane_bool(True),
            )
            streak = C.advance_guard_streak(streak, lane_bool(guard), lane_bool(True))
            total += float(out.blocker_reward[0])
        return total

    rows = [
        ("always guard", run(lambda t, threat, res: True)),
        ("never guard", run(lambda t, threat, res: False)),
        ("timed guard", run(lambda t, threat, res: threat or res)),
        ("late guard", run(lambda t, threat, res: not threat and not res)),
        ("guard half the time", run(lambda t, threat, res: (t // 8) % 2 == 0)),
    ]
    return report(rows, "timed guard")


# --------------------------------------------------------------------------
def probe_ability_landing():
    banner("ability_landing -- can spraying beat aiming?")
    cfg = C.AbilityLandingConfig()

    def run(fire_policy):
        total = 0.0
        for t in range(HORIZON):
            # Opponent orbits: aim error and range sweep in and out of the window.
            aim_error = 45.0 - 44.0 * (1.0 if (t % 40) < 8 else 0.0)
            distance = 6.0 if (t % 40) < 8 else 14.0
            aimed = aim_error <= cfg.aim_tolerance_degrees and (
                distance <= cfg.effective_range_blocks
            )
            slot = fire_policy(t, aimed)
            damage = DAMAGE if (slot != C.ABILITY_NONE and aimed) else 0.0
            out = C.ability_landing_signals(
                cfg,
                ability_choice=lane_i32(slot),
                ability_accepted=lane_bool(slot != C.ABILITY_NONE),
                attributed_damage=lane_f32(damage),
                aim_error_degrees=lane_f32(aim_error),
                distance=lane_f32(distance),
                current_tick=lane_i32(t),
                valid=lane_bool(True),
            )
            total += float(out.reward[0])
        return total

    primary, secondary = cfg.primary_slot, cfg.secondary_slot
    rows = [
        ("spray every tick", run(lambda t, aimed: primary)),
        ("never fire", run(lambda t, aimed: C.ABILITY_NONE)),
        ("fire only when aimed", run(lambda t, aimed: primary if aimed else C.ABILITY_NONE)),
        (
            "aimed, secondary after unlock",
            run(
                lambda t, aimed: (
                    (secondary if t >= cfg.secondary_unlock_tick else primary)
                    if aimed
                    else C.ABILITY_NONE
                )
            ),
        ),
        (
            "secondary always (early = locked)",
            run(lambda t, aimed: secondary if aimed else C.ABILITY_NONE),
        ),
    ]
    return report(rows, "aimed, secondary after unlock")


# --------------------------------------------------------------------------
def probe_evasion():
    banner("evasion -- can running away beat holding the band?")
    cfg = C.EvasionConfig()
    band_mid = 0.5 * (cfg.hold_minimum_blocks + cfg.hold_maximum_blocks)

    def run(policy):
        total = 0.0
        for t in range(HORIZON):
            phase = t % ATTACK_PERIOD
            resolves = phase == 0 and t > 0
            threat = 0 < (ATTACK_PERIOD - phase) <= THREAT_WINDOW
            distance, dodge = policy(t, threat, resolves)
            # A dodge on the resolving tick avoids the hit; distance does not.
            hit = resolves and dodge == C.DODGE_NONE
            out = C.evasion_signals(
                cfg,
                distance=lane_f32(distance),
                incoming_attack_active=lane_bool(threat),
                incoming_attack_resolved=lane_bool(resolves),
                damage_taken=lane_f32(DAMAGE if hit else 0.0),
                dodge_choice=lane_i32(dodge),
                valid=lane_bool(True),
            )
            total += float(out.reward[0])
        return total

    rows = [
        ("flee, dodge on resolve", run(lambda t, th, r: (40.0, 1 if r else 0))),
        ("hold band, never dodge", run(lambda t, th, r: (band_mid, 0))),
        ("hold band, dodge on resolve", run(lambda t, th, r: (band_mid, 1 if r else 0))),
        ("hold band, dodge spam", run(lambda t, th, r: (band_mid, 1))),
        ("hug at 1 block, dodge on resolve", run(lambda t, th, r: (1.0, 1 if r else 0))),
    ]
    return report(rows, "hold band, dodge on resolve")


# --------------------------------------------------------------------------
def probe_checkpoint_route():
    banner("checkpoint_route -- can oscillating or idling beat travelling?")
    cfg = C.CheckpointRouteConfig()
    start = 60.0

    def run(step_policy):
        total, distance, stall = 0.0, start, lane_i32(0)
        for t in range(HORIZON):
            step = step_policy(t)
            after = max(0.0, distance - step)
            out = C.checkpoint_route_signals(
                cfg,
                distance_before=lane_f32(distance),
                distance_after=lane_f32(after),
                fall_damage=lane_f32(0.0),
                stall_ticks=stall,
                valid=lane_bool(True),
            )
            total += float(out.reward[0])
            stall = out.next_stall_ticks
            distance = after
            if bool(out.arrived[0]):
                break
        return total

    rows = [
        ("direct sprint (0.4/tick)", run(lambda t: 0.4)),
        ("direct walk (0.15/tick)", run(lambda t: 0.15)),
        ("oscillate +-0.4", run(lambda t: 0.4 if t % 2 == 0 else -0.4)),
        ("idle", run(lambda t: 0.0)),
        ("wander away", run(lambda t: -0.2)),
    ]
    return report(rows, "direct sprint (0.4/tick)")


# --------------------------------------------------------------------------
def probe_punish_window():
    banner("punish_window -- can mashing beat waiting for the window?")
    cfg = C.PunishWindowConfig()
    recovery_len = 8

    def run(attack_policy):
        total = 0.0
        for t in range(HORIZON):
            phase = t % ATTACK_PERIOD
            opponent_active = phase < 6
            recovery = 6 <= phase < 6 + recovery_len
            closing = phase == 6 + recovery_len - 1
            attack = attack_policy(t, recovery, opponent_active)
            damage = DAMAGE if (attack and not opponent_active) else 0.0
            out = C.punish_window_signals(
                cfg,
                attack_requested=lane_bool(attack),
                damage_dealt=lane_f32(damage),
                recovery_window_active=lane_bool(recovery),
                opponent_attack_active=lane_bool(opponent_active),
                window_closing=lane_bool(closing),
                valid=lane_bool(True),
            )
            total += float(out.reward[0])
        return total

    rows = [
        ("mash every tick", run(lambda t, rec, act: True)),
        ("never attack", run(lambda t, rec, act: False)),
        ("attack only in recovery", run(lambda t, rec, act: rec)),
        ("attack whenever not active", run(lambda t, rec, act: not act)),
    ]
    return report(rows, "attack only in recovery")


def main() -> int:
    """Run every probe and return a shell exit code, so this can gate a change."""

    results = {
        "guard_duel": probe_guard_duel(),
        "ability_landing": probe_ability_landing(),
        "evasion": probe_evasion(),
        "checkpoint_route": probe_checkpoint_route(),
        "punish_window": probe_punish_window(),
    }
    banner("SUMMARY")
    for name, ok in results.items():
        print(f"  {name:<20} {'PASS' if ok else 'FAIL'}")
    passed = sum(results.values())
    print(f"\n  {passed}/{len(results)} contracts resist their degenerate policy")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
