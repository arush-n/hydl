"""Can we start building agents? Answer it by training one.

Both blockers that stood between here and a trainable combat agent are closed:
the action mask does not go unsatisfiable (handoff rev 181's corrected 24/24
native trace, independently consistent with `test_invariants_unique`), and the
Java port now accepts independent encoder/recurrent widths (`PolicyShapeTest`,
asymmetric 2/3, 27/27 green -- run, not read).

So the remaining question is not architectural, it is empirical: does a policy
on the **current** contract actually train? This runs the smallest thing that
answers it, and it deliberately reports the numbers that distinguish "the loop
ran" from "the agent learned" -- a smoke that completes zero episodes is not
competence evidence, and this prints episode counts so that cannot be blurred.

    # from the repository root:
    PYTHONPATH=. python -u adk/tools/can_we_train.py --updates 5

Every shape comes from `PPOConfig.from_environment_spec`, never from a literal.
Hard-coding 8271/99 here is exactly how a trainer silently keeps building
against a retired contract; worse, a hand-built config defaults
`action_transport` to `"scalar"`, which is wrong for a 12-head factored action
space and would train a differently-shaped policy without complaining.

Note on scope: this trains in JAX only. It needs no bridge and touches no
native session, so it does not take the evidence lease on 127.0.0.1:5556 and
is safe to run while a deployment decision is pending.
"""
from __future__ import annotations

import argparse
import time

import jax
import numpy as np

# Reported per update. Anything absent from the metrics tuple is skipped rather
# than crashing, so a Gym-side rename degrades the report instead of the run.
COLUMNS = (
    ("episodes_completed", "episodes", "{:>9.0f}"),
    ("mean_episode_return", "return", "{:>9.3f}"),
    ("mean_episode_length", "length", "{:>9.1f}"),
    ("episode_successes", "wins", "{:>9.0f}"),
    ("episode_deaths", "deaths", "{:>9.0f}"),
    # Without this, an episode that ends 0-wins-0-deaths is uninterpretable:
    # it is a timeout, not a fight anybody resolved.
    ("episode_other_terminal", "other", "{:>9.0f}"),
    ("total_loss", "loss", "{:>9.4f}"),
    ("entropy", "entropy", "{:>9.3f}"),
    # Whether the critic is fitting at all. A flat ~0 here with a falling loss
    # means the value head is predicting the mean, not the episode.
    ("explained_variance", "explvar", "{:>9.3f}"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--encoder-size", type=int, default=64)
    ap.add_argument("--recurrent-size", type=int, default=128,
                    help="deliberately != encoder-size: the equal-width "
                         "constraint was retired and this exercises that")
    ap.add_argument("--checkpoint", default=None,
                    help="save trained weights here (.npz) so the "
                         "Java port can be checked against a "
                         "TRAINED policy, not near-zero weights")
    ap.add_argument("--loadout", default="iron_sword")
    ap.add_argument("--opponent", default="iron_sword",
                    help="target profile; an armed opponent is what "
                         "lets an episode terminate at all")
    ap.add_argument("--inert-opponent", action="store_true",
                    help="control: replace the deterministic first-legal "
                         "opponent controller with an inert one. Deaths should "
                         "collapse to ~0 -- if they do not, something other "
                         "than the opponent is killing the agent, and 'the "
                         "policy loses' is the wrong diagnosis")
    ap.add_argument("--world", default="open_flat",
                    help="world capability provider: 'open_flat' (default, and "
                         "the control every prior number was taken on) or a "
                         "synthetic arena -- parkour/lava/islands")
    ap.add_argument("--geometry", default=None,
                    help="real 9x9x9 geometry arena: pit/lava_pool/pillars/"
                         "corridor/ledge/arena_pit/flat. UNVERIFIED -- no run "
                         "has yet shown an agent standing on authored ground, "
                         "so a result here is not yet trustworthy")
    args = ap.parse_args()

    # Validate names BEFORE the heavy setup below, so a typo fails immediately
    # rather than behind the JAX imports and `arsenal_runtime_config`.
    #
    # Honest provenance: `--geometry nope` was once observed to return nothing
    # within a 600 s timeout, which prompted this. That run was launched during
    # a period of GPU contention (four JAX processes, 97% of device memory), so
    # the timeout is NOT cleanly attributable to where the check sat -- the
    # ordering is right on its own merits, but treat the 600 s as unexplained
    # rather than as evidence for it.
    #
    # Registry lookups only here -- no provider is constructed.
    from adk.environments.frames import ARENAS
    from adk.environments.synthetic import PRESETS
    if args.world != "open_flat" and args.world not in PRESETS:
        raise SystemExit(f"unknown world {args.world!r}; "
                         f"have 'open_flat' and {sorted(PRESETS)}")
    if args.geometry is not None and args.geometry not in ARENAS:
        raise SystemExit(f"unknown geometry {args.geometry!r}; "
                         f"have {sorted(ARENAS)}")

    from hytalegym.jax.combat.arsenal import (
        arsenal_runtime_config,
        hytale_0_5_7_loadouts,
        open_flat_arsenal_world_capabilities,
    )
    from hytalegym.jax.combat.types import default_combat_params
    from hytalegym.jax.combat.opponents.runtime.policy import (
        inert_opponent_ability_slots,
    )
    from hytalegym.jax.training.arsenal import make_arsenal_ppo_environment
    from hytalegym.jax.training.ppo import initialize_training, make_train_step
    from hytalegym.jax.training.checkpoint import save_policy_checkpoint
    from hytalegym.jax.training.types import PPOConfig

    # `target_active=True` matters: with an inert target the agent cannot end an
    # episode, so the run would complete zero episodes and read as a training
    # failure when it is actually an empty scene.
    #
    # The loadout is batch-first with one row per profile name, and the
    # environment hard-checks `keys.shape[0] != expected_batch`, so num_envs is
    # expressed by repeating the name rather than by a separate batch argument.
    # `target_profiles` is passed explicitly: omitting it resolves the empty
    # name to profile index 0 (`iron_sword`), which is a real armed opponent but
    # only by coincidence of catalog ordering.
    params = default_combat_params(microticks=1, target_active=True)
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts(
        [args.loadout] * args.num_envs,
        target_profiles=[args.opponent] * args.num_envs,
    ))
    opponent_kwargs = (
        {"opponent_ability_provider": inert_opponent_ability_slots}
        if args.inert_opponent else {}
    )
    # Names were already validated above; this only builds. Registries are
    # looked up rather than duplicated into `choices=`, which would drift the
    # moment an arena is added.
    if args.world == "open_flat":
        world_capability_provider = open_flat_arsenal_world_capabilities
    else:
        world_capability_provider = PRESETS[args.world].provider()

    world_kwargs = {"world_capability_provider": world_capability_provider}
    if args.geometry is not None:
        from adk.environments.frames import build as build_geometry
        # One immutable frame broadcast across the batch; built once here
        # because conversion happens outside JIT by design.
        world_kwargs["geometry_provider"] = build_geometry(
            args.geometry, batch_size=args.num_envs)

    environment = make_arsenal_ppo_environment(
        params, runtime,
        **world_kwargs,
        **opponent_kwargs,
    )

    config = PPOConfig.from_environment_spec(
        environment.spec,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=2,
        num_minibatches=2,
        encoder_size=args.encoder_size,
        recurrent_size=args.recurrent_size,
    )
    print(f"loadout: {args.loadout} vs {args.opponent}  (batch {args.num_envs})")
    print("opponent controller: "
          + ("INERT (control run)" if args.inert_opponent
             else "first_legal_opponent_ability_slots (armed, default)"))
    print(f"contract: obs {config.observation_size}  actions {config.action_size}")
    print(f"heads: {list(config.action_head_sizes)}")
    print(f"transport: {config.action_transport}  "
          f"distribution: {config.action_distribution}")
    print(f"widths: encoder {config.encoder_size} / recurrent "
          f"{config.recurrent_size}"
          f"{'  (ASYMMETRIC)' if config.encoder_size != config.recurrent_size else ''}")

    key = jax.random.key(0)
    start = time.time()
    state = initialize_training(key, config, environment=environment)
    step = make_train_step(config, environment=environment)
    print(f"initialised in {time.time() - start:.1f}s\n")

    present: list[tuple[str, str, str]] = []
    history: list[tuple[int, float]] = []  # (episodes, mean return) per update
    for update in range(args.updates):
        began = time.time()
        key, sub = jax.random.split(key)
        state, metrics = step(state, sub)

        if not present:
            present = [c for c in COLUMNS if hasattr(metrics, c[0])]
            print(f"{'update':>7}"
                  + "".join(f" {label:>9}" for _, label, _ in present)
                  + f" {'seconds':>8}")

        row = f"{update + 1:>7d}"
        for field, _, fmt in present:
            row += " " + fmt.format(float(np.asarray(getattr(metrics, field))))
        print(row + f" {time.time() - began:>8.1f}", flush=True)
        history.append((int(np.asarray(metrics.episodes_completed)),
                        float(np.asarray(metrics.mean_episode_return))))

    steps = int(np.asarray(state.total_environment_steps))
    print(f"\ntotal environment steps: {steps}")

    if args.checkpoint:
        # `state.policy_params`, not the training handle. Passing the wrong
        # object writes the initialization rather than the updated policy.
        # Explicit update/step metadata is the deployment proof; actor std is
        # printed only as debugging telemetry because dispersion can overlap
        # between trained and synthetic policies.
        written = save_policy_checkpoint(
            args.checkpoint, state.policy_params, config,
            metadata={"updates": args.updates,
                      "environment_steps": steps,
                      "loadout": args.loadout,
                      "opponent": args.opponent})
        std = float(np.std(np.asarray(state.policy_params.actor.kernel)))
        print(f"checkpoint: {written}")
        print(f"actor_kernel std: {std:.6f}  (telemetry only; consult the "
              "checkpoint's update/step provenance to establish training)")

    # Episode-weighted, because an update that completed 2 episodes should not
    # count as much as one that completed 40. Comparing the first and last
    # thirds is the cheapest honest read on whether return moved at all; it is
    # not a significance test and a few hundred episodes is a small sample.
    total = sum(n for n, _ in history)
    print(f"completed episodes: {total}")
    if total == 0:
        print("VERDICT: the loop runs, but zero episodes completed -- this is "
              "plumbing evidence only, not competence evidence.")
        return 0

    third = max(1, len(history) // 3)
    def weighted(rows):
        n = sum(c for c, _ in rows)
        return (sum(c * r for c, r in rows) / n) if n else float("nan"), n
    early, early_n = weighted(history[:third])
    late, late_n = weighted(history[-third:])
    print(f"mean episode return: first {third} updates {early:+.3f} "
          f"(n={early_n})  ->  last {third} updates {late:+.3f} (n={late_n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
