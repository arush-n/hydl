"""Run a minigame on a declared scene and log it. The end-to-end entry point.

    python -m adk.tools.run_minigame --minigame sprint --weapon iron_sword
    python -m adk.tools.run_minigame --minigame spacing --weapon iron_daggers \
        --world region --node-seed 111111

Uses a scripted driver, not a trained policy: the question this answers is
"does the shaped signal move on this scene", which is the precondition for
training on it. A minigame whose reward is flat under a driver that exercises
the behaviour is broken setup, and spending a PPO run to discover that is waste.

Reports the shaped term separately from the native reward, because the whole
point of `tasks.shaped()` is that they stay comparable.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from adk.runs import Run
from adk.scenarios import probes
from adk.environments.config import SceneConfig
from adk.scenarios.minigames import REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minigame", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--weapon", default="iron_sword")
    ap.add_argument("--opponent-weapon", default="iron_sword")
    ap.add_argument("--world", default="open_flat",
                    choices=("open_flat", "fail_closed", "region"))
    ap.add_argument("--task", default="baseline")
    ap.add_argument("--node-seed", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--driver", default="aggressor",
                    help="scripted policy from adk.scenarios.probes")
    ap.add_argument("--opponent-distance", type=float, default=None,
                    help="default closes the standoff to 2.0; the ruleset's "
                         "2.8 leaves the opponent nearly harmless")
    args = ap.parse_args()

    config = SceneConfig(
        weapon=args.weapon, opponent_weapon=args.opponent_weapon,
        world=args.world, task=args.task, node_seed=args.node_seed,
        num_envs=args.num_envs, minigame=args.minigame,
        minigame_weight=args.weight,
        **({} if args.opponent_distance is None
           else {"opponent_distance": args.opponent_distance}))
    scene = config.build()

    # The same scene without shaping, stepped in lockstep, so the shaped term
    # can be reported on its own rather than inferred by subtraction from a
    # differently-seeded run.
    bare = jax.jit(scene.environment.step_detailed)

    names = probes.head_names()
    state, obs, mask = scene.reset(jax.random.key(5))
    driver = probes.make_policy(args.driver, names)

    label = f"{args.minigame}-{args.weapon}-{args.world}"
    with Run.create(label, scene.provenance) as run:
        shaped_total = native_total = 0.0
        prev_health = np.asarray(state.runtime.combat.health).copy()
        damage_taken = damage_dealt = 0.0
        for tick in range(args.ticks):
            actions = np.asarray(driver(np.asarray(mask), tick, None))
            keys = jax.random.split(jax.random.key(4000 + tick), args.num_envs)
            before = state
            state, obs, reward, done, mask, info = scene.step(
                before, obs, jnp.asarray(actions), keys)
            _, _, native, _, _, _ = bare(
                before, obs, jnp.asarray(actions), keys)

            shaped_row = float(np.asarray(reward).sum())
            native_row = float(np.asarray(native).sum())
            shaped_total += shaped_row - native_row
            native_total += native_row

            health = np.asarray(state.runtime.combat.health)
            drop = np.maximum(prev_health - health, 0.0)
            damage_taken += float(drop[:, 0].sum())
            damage_dealt += float(drop[:, 1].sum())
            prev_health = health

            if tick % 25 == 0:
                run.log(tick=tick, shaped=round(shaped_row - native_row, 5),
                        native=round(native_row, 5))

        game = REGISTRY[args.minigame]
        # A shaped term that never moved means the driver did not exercise the
        # behaviour, or the scene cannot express it. Either way the number is
        # not evidence about a policy.
        verdict = "SIGNAL" if abs(shaped_total) > 1e-6 else "FLAT - setup is void"
        blocked = [r for r in game.requires
                   if r == "armed_opponent" and damage_taken <= 0.0]
        run.finish(status="completed", shaped_total=round(shaped_total, 4),
                   native_total=round(native_total, 4),
                   damage_taken=damage_taken, damage_dealt=damage_dealt,
                   verdict=verdict, unmet_requires=blocked)

        print(f"{label}")
        print(f"  teaches        : {game.teaches}")
        print(f"  shaped total   : {shaped_total:+.4f}   {verdict}")
        print(f"  native total   : {native_total:+.4f}")
        print(f"  damage dealt   : {damage_dealt:.1f}")
        print(f"  damage taken   : {damage_taken:.1f}"
              + ("   <-- opponent never landed a hit; "
                 "every `armed_opponent` result here is void"
                 if blocked else ""))
        print(f"  logged to      : adk/runs/{run.directory.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
