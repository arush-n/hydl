"""The P2 probe as a repo tool: does a trunk's entity row carry entity 1's body-frame place? (``docs/TRAINING_PLAN.md`` 8.3,
stage PT1; the probe's first run is ``scratchpad/p2_logs``, 2026-10-25.)

Collect: on the walk_to (A: entity 1 at ``blocks.OTHER_XYZ``), come_here (B: placed per lane) and track (C: on its mover
path) lanes, run each set on its teachers (the trunk runs with its carried state, every decision's trunk output kept,
``skill_dagger_v3.make_kernels`` with the exact path, no slot cut). Per decision inside each lane's horizon, keep entity
1's row of the trunk's encoding (``state_tick.entity_rows``), its raw token row, the context, and the true place. Decisions
where entity 1 has no row are counted and dropped.

Fit: ridge read-outs on standardized inputs, lambda by validation RMSE, lanes split 60/20/20 (``split_lanes``). Within each
set and across sets (A to B, A to C, B to C). Inputs: the trunk row (i), the raw token (ii, the ceiling), and the context
(iii, within-set only). Targets: entity 1's forward, right, up, distance and bearing (sin, cos) in the body frame.

CLI (from the worktree root, nice 10)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu nice -n 10 \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.bench.entity_probe --trunk TRUNK.npz \\
        [--lanes 256] [--seed 9301] --json OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.tokens_v3 import EntityTokenV3
from experimental.neural_agent.skill_agent.model.skill_module import Trunk, load_trunk
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.training import blocks
from experimental.neural_agent.skill_agent.training import distill_v3 as v3
from experimental.neural_agent.skill_agent.training import skill_dagger_v3 as d3
from experimental.neural_agent.skill_agent.training.skill_dagger import SkillTrainConfig, merge
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY

SETS = {
    "A": ("walk_to", "fixed: walk_to lanes, entity 1 at blocks.OTHER_XYZ, the agent walks"),
    "B": ("come_here", "placed: come_here lanes, entity 1 placed per lane"),
    "C": ("track", "moving: track lanes, entity 1 on its mover path"),
}
TARGETS = ("forward", "right", "up", "distance", "bearing_sin", "bearing_cos")
INPUTS = ("trunk", "raw", "context")
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
FEATURE = {name: index for index, name in enumerate(EntityTokenV3.FEATURES)}
SCALE = 16.0  # state/frames.py POSITION_SCALE_BLOCKS


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def collect(skill: str, lanes_n: int, seed: int, trunk: Trunk, max_ticks: int | None = None) -> tuple[dict, dict]:
    """One set's decisions inside their horizons (arrays by name, the rows of decisions where entity 1 has a row) and its
    meta. The lanes and actions are the trainer's: the same split of the seed, and the teachers' executed factors."""

    started = time.perf_counter()
    strategy = d3._strategy(skill, SkillTrainConfig(lanes=lanes_n, seed=seed))
    known = tuple(sorted({spec.verb for spec in strategy.skills}))
    kernels = d3.make_kernels(strategy, trunk, known)
    kind = jnp.asarray(np.arange(lanes_n) % len(strategy.kinds), dtype=jnp.int32)
    key_lanes, key_rollout = jax.random.split(jax.random.key(seed))
    lanes = strategy.episodes(key_lanes, kind, False)
    env = blocks.environment(lanes_n, strategy.config.maximum_turn_degrees)
    mover = getattr(lanes.goals[0], "mover", None)
    if mover is not None:
        env = env.with_mover(mover)
    batch = env.batch
    horizon = np.asarray(lanes.horizon)
    full = int(horizon.max())
    ticks = full if max_ticks is None else min(full, max_ticks)
    keys = jax.random.split(key_rollout, full + 2)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, dtype=jnp.float32), (batch, 2))
    carry = env.reset(jax.random.split(keys[0], batch), spawn, lanes.spawn_yaw, d3.placed_other(lanes, batch))
    neutral = jnp.broadcast_to(jnp.asarray(blocks.NEUTRAL, dtype=jnp.int32), (batch, blocks.HEAD_COUNT))
    carry, observation, _ = env.step(carry, neutral, jax.random.split(keys[1], batch))
    start = np.asarray(carry.runtime.combat.position[:, TARGET_ENTITY])
    layer_step = v3._layer_program(v3._world_tick(carry.runtime.combat))
    state = v3._start_state(trunk.params, batch, trunk.config)
    active = lanes.active[:, 0]
    last = neutral
    store: dict[str, list] = {name: [] for name in ("agent", "target", "yaw", "row", "valid", "keep", "trunk", "raw",
                                                     "context")}

    def gather(array: jax.Array, pick: jax.Array) -> jax.Array:
        index = jnp.broadcast_to(pick[:, None, None], (batch, 1, array.shape[2]))
        return jnp.take_along_axis(array, index, axis=1)[:, 0]

    for index in range(ticks):
        combat = carry.runtime.combat
        taught = kernels.teachers(combat, lanes)
        executed = merge(taught.labels, taught.beside, strategy.spec, active)
        position = combat.position[:, AGENT_ENTITY]
        yaw = combat.yaw[:, AGENT_ENTITY]
        _, last_action, _, _ = state_tick.host_inputs(observation, last, jnp.zeros((batch,), jnp.bool_))
        reset = jnp.full((batch,), index == 0)
        tokens, after = layer_step(state, observation, position, yaw, last_action, reset)
        view, state = kernels.trunk(trunk.params, tokens, after, reset)  # the exact path: no slot cut
        row = state_tick.entity_rows(jnp.full((batch, 1), TARGET_ENTITY, dtype=jnp.int32), view.entity_ids)[:, 0]
        pick = jnp.maximum(row, 0)
        features, valid = tokens["entity"]
        store["agent"].append(np.asarray(position))
        store["target"].append(np.asarray(combat.position[:, TARGET_ENTITY]))
        store["yaw"].append(np.asarray(yaw))
        store["row"].append(np.asarray(row))
        store["valid"].append(np.asarray(valid[jnp.arange(batch), pick] & (row >= 0)))
        store["keep"].append(index < horizon)
        store["trunk"].append(np.asarray(gather(view.encoded.tokens["entity"], pick)))
        store["raw"].append(np.asarray(gather(features, pick)))
        store["context"].append(np.asarray(view.encoded.context))
        carry, observation, _ = env.step(carry, executed, jax.random.split(keys[index + 2], batch))
        last = executed

    stacked = {name: np.stack(values) for name, values in store.items()}  # [T, B, ...]
    keep, row = stacked["keep"], stacked["row"]
    present = keep & (row >= 0)
    kind_np = np.asarray(kind)
    data = {
        "lane": np.nonzero(present)[1].astype(np.int32), "tick": np.nonzero(present)[0].astype(np.int32),
        "agent": stacked["agent"][present], "target": stacked["target"][present],
        "yaw": stacked["yaw"][present], "trunk": stacked["trunk"][present], "raw": stacked["raw"][present],
        "context": stacked["context"][present],
        "visible": stacked["raw"][present][:, FEATURE["visible"]],
        "remembered": stacked["raw"][present][:, FEATURE["remembered"]],
    }
    meta = {
        "skill": skill, "lanes": lanes_n, "seed": seed, "ticks_run": ticks, "horizon_ticks": horizon.tolist(),
        "kinds": list(strategy.kinds), "kind_lanes": np.bincount(kind_np, minlength=len(strategy.kinds)).tolist(),
        "mover": mover is not None, "start_target_xyz_mean": start.mean(axis=0).tolist(),
        "start_target_xyz_min": start.min(axis=0).tolist(), "start_target_xyz_max": start.max(axis=0).tolist(),
        "decisions_in_horizon": int(keep.sum()), "decisions_row_missing": int((keep & (row < 0)).sum()),
        "decisions_row_present": int(present.sum()),
        "row_valid_mismatch": int((present & ~stacked["valid"]).sum()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    _log(f"  {skill}: {meta['decisions_row_present']} decisions with a row ({meta['decisions_row_missing']} of "
         f"{meta['decisions_in_horizon']} in-horizon decisions had none) in {meta['seconds']} s")
    return data, meta


def targets(data: dict) -> dict:
    """Entity 1's true place from the agent, in its body frame (``blocks.body_vector``), and the planar distance and
    bearing of it."""

    offset = (data["target"] - data["agent"]).astype(np.float32)
    body = np.asarray(blocks.body_vector(jnp.asarray(offset), jnp.asarray(data["yaw"], jnp.float32)), np.float64)
    forward, right, up = body[:, 0], body[:, 1], body[:, 2]
    bearing = np.arctan2(right, forward)
    return {"forward": forward, "right": right, "up": up, "distance": np.hypot(forward, right),
            "bearing_sin": np.sin(bearing), "bearing_cos": np.cos(bearing)}


def split_lanes(lanes_n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(lanes_n)
    n_train, n_val = int(round(0.6 * lanes_n)), int(round(0.2 * lanes_n))
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def ridge_fit(features: np.ndarray, y: np.ndarray, alpha: float) -> tuple:
    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (features - mu) / sd
    ym = y.mean()
    n, d = z.shape
    gram = z.T @ z / n + alpha * np.eye(d)
    weights = np.linalg.solve(gram, z.T @ (y - ym) / n)
    return mu, sd, ym, weights


def predict(model: tuple, features: np.ndarray) -> np.ndarray:
    mu, sd, ym, weights = model
    return ((features - mu) / sd) @ weights + ym


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    if y.size == 0:
        return {"r2": None, "rmse": None, "n": 0}
    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"r2": 1.0 - sse / sst if sst > 0 else None, "rmse": float(np.sqrt(sse / y.size)), "n": int(y.size)}


def fit_best(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[tuple, float]:
    """The ridge model with the lowest validation RMSE over ALPHAS, and its alpha."""

    best = None
    for alpha in ALPHAS:
        model = ridge_fit(x_train, y_train, alpha)
        rmse = scores(y_val, predict(model, x_val))["rmse"]
        if best is None or rmse < best[0]:
            best = (rmse, alpha, model)
    return best[2], best[1]


def fit_stage(data: dict, meta: dict, sets: list[str], lanes_n: int, trunk_sha: str) -> dict:
    tgt = {name: targets(data[name]) for name in sets}
    masks = {name: dict(zip(("train", "val", "test"),
                            (np.isin(data[name]["lane"], part) for part in split_lanes(lanes_n))))
             for name in sets}

    frame = {}
    for name in sets:
        d = data[name]
        vis = d["visible"] > 0.5
        rem = (d["remembered"] > 0.5) & ~vis
        check = {}
        for label, sel in (("visible", vis), ("remembered_only", rem)):
            if sel.sum() < 3:
                check[label] = None
                continue
            raw_forward = d["raw"][sel][:, FEATURE["forward"]].astype(np.float64)
            x = np.stack([raw_forward, np.ones_like(raw_forward)], axis=1)
            coef, *_ = np.linalg.lstsq(x, tgt[name]["forward"][sel], rcond=None)
            resid = tgt[name]["forward"][sel] - x @ coef
            check[label] = {"n": int(sel.sum()), "slope": float(coef[0]), "intercept": float(coef[1]),
                            "max_abs_residual_blocks": float(np.max(np.abs(resid)))}
        frame[name] = check

    within, cross = [], []
    for name in sets:
        d, t, m = data[name], tgt[name], masks[name]
        for input_name in INPUTS:
            x = d[input_name].astype(np.float64)
            for target in TARGETS:
                y = t[target]
                model, alpha = fit_best(x[m["train"]], y[m["train"]], x[m["val"]], y[m["val"]])
                test_pred = predict(model, x[m["test"]])
                vis_sel = d["visible"][m["test"]] > 0.5
                within.append({
                    "set": name, "input": input_name, "target": target, "alpha": alpha,
                    "n_train": int(m["train"].sum()), "n_val": int(m["val"].sum()), "n_test": int(m["test"].sum()),
                    "test": scores(y[m["test"]], test_pred),
                    "test_visible_only": scores(y[m["test"]][vis_sel], test_pred[vis_sel]),
                })
    for source, target_set in (("A", "B"), ("A", "C"), ("B", "C")):
        if source not in sets or target_set not in sets:
            continue
        ds, dt = data[source], data[target_set]
        ms, mt = masks[source], masks[target_set]
        for input_name in ("trunk", "raw"):
            xs = ds[input_name].astype(np.float64)
            xt = dt[input_name].astype(np.float64)
            for target in TARGETS:
                ys, yt = tgt[source][target], tgt[target_set][target]
                model, alpha = fit_best(xs[ms["train"]], ys[ms["train"]], xs[ms["val"]], ys[ms["val"]])
                cross.append({
                    "fit": source, "to": target_set, "input": input_name, "target": target, "alpha": alpha,
                    "n_train": int(ms["train"].sum()), "n_test": int(mt["test"].sum()),
                    "test": scores(yt[mt["test"]], predict(model, xt[mt["test"]])),
                })
    return {"sets": {name: meta[name] for name in sets}, "trunk_sha256": trunk_sha, "lanes": lanes_n,
            "frame_check": frame, "within": within, "cross": cross}


def run_probe(trunk_path: Path, lanes_n: int = 256, seed: int = 9301, *, sets: Sequence[str] = ("A", "B", "C"),
              max_ticks: int | None = None) -> dict:
    """Collect the sets and fit them: the result of the P2 probe, as ``p2_results.json`` holds it."""

    started = time.perf_counter()
    trunk = load_trunk(trunk_path)
    data, meta = {}, {}
    for name in sets:
        _log(f"set {name} ({SETS[name][0]}): {SETS[name][1]}")
        data[name], meta[name] = collect(SETS[name][0], lanes_n, seed, trunk, max_ticks)
    result = fit_stage(data, meta, list(sets), lanes_n, trunk.sha256)
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    return result


def print_tables(result: dict) -> None:
    """The result as text: decisions per set, the frame check, and the within- and across-set tables."""

    lines = ["Decisions per set (inside horizon; row = entity 1 has a token row):"]
    for name, meta in result["sets"].items():
        share = meta["decisions_row_missing"] / max(meta["decisions_in_horizon"], 1)
        lines.append(f"  {name} {meta['skill']}: in-horizon {meta['decisions_in_horizon']}, no row "
                     f"{meta['decisions_row_missing']} ({share:.1%}), row present {meta['decisions_row_present']}, "
                     f"valid mismatches {meta['row_valid_mismatch']}")
    lines.append("Frame check (raw forward x SCALE vs body forward, blocks):")
    for name, check in result["frame_check"].items():
        for label, value in check.items():
            if value is None:
                lines.append(f"  {name} {label}: too few rows")
            else:
                lines.append(f"  {name} {label}: n={value['n']} slope={value['slope']:.4f} (x{SCALE:g} expected 1) "
                             f"intercept={value['intercept']:.4f} max|resid|={value['max_abs_residual_blocks']:.4f} blocks")
    lines.append("Within-set test (R2 / RMSE in blocks; sin, cos unitless):")
    for row in result["within"]:
        t, v = row["test"], row["test_visible_only"]
        r2 = "n/a" if t["r2"] is None else f"{t['r2']:.4f}"
        vr2 = "n/a" if v["r2"] is None else f"{v['r2']:.4f}"
        lines.append(f"  {row['set']} {row['input']:8s}{row['target']:12s} alpha={row['alpha']:<7g} R2 {r2:8s} "
                     f"RMSE {t['rmse']:.4f} | visible-only R2 {vr2} (n={v['n']})")
    lines.append("Cross-set test (fit on the first set's train lanes):")
    for row in result["cross"]:
        t = row["test"]
        r2 = "n/a" if t["r2"] is None else f"{t['r2']:.4f}"
        lines.append(f"  {row['fit']}->{row['to']} {row['input']:6s}{row['target']:12s} alpha={row['alpha']:<7g} "
                     f"R2 {r2:8s} RMSE {t['rmse']:.4f} (n={t['n']})")
    sys.stderr.write("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The P2 probe as a repo tool: a trunk's entity row against entity 1's place.")
    parser.add_argument("--trunk", type=Path, required=True, help="the trunk (.npz with its .json)")
    parser.add_argument("--lanes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--json", type=Path, required=True, help="the result (.json); must not exist")
    args = parser.parse_args(argv)
    if args.json.exists():
        parser.error(f"{args.json} exists: a run never overwrites another")
    result = run_probe(args.trunk, args.lanes, args.seed)
    print_tables(result)
    args.json.write_text(json.dumps(result, indent=1, default=blocks.json_default) + "\n")
    return 0


__all__ = ["SETS", "collect", "fit_stage", "main", "run_probe", "split_lanes", "targets"]


if __name__ == "__main__":
    raise SystemExit(main())
