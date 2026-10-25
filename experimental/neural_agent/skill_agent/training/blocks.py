"""Training building blocks that every skill's strategy shares: frames, request features, sampling, rollout bookkeeping,
the hold rule, legality, dataset chunks, saving and run provenance.

A skill's training strategy (``neural_skills/<skill>/train.py``) composes these through ``training/skill_dagger.py``;
the A2 reference trainer (``training/distill.py``) uses them too. Moved here from ``distill.py`` on 2026-10-19, with
public names and no change in behaviour.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.bench import bench_tick
from experimental.neural_agent.skill_agent.model.observation_adapter import legal_tokens
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES as HEAD_NAMES,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES as HEAD_SIZES,
)

REQUEST_SLOTS = 6
HEAD_COUNT = len(HEAD_NAMES)
HEAD_STARTS = tuple(int(start) for start in np.cumsum((0,) + tuple(HEAD_SIZES[:-1])))
"""Offset of each head's choices in the flattened action mask (``carry.action_mask``)."""
NEUTRAL = np.asarray(st.NEUTRAL_FACTORS, dtype=np.int32)
SPAWN_XZ = st.DEFAULT_SPAWN_XZ
OTHER_XYZ = st.DEFAULT_OTHER_XYZ
EYE_XYZ = st.DEFAULT_EYE_XYZ
MODE_PARALLEL = (1.0, 0.0, 0.0)
URGENCY = 0.5
POSITION_SCALE = 16.0
AGE_SCALE = 256.0
HOLD_TICKS = 5
"""Consecutive decisions on target that make a held success (``hold_update``)."""


def wrap_degrees(degrees: jax.Array) -> jax.Array:
    return jnp.mod(degrees + 180.0, 360.0) - 180.0


def facing_degrees(dx: jax.Array, dz: jax.Array) -> jax.Array:
    """The engine's yaw of the bearing (dx, dz), in degrees: forward at yaw y is (-sin y, -cos y)."""

    return jnp.rad2deg(jnp.arctan2(-dx, -dz))


def body_vector(vector: jax.Array, yaw_degrees: jax.Array) -> jax.Array:
    """A world vector [B, 3] as (forward, right, up) in the body frame: forward (-sin y, 0, -cos y), right
    (cos y, 0, -sin y)."""

    yaw = jnp.deg2rad(yaw_degrees)
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    forward = -jnp.sin(yaw) * x - jnp.cos(yaw) * z
    right = jnp.cos(yaw) * x - jnp.sin(yaw) * z
    return jnp.stack([forward, right, y], axis=1)


def body_vector_np(vector: np.ndarray, yaw_degrees: np.ndarray) -> np.ndarray:
    """:func:`body_vector` for N world vectors [N, 3] in numpy: (forward, right, up) in each body frame."""

    yaw = np.deg2rad(yaw_degrees)
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    forward = -np.sin(yaw) * x - np.cos(yaw) * z
    right = np.cos(yaw) * x - np.sin(yaw) * z
    return np.stack([forward, right, y], axis=1)


def head_vector(vector: jax.Array, head_yaw_degrees: jax.Array, head_pitch_degrees: jax.Array) -> jax.Array:
    """A world vector [B, 3] as (forward, right, up) in the head frame: ``body_vector`` at the head's yaw, then pitched
    about the right axis, positive pitch up. The head's forward is the gaze, so a target the head faces is (1, 0, 0)."""

    body = body_vector(vector, head_yaw_degrees)
    pitch = jnp.deg2rad(head_pitch_degrees)
    forward, right, up = body[:, 0], body[:, 1], body[:, 2]
    return jnp.stack(
        [jnp.cos(pitch) * forward + jnp.sin(pitch) * up, right, -jnp.sin(pitch) * forward + jnp.cos(pitch) * up],
        axis=1,
    )


def head_vector_np(vector: np.ndarray, head_yaw_degrees: np.ndarray, head_pitch_degrees: np.ndarray) -> np.ndarray:
    """:func:`head_vector` for N world vectors [N, 3] in numpy: the body frame at the head's yaw, pitched about the
    right axis (positive pitch up)."""

    body = body_vector_np(vector, head_yaw_degrees)
    pitch = np.deg2rad(head_pitch_degrees)
    forward, right, up = body[:, 0], body[:, 1], body[:, 2]
    return np.stack([np.cos(pitch) * forward + np.sin(pitch) * up, right,
                     -np.sin(pitch) * forward + np.cos(pitch) * up], axis=1)


def planar(xz: jax.Array) -> jax.Array:
    return jnp.stack([xz[:, 0], jnp.zeros_like(xz[:, 0]), xz[:, 1]], axis=1)


def broadcast_rows(mask: jax.Array, leaf: jax.Array) -> jax.Array:
    return mask.reshape(mask.shape + (1,) * (leaf.ndim - 1))


def resample(draw: Callable[[jax.Array], Any], accept: Callable[[Any], jax.Array], key: jax.Array, rounds: int = 500):
    """Redraw the lanes that fail ``accept`` until every lane passes. Rejection keeps the accepted distribution
    exact."""

    key, first = jax.random.split(key)
    value = draw(first)
    for _ in range(rounds):
        ok = accept(value)
        if bool(jnp.all(ok)):
            return value
        key, step_key = jax.random.split(key)
        fresh = draw(step_key)
        value = jax.tree_util.tree_map(lambda old, new: jnp.where(broadcast_rows(ok, old), old, new), value, fresh)
    raise RuntimeError(f"could not sample goals in the requested region within {rounds} rounds")


def argument_row(position: jax.Array, scalars: jax.Array, ticks: jax.Array,
                 direction: jax.Array | None = None) -> jax.Array:
    """One request's 15 argument features in ``ARG_GROUPS`` order: position, direction, scalars, mode, urgency, age.

    The direction is zeros unless given.
    """

    batch = position.shape[0]
    if direction is None:
        direction = jnp.zeros((batch, 3), dtype=jnp.float32)
    mode = jnp.broadcast_to(jnp.asarray(MODE_PARALLEL, dtype=jnp.float32), (batch, 3))
    urgency = jnp.full((batch, 1), URGENCY, dtype=jnp.float32)
    age = jnp.full((batch, 1), ticks / AGE_SCALE, dtype=jnp.float32)
    return jnp.concatenate([position, direction, scalars, mode, urgency, age], axis=1)


@functools.lru_cache(maxsize=None)
def environment(batch: int, maximum_turn_degrees: float) -> st.FlatTaskEnvironment:
    return st.make_flat_npc_environment(batch, maximum_turn_degrees=maximum_turn_degrees)


def argument_row_np(position: np.ndarray, scalars: np.ndarray, ticks: np.ndarray,
                    direction: np.ndarray | None = None) -> np.ndarray:
    """:func:`argument_row` for N rows in numpy: position, direction (zeros unless given), scalars, mode, urgency and age.
    A numpy twin, so the batched lowering runs no eager jnp."""

    rows = position.shape[0]
    if direction is None:
        direction = np.zeros((rows, 3), np.float32)
    mode = np.broadcast_to(np.asarray(MODE_PARALLEL, np.float32), (rows, 3))
    urgency = np.full((rows, 1), URGENCY, np.float32)
    age = (np.asarray(ticks, np.float32) / AGE_SCALE)[:, None]
    return np.concatenate([position, direction, scalars, mode, urgency, age], axis=1)


@functools.lru_cache(maxsize=None)
def token_kernel() -> Callable[..., dict]:
    """The legal observation's token sets at one decision, jitted: ``f(observation, body_yaw) -> type -> (features,
    mask)``, in the body frame."""

    return jax.jit(lambda observation, yaw: legal_tokens(
        observation, capacities=bench_tick.CAPACITIES, body_yaw_degrees=yaw))


def legal_rows(mask: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Per lane: every head's factor is in range and allowed by the flattened action mask."""

    lanes = np.arange(factors.shape[0])
    legal = np.ones(factors.shape[0], dtype=bool)
    for head, (start, size) in enumerate(zip(HEAD_STARTS, HEAD_SIZES)):
        choice = factors[:, head]
        in_range = (choice >= 0) & (choice < size)
        legal &= in_range & mask[lanes, start + np.clip(choice, 0, size - 1)]
    return legal


def host_rows(value: Any, rows: np.ndarray) -> np.ndarray:
    return np.asarray(value)[rows]


def chunk(tokens: dict, requests: Any, labels: jax.Array, label_mask: jax.Array, done: jax.Array,
          keep: np.ndarray) -> dict[str, np.ndarray]:
    """The dataset rows of one tick: the entity and self token sets, the requests, the teacher's labels and done
    flags."""

    rows = np.flatnonzero(keep)
    entity, entity_valid = tokens["entity"]
    self_features, self_valid = tokens["self"]
    return {
        "entity": host_rows(entity, rows),
        "entity_valid": host_rows(entity_valid, rows),
        "self": host_rows(self_features, rows),
        "self_valid": host_rows(self_valid, rows),
        "verb": host_rows(requests.verb, rows),
        "args": host_rows(requests.args, rows),
        "arg_present": host_rows(requests.arg_present, rows),
        "entity_pointer": host_rows(requests.entity_pointer, rows),
        "cell_pointer": host_rows(requests.cell_pointer, rows),
        "scope": host_rows(requests.scope, rows),
        "valid": host_rows(requests.valid, rows),
        "labels": host_rows(labels, rows),
        "label_mask": host_rows(label_mask, rows),
        "done": host_rows(done, rows),
    }


def concat_chunks(chunks: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def hold_update(streak: np.ndarray, held: np.ndarray, ok: np.ndarray, done_count: int, horizon: np.ndarray,
                active: np.ndarray, hold: int = HOLD_TICKS) -> tuple[np.ndarray, np.ndarray]:
    """One decision's held-success update. ``streak`` counts consecutive on-target ticks; ``held`` takes ``done_count``
    when an active lane's streak reaches ``hold`` at a tick at most ``horizon + hold - 1``, that is, when the streak's
    first tick is inside the horizon. The hold may then finish up to ``hold - 1`` ticks after the horizon."""

    streak = np.where(ok, streak + 1, 0)
    held = np.where((held < 0) & active & (streak >= hold) & (done_count <= horizon + hold - 1), done_count, held)
    return streak, held


def batch_tokens(batch: dict) -> dict:
    """The token sets of a stored batch: only entity and self, which are stored. The other trunk types are absent. The
    encoder and the generalist read an absent type as no valid token, and the flat world fills none of them (no blocks,
    inventory, geometry or kit). The loss equals the one with those four types zero-filled (test_sa_distill.py)."""

    return {name: (batch[name], batch[f"{name}_valid"]) for name in ("entity", "self")}


def wrap_degrees_np(degrees: np.ndarray) -> np.ndarray:
    return np.mod(degrees + 180.0, 360.0) - 180.0


def path_name(path: Sequence[Any]) -> str:
    names = []
    for entry in path:
        if hasattr(entry, "key"):
            names.append(str(entry.key))
        elif hasattr(entry, "name"):
            names.append(str(entry.name))
        elif hasattr(entry, "idx"):
            names.append(str(entry.idx))
        else:
            names.append(str(entry))
    return ".".join(names)


def save_params(path: Path, params: Any) -> None:
    flat = {}
    for entries, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        flat[path_name(entries)] = np.asarray(leaf)
    np.savez(path, **flat)


def load_params(path: Path, template: Any, *, new_from: Any = None) -> Any:
    """Parameters saved by :func:`save_params`, in the structure of ``template`` (an initialised tree of the same
    configuration). Every leaf of the template must be in the file, unless ``new_from`` (an initialised tree of the same
    configuration) is given: a leaf the file lacks, such as a token type added since the file was saved, then takes its
    value from ``new_from`` (S12-a: the ``sound`` type's parameters start at their initial values)."""

    stored = np.load(path)
    fresh = {} if new_from is None else {path_name(entries): leaf for entries, leaf in
                                         jax.tree_util.tree_flatten_with_path(new_from)[0]}
    leaves = jax.tree_util.tree_flatten_with_path(template)[0]
    values = [jnp.asarray(stored[name] if name in stored.files or new_from is None else fresh[name])
              for name in (path_name(entries) for entries, _ in leaves)]
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), values)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def git_output(*args: str) -> str | None:
    """The stdout of ``git <args>`` run in this package's folder, or None when git cannot answer."""

    try:
        done = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
                              check=False)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def provenance() -> dict[str, Any]:
    """What tells one report's run from another's: the commit, whether this package had uncommitted changes (the
    shared tree always has some elsewhere), the command line, the load at the start (``main`` adds the end), and two
    settings this trainer does not set, so E12 runs can be told apart. ``world_capabilities`` is the flat
    environment's default (``tasks/simple_tasks.py``)."""

    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain", "--", str(Path(__file__).resolve().parents[1]))
    return {
        "base_commit": None if head is None else head.strip(),
        "dirty": None if status is None else bool(status.strip()),
        "argv": list(sys.argv),
        "load_average": {"start": list(os.getloadavg()), "end": None},
        "world_capabilities": "open_flat",
        "view_sector_full_angle_radians": None,
        "legal_observation_sensor_range": False,
    }


__all__ = [
    "AGE_SCALE",
    "EYE_XYZ",
    "HEAD_COUNT",
    "HEAD_NAMES",
    "HEAD_SIZES",
    "HEAD_STARTS",
    "HOLD_TICKS",
    "MODE_PARALLEL",
    "NEUTRAL",
    "OTHER_XYZ",
    "POSITION_SCALE",
    "REQUEST_SLOTS",
    "SPAWN_XZ",
    "URGENCY",
    "argument_row",
    "batch_tokens",
    "body_vector",
    "broadcast_rows",
    "chunk",
    "concat_chunks",
    "environment",
    "facing_degrees",
    "git_output",
    "head_vector",
    "hold_update",
    "host_rows",
    "json_default",
    "legal_rows",
    "load_params",
    "path_name",
    "planar",
    "provenance",
    "resample",
    "save_params",
    "token_kernel",
    "wrap_degrees",
    "wrap_degrees_np",
]
