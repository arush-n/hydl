"""Behaviour cloning of J1 from recorded J0 decisions (ADR-001 D2).

ScriptedJev (J0, ``scripted.py``) makes one goal decision per lane from the
belief B_t and the executive state E_t. This module fits J1 (``network.py``) to
a dataset of those decisions by supervised learning: the skill, and the replace
decision. It does not fit the goal parameters. ``init_j1`` zeroes the params
head, and the dataset carries no target parameters, so that head is not trained.

Dataset: an .npz, or a dict, with one row per decision. N is the row count and
K = ``num_skills`` the catalog size.

    features          float32[N, 10]   belief features, columns BELIEF_FEATURE_NAMES
    feature_known     bool[N, 10]
    latent            float32[N, 64]
    exec_skill_index  int32[N, 4]      catalog index, -1 in an empty slot
    exec_priority     float32[N, 4]
    exec_params       float32[N, 4, 4]
    exec_goal_valid   bool[N, 4]
    exec_budget       float32[N, 3]
    exec_version      int32[N]
    current_skill     int32[N]         skill in force before the decision
    target_skill      int32[N]         skill that J0 chose
    target_replace    bool[N]          target_skill != current_skill
    num_skills        scalar integer   K

Loss for one row, with skill logits z masked to ``j1_allowed_skills`` and replace
logit r::

    loss = cross_entropy(z, target_skill) + binary_cross_entropy(r, target_replace)

Commands::

    python -m experimental.neural_agent.jev.train_j1 --synthetic 4096 --out data.npz
    python -m experimental.neural_agent.jev.train_j1 --data data.npz --out params.npz
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    BELIEF_LATENT_WIDTH,
    MAX_GOALS,
    PARAM_WIDTH,
    SKILL_IDLE,
    TICKS_SATURATION,
    ComputeBudget,
    ExecutiveArrays,
    GoalPayload,
    SharedBeliefState,
    feature_index,
)
from experimental.neural_agent.jev.network import (
    BUDGET_WIDTH,
    REPLACE_THRESHOLD,
    J1Params,
    apply_j1,
    init_j1,
    j1_allowed_skills,
    j1_inputs,
)
from experimental.neural_agent.jev.scripted import (
    EVADE_DAMAGE_WINDOW_TICKS,
    EVADER_SKILL_ID,
)

# Same values as the optimizer of train_j1_ppo.py.
MAX_GRADIENT_NORM: float = 0.5
ADAM_EPSILON: float = 1.0e-5
LEARNING_RATE: float = 1.0e-2  # measured 2026-10-17: 3e-4 plateaus near 0.83 skill accuracy on the synthetic set
# Logit given to a skill outside the mask. Finite, as in train_j1_ppo.py.
_MASKED_LOGIT: float = -1.0e9
# Share of executive goal slots that hold a goal in a synthetic row.
SYNTHETIC_GOAL_PROBABILITY: float = 0.75

# Per-row fields of a dataset: name -> (dtype, shape after the leading row axis).
FIELDS: dict[str, tuple[type, tuple[int, ...]]] = {
    "features": (np.float32, (BELIEF_FEATURE_COUNT,)),
    "feature_known": (np.bool_, (BELIEF_FEATURE_COUNT,)),
    "latent": (np.float32, (BELIEF_LATENT_WIDTH,)),
    "exec_skill_index": (np.int32, (MAX_GOALS,)),
    "exec_priority": (np.float32, (MAX_GOALS,)),
    "exec_params": (np.float32, (MAX_GOALS, PARAM_WIDTH)),
    "exec_goal_valid": (np.bool_, (MAX_GOALS,)),
    "exec_budget": (np.float32, (BUDGET_WIDTH,)),
    "exec_version": (np.int32, ()),
    "current_skill": (np.int32, ()),
    "target_skill": (np.int32, ()),
    "target_replace": (np.bool_, ()),
}
# Catalog-index fields and the lowest value each may take. The highest is K - 1.
_INDEX_FIELDS: dict[str, int] = {
    "exec_skill_index": -1,
    "current_skill": 0,
    "target_skill": 0,
}


def _positive_int(value: Any, name: str) -> int:
    """Return ``value`` as an int of at least 1. Bools raise TypeError."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return int(value)


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    """Check every field of a dataset and raise at the first fault.

    Args:
        dataset: A mapping with the fields of ``FIELDS`` and the scalar
            ``num_skills``. Values may be numpy or JAX arrays.

    Raises:
        ValueError: If a field is missing, has the wrong shape or dtype, does not
            match the row count of ``features``, or holds a catalog index outside
            its range. The message names the field.
    """

    for name in (*FIELDS, "num_skills"):
        if name not in dataset:
            raise ValueError(f"dataset is missing field {name!r}")
    num_skills = np.asarray(dataset["num_skills"])
    if (
        num_skills.shape != ()
        or not np.issubdtype(num_skills.dtype, np.integer)
        or num_skills < 1
    ):
        raise ValueError(f"num_skills must be a scalar integer >= 1, got {num_skills!r}")
    shape = np.shape(dataset["features"])
    if not shape or shape[0] < 1:
        raise ValueError("features must have at least one row on its leading axis")
    rows = shape[0]
    for name, (dtype, trailing) in FIELDS.items():
        array = np.asarray(dataset[name])
        expected = (rows, *trailing)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
        if array.dtype != np.dtype(dtype):
            raise ValueError(
                f"{name} must have dtype {np.dtype(dtype).name}, got {array.dtype.name}"
            )
        if name in _INDEX_FIELDS:
            low, high = _INDEX_FIELDS[name], int(num_skills)
            if array.min() < low or array.max() >= high:
                raise ValueError(
                    f"{name} must hold values in [{low}, {high}), "
                    f"got values from {array.min()} to {array.max()}"
                )


def _endless_batches(
    arrays: dict[str, np.ndarray], rows: int, batch: int, key: jax.Array
) -> Iterator[dict[str, np.ndarray]]:
    """Yield full batches forever, one fresh random permutation per epoch."""

    while True:
        key, epoch_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(epoch_key, rows))
        for start in range(0, rows - batch + 1, batch):
            index = order[start : start + batch]
            yield {name: array[index] for name, array in arrays.items()}


def make_dataset_batches(
    dataset: Mapping[str, Any], batch: int, key: jax.Array
) -> Iterator[dict[str, np.ndarray]]:
    """Return an endless iterator of shuffled batches of ``batch`` rows.

    Each epoch is a fresh permutation drawn from ``key``, cut into full batches.
    The rows left after the last full batch of an epoch are skipped for that epoch.
    The dataset is validated when this function is called, not on the first ``next``.

    Args:
        dataset: A dataset that passes :func:`validate_dataset`.
        batch: Rows per batch, a positive int no larger than the row count.
        key: A JAX PRNG key for the epoch permutations.

    Returns:
        An iterator of dicts that hold the per-row fields of ``FIELDS`` only, each
        array with leading axis ``batch``.

    Raises:
        ValueError: If the dataset is invalid or ``batch`` is outside [1, N].
    """

    validate_dataset(dataset)
    rows = int(np.shape(dataset["features"])[0])
    size = _positive_int(batch, "batch")
    if size > rows:
        raise ValueError(f"batch must not exceed the {rows} rows, got {size}")
    arrays = {name: np.asarray(dataset[name]) for name in FIELDS}
    return _endless_batches(arrays, rows, size, key)


def _records(
    batch: Mapping[str, Any],
) -> tuple[SharedBeliefState, ExecutiveArrays, GoalPayload]:
    """Build the three contract records of a batch. The goal params are zeros, the slot -1."""

    lanes = np.shape(batch["current_skill"])[0]
    belief = SharedBeliefState(
        features=batch["features"],
        feature_known=batch["feature_known"],
        latent=batch["latent"],
    )
    executive = ExecutiveArrays(
        skill_index=batch["exec_skill_index"],
        priority=batch["exec_priority"],
        params=batch["exec_params"],
        goal_valid=batch["exec_goal_valid"],
        budget=batch["exec_budget"],
        version=batch["exec_version"],
    )
    current_goal = GoalPayload(
        skill_index=batch["current_skill"],
        params=jnp.zeros((lanes, PARAM_WIDTH), jnp.float32),
        executive_version=jnp.zeros((lanes,), jnp.int32),
        goal_slot=jnp.full((lanes,), -1, jnp.int32),
    )
    return belief, executive, current_goal


def build_inputs(batch: Mapping[str, Any], *, num_skills: int) -> jax.Array:
    """Return the J1 input matrix float32[N, D] of a batch, built by ``j1_inputs``.

    Args:
        batch: A dict of per-row arrays, as :func:`make_dataset_batches` yields.
        num_skills: Catalog size K. It is keyword-only because a batch does not carry it.

    Returns:
        The input matrix, with D = ``network.input_width(K)``.
    """

    belief, executive, current_goal = _records(batch)
    return j1_inputs(belief, executive, current_goal, num_skills=num_skills)


def bc_loss(
    params: J1Params, batch: Mapping[str, Any]
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the behaviour-cloning loss of a batch and its metrics.

    The skill logits are masked to ``j1_allowed_skills``, as ``NetworkJev`` masks
    them, so the cross-entropy ranges over the skills J1 may pick. The replace
    term is the binary cross-entropy of the replace logit.

    Args:
        params: J1Params. K is read from ``params.w_skill``.
        batch: A dict of per-row arrays, as :func:`make_dataset_batches` yields.

    Returns:
        ``(loss, metrics)``. ``loss`` is ``skill_loss + replace_loss``. ``metrics``
        holds ``skill_loss``, ``replace_loss``, ``skill_accuracy`` (the fraction of
        rows whose masked argmax equals ``target_skill``) and ``replace_accuracy``
        (the fraction of rows whose sigmoid of the replace logit, compared with
        ``REPLACE_THRESHOLD``, matches ``target_replace``).
    """

    num_skills = int(params.w_skill.shape[1])
    belief, executive, current_goal = _records(batch)
    x = j1_inputs(belief, executive, current_goal, num_skills=num_skills)
    skill_logits, replace_logit, _ = apply_j1(params, x)
    allowed = j1_allowed_skills(executive, num_skills=num_skills)
    masked = jnp.where(allowed, skill_logits, _MASKED_LOGIT)
    target_skill = jnp.asarray(batch["target_skill"], dtype=jnp.int32)
    target_replace = jnp.asarray(batch["target_replace"], dtype=jnp.bool_)
    skill_loss = optax.softmax_cross_entropy_with_integer_labels(
        masked, target_skill
    ).mean()
    replace_loss = optax.sigmoid_binary_cross_entropy(
        replace_logit, target_replace.astype(jnp.float32)
    ).mean()
    replace_prediction = jax.nn.sigmoid(replace_logit) > REPLACE_THRESHOLD
    metrics = {
        "skill_loss": skill_loss,
        "replace_loss": replace_loss,
        "skill_accuracy": jnp.mean(jnp.argmax(masked, axis=-1) == target_skill),
        "replace_accuracy": jnp.mean(replace_prediction == target_replace),
    }
    return skill_loss + replace_loss, metrics


def train_j1(
    dataset: Mapping[str, Any],
    *,
    updates: int,
    batch: int,
    lr: float,
    key: jax.Array,
    num_skills: int,
) -> tuple[J1Params, list[dict[str, float]]]:
    """Fit J1 to a dataset with Adam on :func:`bc_loss`.

    The optimizer is Adam with the learning rate ``lr`` and global-norm clipping at
    ``MAX_GRADIENT_NORM``. Each update takes one batch, in the order of
    :func:`make_dataset_batches`.

    Args:
        dataset: A dataset that passes :func:`validate_dataset`.
        updates: Number of gradient steps, a positive int.
        batch: Rows per step, a positive int no larger than the row count.
        lr: Adam learning rate, a positive number.
        key: A JAX PRNG key for the initialization and the batch order.
        num_skills: Catalog size K. It must equal the dataset's ``num_skills``.

    Returns:
        ``(params, history)``. ``history`` holds one dict per update: ``update``
        (1-based), ``loss``, ``skill_loss``, ``replace_loss``, ``skill_accuracy`` and
        ``replace_accuracy``, measured on that update's batch before its step.

    Raises:
        ValueError: If ``num_skills`` differs from the dataset, ``updates`` is not
            positive, or ``lr`` is not positive.
    """

    validate_dataset(dataset)
    dataset_skills = int(np.asarray(dataset["num_skills"]))
    if num_skills != dataset_skills:
        raise ValueError(
            f"num_skills={num_skills} does not match the dataset's {dataset_skills}"
        )
    count = _positive_int(updates, "updates")
    if not lr > 0.0:
        raise ValueError(f"lr must be positive, got {lr}")

    init_key, batch_key = jax.random.split(key)
    params = init_j1(init_key, num_skills=num_skills)
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRADIENT_NORM),
        optax.adam(float(lr), eps=ADAM_EPSILON),
    )
    optimizer_state = optimizer.init(params)

    @jax.jit
    def step(current, state, data):
        (loss, metrics), gradients = jax.value_and_grad(bc_loss, has_aux=True)(
            current, data
        )
        step_updates, state = optimizer.update(gradients, state, current)
        current = optax.apply_updates(current, step_updates)
        return current, state, {"loss": loss, **metrics}

    batches = make_dataset_batches(dataset, batch, batch_key)
    history: list[dict[str, float]] = []
    for index in range(count):
        params, optimizer_state, stats = step(params, optimizer_state, next(batches))
        row = {"update": index + 1}
        row.update({name: float(value) for name, value in stats.items()})
        history.append(row)
    return params, history


def synthetic_catalog(num_skills: int) -> tuple[str, ...]:
    """Return the catalog of the synthetic dataset: idle at 0, the evader at 1, then skill_2 on."""

    return (SKILL_IDLE, EVADER_SKILL_ID) + tuple(
        f"skill_{index}" for index in range(2, num_skills)
    )


def scripted_target_skill(
    features: Any,
    feature_known: Any,
    skill_index: Any,
    priority: Any,
    goal_valid: Any,
    *,
    evader_index: int,
) -> np.ndarray:
    """Return the skill that J0 chooses for each row, computed in numpy.

    This is the rule of ``ScriptedJev.__call__`` on the same inputs. It is written
    out so that the synthetic labels need no ScriptedJev call. The tests check it
    against ScriptedJev.

    Args:
        features: float32[N, 10], with the damage counter published as
            counter / TICKS_SATURATION.
        feature_known: bool[N, 10].
        skill_index: int32[N, 4], the executive skill indices.
        priority: float32[N, 4].
        goal_valid: bool[N, 4].
        evader_index: Catalog index of the evader skill.

    Returns:
        int32[N]. The evader in the evade window, else the highest-priority valid
        goal (the lowest slot wins a tie), else 0 (idle).
    """

    column = feature_index("ticks_since_damage")
    ticks = np.asarray(features, np.float32)[:, column] * np.float32(TICKS_SATURATION)
    known = np.asarray(feature_known, np.bool_)[:, column]
    skill = np.asarray(skill_index, np.int32)
    values = np.asarray(priority, np.float32)
    valid = np.asarray(goal_valid, np.bool_)
    is_evader = valid & (skill == evader_index)
    evading = known & (ticks < EVADE_DAMAGE_WINDOW_TICKS) & is_evader.any(axis=1)
    evader_slot = np.argmax(np.where(is_evader, values, -np.inf), axis=1)
    best_slot = np.argmax(np.where(valid, values, -np.inf), axis=1)
    slot = np.where(evading, evader_slot, best_slot)
    has_goal = evading | valid.any(axis=1)
    rows = np.arange(skill.shape[0])
    return np.where(has_goal, skill[rows, slot], 0).astype(np.int32)


def synthetic_dataset(
    key: jax.Array, n: int, num_skills: int = 3
) -> dict[str, np.ndarray]:
    """Return ``n`` decisions whose target is J0's choice on random states.

    The catalog is :func:`synthetic_catalog`, so the evader is index 1. Half of
    the rows have ``ticks_since_damage`` inside the evade window, and half outside
    it, so both branches of the rule are exercised. The other belief features are
    drawn from plausible ranges, and all are known. A goal slot holds a goal with
    probability ``SYNTHETIC_GOAL_PROBABILITY``. Each goal names a skill in 1..K-1,
    with priority U(0, 1). The latent is zero: the rule does not read it, and random
    noise there is a per-row fingerprint that a fit can memorize. The goal parameters
    are zero, and ``exec_version`` is 1.
    ``target_skill`` is :func:`scripted_target_skill` on the row, and
    ``target_replace`` is ``target_skill != current_skill``.

    Args:
        key: A JAX PRNG key.
        n: Number of rows, a positive int.
        num_skills: Catalog size K, at least 2 (idle and an evader).

    Returns:
        A dict with every field of ``FIELDS`` and the scalar ``num_skills``.

    Raises:
        TypeError: If ``n`` or ``num_skills`` is not an integer.
        ValueError: If ``n`` is not positive or ``num_skills`` is below 2.
    """

    rows = _positive_int(n, "n")
    skills = _positive_int(num_skills, "num_skills")
    if skills < 2:
        raise ValueError("num_skills must be at least 2: idle, and an evader to label")
    keys = jax.random.split(key, 15)
    window = int(EVADE_DAMAGE_WINDOW_TICKS)
    saturation = int(TICKS_SATURATION)

    inside = jax.random.bernoulli(keys[0], 0.5, (rows,))
    damage_ticks = jnp.where(
        inside,
        jax.random.randint(keys[1], (rows,), 0, window),
        jax.random.randint(keys[2], (rows,), window, saturation + 1),
    ).astype(jnp.float32)
    angle = jax.random.uniform(keys[3], (rows,), minval=-math.pi, maxval=math.pi)
    columns = [
        jax.random.bernoulli(keys[4], 0.5, (rows,)),  # target_visible
        jax.random.uniform(keys[5], (rows,), maxval=64.0),  # target_distance_blocks
        jnp.sin(angle),  # target_bearing_sin
        jnp.cos(angle),  # target_bearing_cos
        jax.random.uniform(keys[6], (rows,)),  # self_health_fraction
        jax.random.uniform(keys[7], (rows,)),  # target_health_fraction
        damage_ticks / TICKS_SATURATION,  # ticks_since_damage
        jax.random.randint(keys[8], (rows,), 0, saturation + 1)
        / TICKS_SATURATION,  # ticks_since_target_seen
        jax.random.uniform(keys[9], (rows,), maxval=0.5),  # self_speed
        jax.random.uniform(keys[10], (rows,), minval=-0.5, maxval=0.5),  # closing_speed
    ]
    features = jnp.stack(
        [jnp.asarray(column, dtype=jnp.float32) for column in columns], axis=1
    )
    valid = jax.random.bernoulli(keys[11], SYNTHETIC_GOAL_PROBABILITY, (rows, MAX_GOALS))
    named = jax.random.randint(keys[12], (rows, MAX_GOALS), 1, skills)
    priority = jnp.where(valid, jax.random.uniform(keys[13], (rows, MAX_GOALS)), 0.0)
    current = jax.random.randint(keys[14], (rows,), 0, skills)

    budget = ComputeBudget()
    arrays: dict[str, np.ndarray] = {
        "features": np.asarray(features, np.float32),
        "feature_known": np.ones((rows, BELIEF_FEATURE_COUNT), np.bool_),
        "latent": np.zeros((rows, BELIEF_LATENT_WIDTH), np.float32),
        "exec_skill_index": np.asarray(jnp.where(valid, named, -1), np.int32),
        "exec_priority": np.asarray(priority, np.float32),
        "exec_params": np.zeros((rows, MAX_GOALS, PARAM_WIDTH), np.float32),
        "exec_goal_valid": np.asarray(valid, np.bool_),
        "exec_budget": np.tile(
            np.asarray(
                [
                    budget.jev_latency_ms,
                    budget.search_expansions,
                    budget.executive_calls_per_minute,
                ],
                np.float32,
            ),
            (rows, 1),
        ),
        "exec_version": np.ones((rows,), np.int32),
        "current_skill": np.asarray(current, np.int32),
    }
    target = scripted_target_skill(
        arrays["features"],
        arrays["feature_known"],
        arrays["exec_skill_index"],
        arrays["exec_priority"],
        arrays["exec_goal_valid"],
        evader_index=synthetic_catalog(skills).index(EVADER_SKILL_ID),
    )
    arrays["target_skill"] = target
    arrays["target_replace"] = target != arrays["current_skill"]
    arrays["num_skills"] = np.asarray(skills, np.int32)
    return arrays


def save_params(path: str | Path, params: J1Params) -> None:
    """Write the leaves of ``params`` to an .npz, one array per J1Params field."""

    np.savez(path, **{name: np.asarray(leaf) for name, leaf in zip(J1Params._fields, params)})


def load_params(path: str | Path) -> J1Params:
    """Read the J1Params that :func:`save_params` wrote."""

    with np.load(path) as archive:
        return J1Params(**{name: jnp.asarray(archive[name]) for name in J1Params._fields})


def save_metrics(path: str | Path, history: Sequence[Mapping[str, float]]) -> None:
    """Write the per-update rows of :func:`train_j1` to a CSV with a header line."""

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main(argv: Sequence[str] | None = None) -> int:
    """Write a synthetic dataset, or train J1 on a dataset and write its params and metrics.

    Args:
        argv: Command-line arguments without the program name, or None for
            ``sys.argv[1:]``.

    Returns:
        Process exit status 0.
    """

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.train_j1",
        description=(
            "Behaviour cloning of J1 from J0 decisions (CPU). With --synthetic N, "
            "write a synthetic dataset of N rows to --out and exit. Otherwise train "
            "on --data, write the parameters to --out, and write metrics.csv next to it."
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    parser.add_argument("--data", type=Path, help="dataset .npz to train on")
    parser.add_argument(
        "--synthetic",
        type=int,
        metavar="N",
        help="write a synthetic dataset of N rows to --out instead of training",
    )
    parser.add_argument("--updates", type=int, default=200, help="Adam updates (default 200)")
    parser.add_argument("--batch", type=int, default=4096, help="rows per update (default 4096, full-batch on the synthetic set)")
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help=f"Adam learning rate (default {LEARNING_RATE})"
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0)")
    args = parser.parse_args(argv)

    key = jax.random.key(args.seed)
    if args.synthetic is not None:
        dataset = synthetic_dataset(key, args.synthetic)
        np.savez(args.out, **dataset)
        print(f"wrote {args.synthetic} synthetic rows to {args.out}")
        return 0
    if args.data is None:
        parser.error("--data is required unless --synthetic is given")
    with np.load(args.data) as archive:
        dataset = {name: archive[name] for name in archive.files}
    validate_dataset(dataset)
    params, history = train_j1(
        dataset,
        updates=args.updates,
        batch=args.batch,
        lr=args.lr,
        key=key,
        num_skills=int(dataset["num_skills"]),
    )
    save_params(args.out, params)
    metrics_path = args.out.with_name("metrics.csv")
    save_metrics(metrics_path, history)
    last = history[-1]
    print(
        f"update {int(last['update'])}: loss {last['loss']:.4f}, "
        f"skill_accuracy {last['skill_accuracy']:.3f}, "
        f"replace_accuracy {last['replace_accuracy']:.3f}; "
        f"params {args.out}, metrics {metrics_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
