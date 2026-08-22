"""Algorithm-neutral parameter checkpoints for arbitrary PyTrees.

``adk.core.lifecycle.save_checkpoint`` is bound to ``RecurrentPolicyParams``
and ``PPOConfig`` because its payload serialisation lives in the Gym's PPO
checkpoint module.  Behaviour cloning, evolutionary search, and scripted
controllers carry neither type, so they could not checkpoint at all.

Only the *payload* was PPO-shaped.  The valuable half -- refusing to write or
restore against a build whose contracts have drifted -- is already
algorithm-neutral, so this module reuses it unchanged and parameterises what is
stored.

Restoring takes a ``like`` template rather than serialising the tree structure.
That keeps the archive a plain ``.npz`` a human can inspect, and it makes a
structural mismatch a loud error instead of a silently rebuilt tree.

Nothing here is specific to parameters.  A batch collected through
``handle.collect`` is also a PyTree, so the same pair stores and reloads
recorded episodes -- which is what an offline or behaviour-cloning agent needs:

    _state, batch = handle.compile_collector(policy, record, steps)(key)
    save_parameters(path, handle.built, batch, run_metadata={"kind": "replay"})

Storing recorded data through the same surface-stamped path is deliberate.  The
learner observation width has moved three times (7818 -> 8259 on 2026-07-31,
8259 -> 8271 on 2026-08-03), and a
replay archive that does not record the surface it was collected on will happily
train a network against a shape that no longer exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from adk.contracts.stamp import require_current
from adk.runtime.env_adapter import BuiltAgent


ADK_PARAMETER_CHECKPOINT_SCHEMA = "hytalerl_adk_parameters_v1"

_LEAVES_NAME = "leaves.npz"
_METADATA_NAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class LoadedParameters:
    """One restored PyTree and the metadata stored beside it."""

    parameters: Any
    metadata: Mapping[str, Any]


def _surface_identity(built: BuiltAgent) -> dict[str, Any]:
    """Capture the surface a restored parameter tree must still match."""

    stamp = built.stamp
    return {
        "observation_size": int(stamp.observation_size),
        "action_head_names": list(stamp.action_head_names),
        "action_head_sizes": [int(size) for size in stamp.action_head_sizes],
        "agent": built.spec.name,
        "scene": built.spec.scene,
    }


def _leaf_keys(parameters: Any) -> list[str]:
    """Stable, human-readable archive keys derived from the tree's key paths."""

    entries, _ = jax.tree_util.tree_flatten_with_path(parameters)
    return [jax.tree_util.keystr(path) for path, _ in entries]


def save_parameters(
    path: str | Path,
    built: BuiltAgent,
    parameters: Any,
    *,
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write an arbitrary parameter PyTree with the current build's identity.

    Refuses to write against a drifted build, exactly as the PPO path does: a
    checkpoint that cannot state which surface produced it is not worth having.
    """

    require_current(built.stamp, built.world_geometry_config)

    leaves, _ = jax.tree_util.tree_flatten(parameters)
    if not leaves:
        raise ValueError("parameters contain no arrays to checkpoint")
    keys = _leaf_keys(parameters)

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / _LEAVES_NAME,
        **{key: np.asarray(leaf) for key, leaf in zip(keys, leaves)},
    )
    metadata = {
        "schema": ADK_PARAMETER_CHECKPOINT_SCHEMA,
        "surface": _surface_identity(built),
        "leaf_keys": keys,
        "run": dict(run_metadata or {}),
    }
    (destination / _METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_parameters(
    path: str | Path,
    built: BuiltAgent,
    like: Any,
) -> LoadedParameters:
    """Restore a parameter PyTree shaped like ``like`` for the current build.

    ``like`` supplies the tree structure; the archive supplies the values.
    """

    require_current(built.stamp, built.world_geometry_config)

    source = Path(path)
    metadata = json.loads(
        (source / _METADATA_NAME).read_text(encoding="utf-8")
    )
    if metadata.get("schema") != ADK_PARAMETER_CHECKPOINT_SCHEMA:
        raise ValueError(
            "not an ADK parameter checkpoint: "
            f"{metadata.get('schema')!r} != {ADK_PARAMETER_CHECKPOINT_SCHEMA!r}"
        )

    stored = metadata.get("surface", {})
    current = _surface_identity(built)
    drift = {
        name: (stored.get(name), current[name])
        for name in current
        if stored.get(name) != current[name]
    }
    if drift:
        raise RuntimeError(
            "checkpoint does not match the current ADK build:\n- "
            + "\n- ".join(
                f"{name}: checkpoint {was!r} != current {now!r}"
                for name, (was, now) in sorted(drift.items())
            )
        )

    template, treedef = jax.tree_util.tree_flatten(like)
    keys = list(metadata.get("leaf_keys", ()))
    if len(template) != len(keys):
        raise ValueError(
            f"template has {len(template)} leaves but the checkpoint stored "
            f"{len(keys)}; restore with the same tree structure"
        )
    expected = _leaf_keys(like)
    if expected != keys:
        raise ValueError(
            "template tree structure does not match the checkpoint:\n- "
            + "\n- ".join(
                f"position {index}: template {mine!r} != checkpoint {theirs!r}"
                for index, (mine, theirs) in enumerate(zip(expected, keys))
                if mine != theirs
            )
        )

    with np.load(source / _LEAVES_NAME) as archive:
        restored = [jax.numpy.asarray(archive[key]) for key in keys]
    return LoadedParameters(
        parameters=jax.tree_util.tree_unflatten(treedef, restored),
        metadata=metadata,
    )
