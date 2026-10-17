"""B2's smoke: a tiny behaviour-cloning body on the native 0.7.2 traces, held out by world (``BUILD.md`` 7b, B2).

The body is the B1 trunk (``encoder.py``) over the dataset's self and entity tokens, with one linear head per
supervised head of the gen-7 layout (base action, locomotion compass, yaw and pitch deltas: the heads the
native projection labels). The loss is the cross-entropy over supervised entries only::

    L = sum_{rows r, heads h with s[r, h]} -log softmax(W_h c_r + b_h)[a[r, h]]  /  max(1, sum s)

where ``c_r`` is the trunk's context for row r, ``a`` the native label and ``s`` the supervision mask. The
smoke trains on one world's traces and reports on the other's (``BCDataset.split``):

    agreement[h]   = held-out accuracy on supervised rows, beside the majority-class baseline of the training
                     rows (a body that always answers the most common choice)
    jitter[h]      = fraction of consecutive held-out rows of one trace whose choice changes, for the
                     prediction and for the native labels (a cloned body should not flicker more than the NPC)

Finally the trained network is registered as a skill's body and run through the combiner on held-out frames
(the body path check): the factors the combiner emits must equal the network's own argmax. This proves the
trace -> token -> factor -> body -> combiner pipeline end to end; it is a smoke, not a skill.

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.training.bc_smoke [--steps 1500] [--batch 256] [--seed 0] [--dataset NPZ] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.skill_agent.combiner import AgentContext, BodyStep
from experimental.neural_agent.skill_agent.encoder import EncoderConfig, apply_encoder, init_encoder
from experimental.neural_agent.skill_agent.library import SkillLibrary
from experimental.neural_agent.skill_agent.surface import EntityToken, SelfToken, head_layout
from experimental.neural_agent.skill_agent.training.bc_dataset import DEFAULT_OUT, BCDataset, build_dataset

TOKEN_TYPES: tuple[tuple[str, int], ...] = (("entity", len(EntityToken.FEATURES)), ("self", len(SelfToken.FEATURES)))
"""The token sets the dataset carries, in the trunk's order."""
BOX = ((-1.0e6, -64.0, -1.0e6), (1.0e6, 512.0, 1.0e6))


def _heads(dataset: BCDataset, rows: np.ndarray) -> tuple[str, ...]:
    """The heads with at least one supervised training row."""

    supervised = dataset.supervised[rows].any(axis=0)
    return tuple(name for name, on in zip(dataset.head_names, supervised) if on)


def init_body(key: jax.Array, config: EncoderConfig, heads: Sequence[str]) -> dict[str, Any]:
    """The trunk's parameters and one linear head per name."""

    sizes = {head.name: head.size for head in head_layout("gen7")}
    keys = jax.random.split(key, len(heads) + 1)
    out = {"trunk": init_encoder(keys[0], config)}
    for index, name in enumerate(heads):
        weight = jax.random.normal(keys[index + 1], (config.output, sizes[name])) / np.sqrt(config.output)
        out[name] = (weight, jnp.zeros((sizes[name],)))
    return out


def logits_of(params: Mapping[str, Any], batch: Mapping[str, Any], config: EncoderConfig,
              heads: Sequence[str]) -> dict[str, jax.Array]:
    context = apply_encoder(params["trunk"], batch, config).context
    return {name: context @ params[name][0] + params[name][1] for name in heads}


def _loss(params, batch, actions, supervised, config, heads, columns):
    logits = logits_of(params, batch, config, heads)
    total, count = 0.0, 0.0
    for name, column in zip(heads, columns):
        log_p = jax.nn.log_softmax(logits[name])
        picked = jnp.take_along_axis(log_p, actions[:, column][:, None], axis=1)[:, 0]
        mask = supervised[:, column].astype(jnp.float32)
        total = total - jnp.sum(picked * mask)
        count = count + jnp.sum(mask)
    return total / jnp.maximum(count, 1.0)


class ClonedBody:
    """The trained network as a skill body: ``(request, ticks_since_start, observation_tokens, belief_tokens)``.

    ``observation_tokens`` is the trunk's batch for this one agent (``BCDataset.token_batch`` of one row); the
    body fills the heads it was trained on with its argmax and never reports done (a cloned fight runs until
    the middle layer withdraws it).
    """

    def __init__(self, params: Mapping[str, Any], config: EncoderConfig, heads: Sequence[str]) -> None:
        self.heads = tuple(heads)
        self._predict = jax.jit(lambda batch: logits_of(params, batch, config, self.heads))

    def __call__(self, request: Any, ticks_since_start: int, observation_tokens: Any, belief_tokens: Any) -> BodyStep:
        logits = self._predict(observation_tokens)
        return BodyStep({name: int(jnp.argmax(logits[name][0])) for name in self.heads})


def run_smoke(dataset: BCDataset, *, steps: int = 1500, batch_size: int = 256, seed: int = 0,
              held_out_world: str = "hytale_generator", checks: int = 32) -> dict[str, Any]:
    """Train on one world, report on the other, then run the body path check. Returns the report."""

    train, test = dataset.split(held_out_world=held_out_world)
    heads = _heads(dataset, train)
    columns = tuple(dataset.head_names.index(name) for name in heads)
    config = EncoderConfig(token_types=TOKEN_TYPES, width=64, heads=4, layers=2, output=128)
    params = init_body(jax.random.PRNGKey(seed), config, heads)
    optimizer = optax.adam(1e-3)
    state = optimizer.init(params)
    actions = jnp.asarray(dataset.actions)
    supervised = jnp.asarray(dataset.supervised)

    @jax.jit
    def update(params, state, batch, rows):
        loss, grads = jax.value_and_grad(_loss)(params, batch, actions[rows], supervised[rows], config, heads, columns)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    losses = []
    for _ in range(steps):
        rows = rng.choice(train, size=batch_size)
        params, state, loss = update(params, state, dataset.token_batch(rows), jnp.asarray(rows))
        losses.append(float(loss))
    train_seconds = time.perf_counter() - started

    def predict(rows: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, rows.size, 4096):
            chunk = rows[start:start + 4096]
            logits = logits_of(params, dataset.token_batch(chunk), config, heads)
            out.append(np.stack([np.asarray(jnp.argmax(logits[name], axis=-1)) for name in heads], axis=1))
        return np.concatenate(out, axis=0)

    order = test[np.lexsort((dataset.tick[test], dataset.trace_index[test]))]
    predicted = predict(order)
    train_predicted = predict(train)
    report_heads = {}
    for k, (name, column) in enumerate(zip(heads, columns)):
        on = dataset.supervised[order, column]
        labels = dataset.actions[order, column]
        train_on = dataset.supervised[train, column]
        majority = int(np.bincount(dataset.actions[train, column][train_on]).argmax()) if train_on.any() else 0
        same_trace = dataset.trace_index[order][1:] == dataset.trace_index[order][:-1]
        pair = same_trace & on[1:] & on[:-1]
        report_heads[name] = {
            "held_out_rows": int(on.sum()),
            "agreement": float((predicted[on, k] == labels[on]).mean()) if on.any() else None,
            "majority_baseline": float((labels[on] == majority).mean()) if on.any() else None,
            "train_agreement": float((train_predicted[train_on, k] == dataset.actions[train, column][train_on]).mean())
            if train_on.any() else None,
            "jitter_predicted": float((predicted[1:, k] != predicted[:-1, k])[pair].mean()) if pair.any() else None,
            "jitter_native": float((labels[1:] != labels[:-1])[pair].mean()) if pair.any() else None,
        }

    library = SkillLibrary()
    body = ClonedBody(params, config, heads)
    library.create("cloned_fight", description="fight the marked enemy as the recorded NPCs did",
                   scopes=("locomotion", "gaze", "hands"), required=("entity",), done="enemy down", body=body)
    combiner = library.combiner("gen7")
    names = combiner.head_names
    picks = order[np.linspace(0, order.size - 1, num=min(checks, order.size)).astype(int)]
    mismatches = 0
    for agent, row in enumerate(picks):
        request = library.request("cloned_fight", agent=agent, request_id=0, tick=0, entity=7)
        context = AgentContext(world_bounds=BOX, known_entities=frozenset({7}),
                               observation_tokens=dataset.token_batch(np.asarray([row])))
        step = combiner.step(0, agent, [request], context)
        direct = predict(np.asarray([row]))[0]
        mismatches += int(any(step.factors[names.index(name)] != int(direct[k]) for k, name in enumerate(heads)))
    return {
        "schema": "skill_agent_bc_smoke_v1",
        "held_out_world": held_out_world,
        "train_rows": int(train.size), "test_rows": int(test.size),
        "heads": report_heads,
        "steps": steps, "batch": batch_size, "seed": seed,
        "loss_first_last": [losses[0], float(np.mean(losses[-50:]))],
        "train_seconds": round(train_seconds, 2),
        "body_path_check": {"frames": int(picks.size), "mismatches": mismatches},
        "receipt": dict(dataset.receipt),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 smoke: behaviour cloning on the native traces.")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-world", default="hytale_generator")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUT,
                        help="a saved dataset (bc_dataset's CLI writes it); built from the traces when absent")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    dataset = BCDataset.load_npz(args.dataset) if args.dataset.is_file() else build_dataset()
    report = run_smoke(dataset, steps=args.steps, batch_size=args.batch, seed=args.seed,
                       held_out_world=args.held_out_world)
    text = json.dumps(report, indent=1, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


__all__ = ["ClonedBody", "init_body", "logits_of", "run_smoke"]


if __name__ == "__main__":
    raise SystemExit(main())
