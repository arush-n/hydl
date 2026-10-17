"""Stage A of J-S1: supervised fit of the heads on decision rows (SYSTEM_ONE.md 8.4, FORMULAS (M5)).

The trunk is frozen and reads text on the slow path only, so every distinct executive
text is encoded once (``h_E``, gathered per row) and the catalogue once (``e_j``); only
the heads train. Loss over a batch of rows, s~ the masked skill logits::

    L_A = CE(s~, y_skill) + BCE(r, y_replace) + lambda_p * mean_rows ||p - y_params||^2,   lambda_p = 1

The matching skill head (default) trains W_q and W_k with the adapter, the fusion and the
other heads; ``--skill-head fixed`` trains the fixed W_s head instead (the ablation).
Adam with global-norm clipping at 0.5 and eps 1e-5 (the ``train_j1.py`` optimizer),
batches cut from a fresh permutation each epoch (``train_j1._endless_batches``).
The plan head is not trained: no dataset carries the next K_plan skill calls.
The confidence and value heads get no gradient here and keep their initial zeros
(stage B fits the confidence head; stage C is the PPO harness).

Rows: source 1 is ``--data`` under ``--executive`` (default: the CompositionTask
executive, ``tasks/composition.py``); ``--variants N`` adds source 2, every ``--data``
belief row under each training variant (``data.executive_variants``), one encoding per
variant. ``--heldout PATH`` prints held-out agreement with J0 (gate G1); agreement on
held-out variants (the G3 split) is printed when any variant is held out, on the
``--heldout`` belief rows when given. Skills are described by their ids.

Goal reading (gate G3, SYSTEM_ONE.md 8.9, FORMULAS (M9) to (M11)): ``--goal-reading N`` generates N
states over placeholder catalogues (``data.goal_reading_states``), splits them
(``data.goal_reading_split``: K <= 5 train or held-out states, K in {6, 7, 8} the held-out
catalogue sizes), pairs each with ``--rows-per-state`` belief rows (default 32; drawn from
``--data`` for training and from ``--heldout``, when given, for the two test splits), labels them
with the teacher (M10), and trains heads with the structured executive encoder (S = SLOT_WIDTH)
on source 1 plus the training states' rows (plus any ``--variants``). ``--text-path off`` drops
h_E from the fusion (the "text off" ablation); it needs ``--goal-reading``. Rows of every
catalogue are encoded per row: the slot inputs of (M9) (``data.slot_inputs``) and the row's
catalogue embeddings padded to the widest K; the encoder runs inside the loss, so W_slot and w_a
train with the heads. Printed after training: train, held-out recorded rows (``--heldout``),
held-out states and held-out catalogue sizes, each with skill and replace agreement.

Commands (CPU)::

    python -m experimental.neural_agent.jev.system_one.train_supervised \\
        --data experimental/neural_agent/eval/out/traces/train/dataset.npz \\
        --heldout experimental/neural_agent/eval/out/traces/heldout/dataset.npz \\
        --trunk hashed --out /tmp/js1/heads.npz
    python -m experimental.neural_agent.jev.system_one.train_supervised \\
        --data experimental/neural_agent/eval/out/traces/train/dataset.npz \\
        --heldout experimental/neural_agent/eval/out/traces/heldout/dataset.npz \\
        --trunk hashed --goal-reading 256 --text-path off --out /tmp/js1_g3/heads_struct.npz
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.contracts import ExecutiveState
from experimental.neural_agent.jev.lm.prompting import validate_skill_ids
from experimental.neural_agent.jev.system_one.data import (
    SupervisedRows,
    catalogue_embeddings,
    concat_rows,
    executive_variants,
    goal_reading_rows,
    goal_reading_split,
    goal_reading_states,
    rows_from_dataset,
    slot_inputs,
    variant_rows,
)
from experimental.neural_agent.jev.system_one.heads import (
    SKILL_HEADS,
    SLOT_SCALARS,
    SLOT_WIDTH,
    HeadsOutput,
    HeadsParams,
    apply_heads,
    encode_executive,
    heads_dims,
    init_heads,
)
from experimental.neural_agent.jev.system_one.trunk import HashedBagTrunk, TorchTransformerTrunk, TrunkEncoder
from experimental.neural_agent.jev.train_j1 import (
    ADAM_EPSILON,
    MAX_GRADIENT_NORM,
    _endless_batches,
    save_metrics,
)

LAMBDA_PARAMS: float = 1.0
DEFAULT_SKILL_IDS: tuple[str, ...] = ("idle", "pursuit", "evader")
# Measured 2026-10-17 on the recorded rows: the matching head spikes at lr 3e-3 and 1e-3
# (late-batch loss up to 2.6 and 1.2); 3e-4 for 4,000 updates is stable for both heads.
DEFAULT_UPDATES: int = 4000
DEFAULT_BATCH: int = 256
DEFAULT_LR: float = 3.0e-4
DEFAULT_ROWS_PER_STATE: int = 32
_ROW_ARRAYS: tuple[str, ...] = ("u", "allowed_mask", "current_onehot", "y_skill", "y_replace", "y_params")


class EncodedRows(NamedTuple):
    """Rows after the frozen trunk: h float32[N, D] (the row's h_E; None without the text path),
    the SupervisedRows arrays, and for the structured encoder slots float32[N, G, 2D + 4 + P] (the
    row's slot inputs, ``data.slot_inputs``) and skill_embeddings float32[N, K, D] (the row's
    catalogue, zero rows past its size); both None otherwise."""

    h: np.ndarray | None
    u: np.ndarray
    allowed_mask: np.ndarray
    current_onehot: np.ndarray
    y_skill: np.ndarray
    y_replace: np.ndarray
    y_params: np.ndarray
    slots: np.ndarray | None = None
    skill_embeddings: np.ndarray | None = None


def encode_rows(rows: SupervisedRows, trunk: TrunkEncoder, *, structured: bool = False, text: bool = True) -> EncodedRows:
    """Encode each distinct executive text once (slow path) and give every row its h_E.

    With ``structured``, also give every row its slot inputs and its catalogue's embeddings: each
    distinct catalogue and each distinct constraint label is encoded once. Without ``text`` the
    texts are not encoded and h is None.
    """

    arrays = [getattr(rows, name) for name in _ROW_ARRAYS]
    h = None
    if text:
        table = np.asarray(trunk.encode(list(rows.executive_texts)), dtype=np.float32)
        h = table[rows.executive_index]
    if not structured:
        return EncodedRows(h, *arrays)
    labels = sorted({label for item in rows.executives for goal in item.state.goals for label in goal.constraints})
    label_table = dict(zip(labels, np.asarray(trunk.encode(labels), dtype=np.float32))) if labels else {}
    width = rows.allowed_mask.shape[1]
    catalogues: dict[tuple[tuple[str, ...], tuple[str, ...]], np.ndarray] = {}
    skill_tables, slot_tables = [], []
    for item in rows.executives:
        key = (item.skill_ids, item.descriptions)
        if key not in catalogues:
            catalogues[key] = catalogue_embeddings(trunk, item.skill_ids, item.descriptions)
        skill_tables.append(np.pad(catalogues[key], ((0, width - len(item.skill_ids)), (0, 0))))
        slot_tables.append(slot_inputs(item.state, item.skill_ids, catalogues[key], label_table))
    index = rows.executive_index
    return EncodedRows(h, *arrays, slots=np.stack(slot_tables)[index], skill_embeddings=np.stack(skill_tables)[index])


def heads_on_rows(params: HeadsParams, rows: Mapping[str, Any], skill_embeddings: Any = None) -> HeadsOutput:
    """Run the heads on a mapping of encoded rows (the EncodedRows fields that are not None).

    A head with the structured encoder encodes ``rows["slots"]`` first (inside the computation, so
    its weights train). The rows' own ``skill_embeddings`` are used when present, else the shared
    ``skill_embeddings`` float32[K, D].

    Raises:
        ValueError: If the head has the encoder and the rows have no slots, or as ``apply_heads``.
    """

    executive = None
    if heads_dims(params).slot_width:
        if rows.get("slots") is None:
            raise ValueError("heads with the structured encoder need rows encoded with structured=True")
        executive = encode_executive(params, rows["slots"])
    table = rows.get("skill_embeddings")
    return apply_heads(
        params, rows.get("h"), rows["u"], rows["current_onehot"], rows["allowed_mask"],
        skill_embeddings if table is None else table, executive,
    )


def _present(encoded: EncodedRows) -> dict[str, np.ndarray]:
    """Return the EncodedRows fields that are not None."""

    return {name: value for name, value in encoded._asdict().items() if value is not None}


def supervised_loss(
    params: HeadsParams, batch: Mapping[str, Any], skill_embeddings: Any = None
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return L_A of a batch (module formula) and its metrics.

    Metrics: skill_loss, replace_loss, params_loss, skill_accuracy (argmax s~ = y_skill)
    and replace_accuracy ([r > 0] = y_replace).
    """

    out = heads_on_rows(params, batch, skill_embeddings)
    y_skill = jnp.asarray(batch["y_skill"], dtype=jnp.int32)
    y_replace = jnp.asarray(batch["y_replace"], dtype=jnp.bool_)
    skill_loss = optax.softmax_cross_entropy_with_integer_labels(out.skill_logits, y_skill).mean()
    replace_loss = optax.sigmoid_binary_cross_entropy(out.replace_logit, y_replace.astype(jnp.float32)).mean()
    params_loss = jnp.mean(jnp.sum((out.params - batch["y_params"]) ** 2, axis=-1))
    metrics = {
        "skill_loss": skill_loss,
        "replace_loss": replace_loss,
        "params_loss": params_loss,
        "skill_accuracy": jnp.mean(jnp.argmax(out.skill_logits, axis=-1) == y_skill),
        "replace_accuracy": jnp.mean((out.replace_logit > 0.0) == y_replace),
    }
    return skill_loss + replace_loss + LAMBDA_PARAMS * params_loss, metrics


def train_heads(
    encoded: EncodedRows,
    *,
    updates: int,
    batch: int,
    lr: float,
    key: jax.Array,
    skill_head: str = "matching",
    skill_embeddings: Any = None,
    slot_width: int = 0,
    text_path: bool = True,
) -> tuple[HeadsParams, list[dict[str, float]]]:
    """Fit fresh heads to ``encoded`` with Adam on :func:`supervised_loss`.

    Args:
        encoded: The rows after the trunk.
        updates: Gradient steps.
        batch: Rows per step.
        lr: Adam learning rate.
        key: PRNG key for the initialisation and the batch order.
        skill_head: ``"matching"`` (default) or ``"fixed"``.
        skill_embeddings: float32[K, D], the catalogue's e_j; required by the matching head
            unless the rows carry their own (``encode_rows(..., structured=True)``).
        slot_width: S of the structured encoder, 0 for none (``init_heads``).
        text_path: Whether h_E is in the fusion (``init_heads``).

    Returns:
        ``(params, history)``; ``history`` has one dict per update (``update`` 1-based,
        ``loss`` and the metrics), measured on that update's batch before its step.

    Raises:
        ValueError: If ``updates`` or ``lr`` is not positive, ``batch`` is outside [1, N],
            or the matching head has no embeddings.
    """

    rows = int(encoded.u.shape[0])
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError(f"updates must be a positive integer, got {updates!r}")
    if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= rows:
        raise ValueError(f"batch must be an integer in [1, {rows}], got {batch!r}")
    if not lr > 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    table = encoded.skill_embeddings if encoded.skill_embeddings is not None else skill_embeddings
    if skill_head == "matching" and table is None:
        raise ValueError("the matching head needs skill_embeddings float32[K, D]")
    init_key, batch_key = jax.random.split(key)
    dim = int(np.shape(table)[-1]) if skill_head == "matching" else int(encoded.h.shape[1])
    skills = int(encoded.allowed_mask.shape[1])
    params = init_heads(
        init_key, dim, skills if skill_head == "fixed" else None, skill_head=skill_head,
        slot_width=slot_width, text_path=text_path,
    )
    embeddings = None if skill_head == "fixed" or skill_embeddings is None else jnp.asarray(skill_embeddings, dtype=jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(MAX_GRADIENT_NORM), optax.adam(float(lr), eps=ADAM_EPSILON))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def step(current, state, data, table):
        (loss, metrics), gradients = jax.value_and_grad(supervised_loss, has_aux=True)(current, data, table)
        step_updates, state = optimizer.update(gradients, state, current)
        return optax.apply_updates(current, step_updates), state, {"loss": loss, **metrics}

    batches = _endless_batches(_present(encoded), rows, batch, batch_key)
    history: list[dict[str, float]] = []
    for index in range(updates):
        params, optimizer_state, stats = step(params, optimizer_state, next(batches), embeddings)
        history.append({"update": index + 1, **{name: float(value) for name, value in stats.items()}})
    return params, history


def evaluate_heads(params: HeadsParams, encoded: EncodedRows, skill_embeddings: Any = None) -> dict[str, float]:
    """Return rows, skill_agreement, replace_agreement and params_mse of ``params`` on ``encoded``."""

    out = jax.jit(heads_on_rows)(params, _present(encoded), skill_embeddings)
    skill = np.asarray(jnp.argmax(out.skill_logits, axis=-1))
    return {
        "rows": int(skill.shape[0]),
        "skill_agreement": float(np.mean(skill == encoded.y_skill)),
        "replace_agreement": float(np.mean((np.asarray(out.replace_logit) > 0.0) == encoded.y_replace)),
        "params_mse": float(np.mean(np.sum((np.asarray(out.params) - encoded.y_params) ** 2, axis=-1))),
    }


def save_heads(path: str | Path, params: HeadsParams) -> None:
    """Write the HeadsParams leaves to an .npz, one array per field."""

    heads_dims(params)
    np.savez(path, **{name: np.asarray(leaf) for name, leaf in zip(HeadsParams._fields, params)})


def load_heads(path: str | Path) -> HeadsParams:
    """Read the HeadsParams that :func:`save_heads` wrote, checking the keys and every shape.

    A file written before the structured encoder (no w_slot, b_slot, w_a) loads as a head without it.

    Raises:
        ValueError: If the file's keys differ from the HeadsParams fields, or a shape is inconsistent.
    """

    encoder = ("w_slot", "b_slot", "w_a")
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if set(arrays) == set(HeadsParams._fields) - set(encoder) and np.ndim(arrays["w_k"]) == 2:
        rows = 2 * np.shape(arrays["w_k"])[0] + SLOT_SCALARS
        arrays.update(w_slot=np.zeros((rows, 0)), b_slot=np.zeros((0,)), w_a=np.zeros((0,)))
    if set(arrays) != set(HeadsParams._fields):
        raise ValueError(f"{path} must hold exactly the HeadsParams fields, got {sorted(arrays)}")
    params = HeadsParams(**{name: jnp.asarray(arrays[name], dtype=jnp.float32) for name in HeadsParams._fields})
    heads_dims(params)
    return params


def parse_skills(text: str) -> tuple[str, ...]:
    """argparse type: a comma-separated catalogue, idle first by convention."""

    try:
        return validate_skill_ids([part.strip() for part in text.split(",") if part.strip()])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def make_trunk(spec: str) -> TrunkEncoder:
    """Return the trunk named by ``hashed`` or ``torch:<model>``.

    Raises:
        ImportError: For ``torch:<model>`` when ``transformers`` is not installed.
        ValueError: For any other spec.
    """

    if spec == "hashed":
        return HashedBagTrunk()
    if spec.startswith("torch:") and len(spec) > len("torch:"):
        return TorchTransformerTrunk(spec[len("torch:") :])
    raise ValueError(f"--trunk must be 'hashed' or 'torch:<model>', got {spec!r}")


def load_executive(path: Path | None, skill_ids: Sequence[str]) -> ExecutiveState:
    """Return E_t from ``executive.json`` (``ExecutiveState.to_json``), or the CompositionTask one."""

    if path is not None:
        return ExecutiveState.from_json(Path(path).read_text(encoding="utf-8"))
    from experimental.neural_agent.tasks.composition import CompositionTask

    return CompositionTask(tick_budget=512).initial_executive_state(skill_ids)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Return every array of an .npz dataset."""

    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def _agreement_line(label: str, metrics: Mapping[str, float]) -> str:
    return (
        f"{label}: {metrics['rows']} rows, skill_agreement {metrics['skill_agreement']:.4f}, "
        f"replace_agreement {metrics['replace_agreement']:.4f}, params_mse {metrics['params_mse']:.6f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Train the heads (stage A), write them and metrics.csv, print the last metrics and agreements."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.train_supervised",
        description="Stage A of J-S1 (CPU): fit the heads on decision rows with a frozen trunk.",
    )
    parser.add_argument("--data", type=Path, required=True, help="train_j1 dataset .npz (recorded J0 decisions)")
    parser.add_argument("--executive", type=Path, default=None, help="executive.json; default the CompositionTask E_t")
    parser.add_argument("--skills", type=parse_skills, default=DEFAULT_SKILL_IDS, help="catalogue, default idle,pursuit,evader")
    parser.add_argument("--trunk", default="hashed", help="hashed (default) or torch:<model>")
    parser.add_argument("--skill-head", choices=SKILL_HEADS, default="matching", help="matching (default) or fixed")
    parser.add_argument("--variants", type=int, default=0, help="executive variants added as source 2 (default 0)")
    parser.add_argument(
        "--goal-reading", type=int, default=None, metavar="N",
        help="goal-reading mode: N generated states over placeholder catalogues, structured encoder (gate G3)",
    )
    parser.add_argument(
        "--rows-per-state", type=int, default=DEFAULT_ROWS_PER_STATE,
        help=f"belief rows per goal-reading state (default {DEFAULT_ROWS_PER_STATE})",
    )
    parser.add_argument(
        "--text-path", choices=("on", "off"), default="on",
        help="h_E in the fusion (default on); off is the text-off ablation and needs --goal-reading",
    )
    parser.add_argument("--heldout", type=Path, default=None, help="held-out dataset .npz, scored after training")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES, help=f"Adam updates (default {DEFAULT_UPDATES})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"rows per update (default {DEFAULT_BATCH})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Adam learning rate (default {DEFAULT_LR})")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0)")
    parser.add_argument("--out", type=Path, required=True, help="output heads .npz; metrics.csv is written next to it")
    args = parser.parse_args(argv)
    if args.variants < 0:
        parser.error("--variants must be non-negative")
    structured = args.goal_reading is not None
    text = args.text_path == "on"
    if structured and (args.goal_reading < 1 or args.rows_per_state < 1):
        parser.error("--goal-reading and --rows-per-state must be positive")
    if structured and args.skill_head == "fixed":
        parser.error("--goal-reading needs the matching head: the fixed head cannot score other catalogue sizes")
    if not text and not structured:
        parser.error("--text-path off needs the structured encoder of --goal-reading")
    try:
        trunk = make_trunk(args.trunk)
    except (ImportError, ValueError) as error:
        parser.error(str(error))

    executive = load_executive(args.executive, args.skills)
    dataset = load_dataset(args.data)
    heldout = None if args.heldout is None else load_dataset(args.heldout)
    shared = args.skill_head == "matching" and not structured
    embeddings = catalogue_embeddings(trunk, args.skills) if shared else None
    variant_key, train_key = jax.random.split(jax.random.key(args.seed))
    variants = executive_variants(executive, args.skills, variant_key, args.variants)
    train_variants = [variant for variant in variants if not variant.heldout]
    test_variants = [variant for variant in variants if variant.heldout]
    parts = [rows_from_dataset(dataset, executive, args.skills)]
    if train_variants:
        parts.append(variant_rows(dataset, train_variants, args.skills))
    reading: dict[str, tuple[int, SupervisedRows]] = {}
    if structured:
        states_key, rows_key = jax.random.split(jax.random.fold_in(jax.random.key(args.seed), 1))
        train_states, heldout_states, heldout_sizes = goal_reading_split(goal_reading_states(states_key, args.goal_reading))
        train_rows_key, heldout_rows_key, sizes_rows_key = jax.random.split(rows_key, 3)
        print(
            f"goal reading: {args.goal_reading} states, {len(train_states)} train (K <= 5), "
            f"{len(heldout_states)} held-out states (K <= 5), {len(heldout_sizes)} held-out catalogue sizes (K in 6..8), "
            f"{args.rows_per_state} rows per state, text path {args.text_path}"
        )
        if train_states:
            parts.append(goal_reading_rows(train_states, dataset, train_rows_key, args.rows_per_state))
        beliefs = dataset if heldout is None else heldout
        for label, chosen, rows_key_split in (
            ("held-out states", heldout_states, heldout_rows_key),
            ("held-out catalogue sizes", heldout_sizes, sizes_rows_key),
        ):
            if chosen:
                reading[label] = (len(chosen), goal_reading_rows(chosen, beliefs, rows_key_split, args.rows_per_state))
    train = encode_rows(concat_rows(parts), trunk, structured=structured, text=text)

    params, history = train_heads(
        train, updates=args.updates, batch=args.batch, lr=args.lr, key=train_key,
        skill_head=args.skill_head, skill_embeddings=embeddings,
        slot_width=SLOT_WIDTH if structured else 0, text_path=text,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_heads(args.out, params)
    metrics_path = args.out.with_name("metrics.csv")
    save_metrics(metrics_path, history)
    last = history[-1]
    print(
        f"{args.skill_head} head, update {int(last['update'])}: loss {last['loss']:.4f}, "
        f"skill_accuracy {last['skill_accuracy']:.3f}, replace_accuracy {last['replace_accuracy']:.3f}, "
        f"params_loss {last['params_loss']:.6f}"
    )
    states_note = f" + {len(train_states)} goal-reading states" if structured else ""
    print(_agreement_line(
        f"train (source 1 + {len(train_variants)} training variants{states_note})", evaluate_heads(params, train, embeddings)
    ))
    if heldout is not None:
        scored = encode_rows(rows_from_dataset(heldout, executive, args.skills), trunk, structured=structured, text=text)
        print(_agreement_line(f"heldout {args.heldout}", evaluate_heads(params, scored, embeddings)))
    if test_variants:
        rows = variant_rows(dataset if heldout is None else heldout, test_variants, args.skills)
        print(_agreement_line(
            f"held-out variants ({len(test_variants)})",
            evaluate_heads(params, encode_rows(rows, trunk, structured=structured, text=text), embeddings),
        ))
    for label, (count, rows) in reading.items():
        scored = encode_rows(rows, trunk, structured=True, text=text)
        print(_agreement_line(f"{label} ({count} states)", evaluate_heads(params, scored)))
    print(f"heads {args.out}, metrics {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
