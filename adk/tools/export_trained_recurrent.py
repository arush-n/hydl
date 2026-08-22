"""Regenerate the 900-step recurrent reference for a **trained** checkpoint.

`RecurrentTest` proves Java's GRU tracks JAX for 900 steps -- but against the
explicitly synthetic shipped fixture. Its near-uniform logits do not exercise
the paths that only separate under stronger activations, so a green run there
is a statement about arithmetic, not about a trained policy. The fixture's
actor-kernel standard deviation is diagnostic context, not training evidence.

The observations it replays (`case/parity/<state>/expected_observation.bin`) are
**weight-independent**, so they can be reused as-is. Only the JAX reference
depends on the weights. This writes a new `recurrent/` directory for a trained
checkpoint, leaving the shared `case/` fixture untouched:

    python -m adk.tools.export_trained_recurrent <checkpoint.npz> <out-dir>

then, from `experimental/jvm-agent/`:

    java -cp "out;$SERVER_JAR" RecurrentTest <trained-case> case/parity <out-dir>

Reuses captured observations, so it never opens a native session and cannot
disturb the evidence lease on 127.0.0.1:5556.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
JVM_AGENT = _ROOT / "experimental" / "jvm-agent"
CASE = JVM_AGENT / "case"
PARITY = CASE / "parity"
STEPS = CASE / "recurrent" / "steps.txt"


def read_steps() -> list[str]:
    """Parity state names, in the order the fixture replays them."""

    names = []
    for line in STEPS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(
                f"steps.txt row has {len(parts)} fields, expected 2: {line!r}")
        names.append(parts[1])
    return names


def build(checkpoint: Path, destination: Path) -> dict[str, object]:
    import jax.numpy as jnp
    from hytalegym.jax.training.checkpoint import load_policy_checkpoint
    from hytalegym.jax.training.policy import apply_policy

    params, config, _ = load_policy_checkpoint(checkpoint)
    names = read_steps()

    # `RecurrentTest.java:121` prefers each parity state's OWN action_mask and
    # only falls back to the case-root mask when a state has none. Generating
    # the reference with the shared mask instead produced 1e9 logit deltas on
    # exactly the bits the two masks disagreed about -- a harness bug that reads
    # precisely like a port failure. Mirror the Java rule.
    shared = np.fromfile(CASE / "action_mask.bin", dtype=np.uint8).astype(bool)

    def mask_for(state: Path) -> np.ndarray:
        own = state / "action_mask.bin"
        if own.is_file():
            return np.fromfile(own, dtype=np.uint8).astype(bool)
        return shared

    carry = jnp.zeros((1, config.recurrent_size), dtype=jnp.float32)
    carries, logits_all, values = [], [], []
    for name in names:
        path = PARITY / name / "expected_observation.bin"
        if not path.is_file():
            raise FileNotFoundError(
                f"parity state {name!r} from steps.txt has no observation at "
                f"{path}; the fixture and steps.txt disagree")
        observation = np.fromfile(path, dtype="<f4")
        carry, logits, value = apply_policy(
            params,
            jnp.asarray(observation)[None],
            carry,
            jnp.asarray(mask_for(PARITY / name))[None],
        )
        carries.append(np.asarray(carry)[0])
        logits_all.append(np.asarray(logits)[0])
        values.append(float(np.asarray(value)[0]))

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "steps.txt").write_text(
        STEPS.read_text(encoding="utf-8"), encoding="utf-8")
    for label, array in (("carry", np.stack(carries)),
                         ("logits", np.stack(logits_all)),
                         ("value", np.asarray(values))):
        np.ascontiguousarray(array, dtype="<f4").tofile(
            destination / f"{label}.bin")

    stacked = np.stack(carries)
    return {
        "destination": str(destination),
        "steps": len(names),
        "recurrent_size": int(config.recurrent_size),
        "encoder_size": int(config.encoder_size),
        # A carry that has collapsed to zero would make the whole comparison
        # trivial -- the GRU is contractive, so this is the number that says
        # the 900 steps were actually carrying state.
        "carry_abs_mean_final": float(np.abs(stacked[-1]).mean()),
        "carry_abs_max": float(np.abs(stacked).max()),
        "logit_abs_max": float(np.abs(np.stack(logits_all)).max()),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    report = build(Path(argv[0]), Path(argv[1]))
    for key, value in report.items():
        print(f"{key:22} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
