"""Export a long recurrent rollout so Java's GRU can be checked for drift.

**The gap this closes.** Every existing correctness claim against JAX is a
*single step from a zero carry*: `Main` and `EndToEnd` step once, and all 52
`ParityTest` states are evaluated with a fresh zero carry. `HookTest` advances a
carry but only checks that it is private and bounded, never that it is *right*.
`Bench` measures throughput.

So the recurrent path -- the entire reason this network is a GRU -- has never
been compared to JAX beyond step 1. That matters for two reasons:

  * a per-step difference of 1.6e-09 is harmless once and unknown after 900
    ticks, because the carry feeds itself; and
  * the GRU gate order here is non-standard, and a transposed gate can be
    invisible from a zero carry (where the reset gate barely matters) while
    diverging steadily once the state is warm.

An NPC in a real server runs for thousands of ticks with a warm carry, which is
precisely the regime nothing tests.

**Weights come from `case/*.bin`, not from a checkpoint.** The checkpoint that
produced them is not in this workspace (checked: no match across 705 `.npz`
files). Using a *different* checkpoint would confound weight differences with
drift, so this rebuilds `RecurrentPolicyParams` from the same arrays the Java
side loads, and then proves the reconstruction by reproducing
`case/expected_logits.bin` from a zero carry before exporting anything.

Run against the live Gym:

    cd ~/hydl
    # from the repository root (use ":" instead of ";" on POSIX):
    PYTHONPATH="HytaleRL/hytalegym;." \\
        python -u experimental/jvm-agent/tools/export/export_recurrent.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
CASE = JVM_AGENT_ROOT / "case"
PARITY = CASE / "parity"
OUT = CASE / "recurrent"

#: 900 ticks is 30 seconds at 30 TPS -- long enough that a compounding error
#: would have to be very small to stay invisible, short enough to export fast.
STEPS = 900


def _read(name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(CASE / f"{name}.bin", dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise SystemExit(
            f"{name}.bin has {values.size} floats, expected {int(np.prod(shape))}"
        )
    return values.reshape(shape)


def main() -> int:
    import jax.numpy as jnp
    from hytalegym.jax.policy import DenseParams
    from hytalegym.jax.training.types import GRUParams, RecurrentPolicyParams
    from hytalegym.jax.training.policy import apply_policy

    observation_size, hidden = 8271, 32
    actions = 99

    params = RecurrentPolicyParams(
        encoder_input=DenseParams(
            kernel=jnp.asarray(_read("encoder_input_kernel", (observation_size, hidden))),
            bias=jnp.asarray(_read("encoder_input_bias", (hidden,))),
        ),
        encoder_hidden=DenseParams(
            kernel=jnp.asarray(_read("encoder_hidden_kernel", (hidden, hidden))),
            bias=jnp.asarray(_read("encoder_hidden_bias", (hidden,))),
        ),
        gru=GRUParams(
            input_kernel=jnp.asarray(_read("gru_input_kernel", (hidden, 3 * hidden))),
            recurrent_kernel=jnp.asarray(
                _read("gru_recurrent_kernel", (hidden, 3 * hidden))
            ),
            bias=jnp.asarray(_read("gru_bias", (3 * hidden,))),
        ),
        actor=DenseParams(
            kernel=jnp.asarray(_read("actor_kernel", (hidden, actions))),
            bias=jnp.asarray(_read("actor_bias", (actions,))),
        ),
        critic=DenseParams(
            kernel=jnp.asarray(_read("critic_kernel", (hidden, 1))),
            bias=jnp.asarray(_read("critic_bias", (1,))),
        ),
    )

    # Prove the reconstruction before trusting anything built on it. If the
    # PyTree were assembled wrong -- a transposed kernel, a swapped gru kernel --
    # this single step would not reproduce the certified reference.
    reference_observation = _read("observation", (observation_size,))
    reference_mask = np.fromfile(CASE / "action_mask.bin", dtype=np.uint8) != 0
    expected_logits = _read("expected_logits", (actions,))
    _, logits, _ = apply_policy(
        params,
        jnp.asarray(reference_observation)[None],
        jnp.zeros((1, hidden), dtype=jnp.float32),
        jnp.asarray(reference_mask)[None],
    )
    reconstruction_delta = float(
        np.max(np.abs(np.asarray(logits)[0] - expected_logits))
    )
    if reconstruction_delta > 1.0e-6:
        raise SystemExit(
            "rebuilt params do not reproduce case/expected_logits.bin "
            f"(worst {reconstruction_delta:.3e}) -- the PyTree is assembled wrong, "
            "so any drift measured from it would be meaningless"
        )
    print(f"params reconstruction verified: worst {reconstruction_delta:.3e}")

    states = sorted(path for path in PARITY.iterdir() if path.is_dir())
    if not states:
        raise SystemExit("no parity states; run export_parity.py first")

    # Cycle the real, diverse parity observations rather than repeating one.
    # A constant observation drives the GRU to a fixed point, where drift
    # stops accumulating and the test would quietly prove nothing.
    order = [states[index % len(states)] for index in range(STEPS)]

    carry = jnp.zeros((1, hidden), dtype=jnp.float32)
    all_logits = np.zeros((STEPS, actions), dtype=np.float32)
    all_carry = np.zeros((STEPS, hidden), dtype=np.float32)
    all_value = np.zeros((STEPS,), dtype=np.float32)

    for step, state in enumerate(order):
        observation = np.fromfile(
            state / "expected_observation.bin", dtype=np.float32
        )
        if observation.size != observation_size:
            raise SystemExit(f"{state.name} observation width {observation.size}")
        mask_path = state / "action_mask.bin"
        mask = (
            np.fromfile(mask_path, dtype=np.uint8) != 0
            if mask_path.is_file()
            else reference_mask
        )
        carry, logits, value = apply_policy(
            params,
            jnp.asarray(observation)[None],
            carry,
            jnp.asarray(mask)[None],
        )
        all_logits[step] = np.asarray(logits)[0]
        all_carry[step] = np.asarray(carry)[0]
        all_value[step] = float(np.asarray(value)[0])

    # The carry must actually move and stay bounded. A carry pinned at zero, or
    # one that saturates to +/-1 everywhere, would make step-900 agreement
    # trivial rather than meaningful.
    magnitude = np.abs(all_carry)
    moved = int(np.sum(magnitude.max(axis=0) > 1.0e-6))
    if moved < hidden // 2:
        raise SystemExit(
            f"only {moved}/{hidden} carry units ever left zero -- the rollout "
            "does not exercise the recurrent path"
        )
    drift = float(np.max(np.abs(all_carry[-1] - all_carry[0])))
    if drift < 1.0e-4:
        raise SystemExit(
            f"carry barely changed over {STEPS} steps (max {drift:.3e}) -- "
            "agreement at the end would prove nothing"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    all_logits.tofile(OUT / "logits.bin")
    all_carry.tofile(OUT / "carry.bin")
    all_value.tofile(OUT / "value.bin")
    (OUT / "steps.txt").write_text(
        "\n".join(
            ["# step\tstate"]
            + [f"{index}\t{state.name}" for index, state in enumerate(order)]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {STEPS} steps over {len(states)} distinct observations")
    print(f"carry units that moved : {moved}/{hidden}")
    print(f"carry range            : [{all_carry.min():+.4f}, {all_carry.max():+.4f}]")
    print(f"carry drift first->last: {drift:.3e}")
    print(f"value range            : [{all_value.min():+.4f}, {all_value.max():+.4f}]")
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Regenerate the deterministic JVM recurrent trace fixture."
    ).parse_args()
    sys.exit(main())
