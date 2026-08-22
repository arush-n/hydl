"""End-to-end port check: train in JAX, run the weights in Java, compare.

The `jvm-agent` suite runs green against a fixture exported from near-zero
weights (its `actor_kernel` has std ~0.0014). That proves the arithmetic; it
does not prove that a *trained* policy survives the port, because near-uniform
logits never exercise the paths that only diverge under real activations.

This builds a case directory around trained weights and hands it to the same
Java `Main` the suite uses. It deliberately reuses the fixture's already
captured live observation and mask, so **it never opens a native session** and
cannot disturb the evidence lease on 127.0.0.1:5556.

    python -m adk.tools.port_fidelity <checkpoint.npz> [output-dir] [fixture]

Then, from `experimental/jvm-agent/`:

    java -cp "out;$SERVER_JAR" Main <output-dir>

The Java loader derives encoder and recurrent widths independently from the
exported tensors and validates the complete matrix set. This tool therefore
accepts either equal or unequal widths; the Java/JAX comparison on the exact
checkpoint remains the portability gate.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from ..deploy.tensor_identity import POLICY_TENSOR_NAMES

#: Written flat and little-endian in the same canonical order the deployment
#: identity hashes. A model-layout change therefore cannot update the writer
#: without also changing the fail-closed bundle contract.
WEIGHTS = POLICY_TENSOR_NAMES

_ROOT = Path(__file__).resolve().parents[2]
#: Source of the real captured observation. Copied, never regenerated.
FIXTURE = _ROOT / "experimental" / "jvm-agent" / "case"


def build(
    checkpoint: Path,
    destination: Path,
    *,
    fixture: Path | None = None,
) -> dict[str, object]:
    """Write a Java-loadable case directory from a trained checkpoint."""

    # Checked before the JAX import, which costs seconds: a missing fixture is
    # a caller mistake and should not be paid for at framework-load speed.
    fixture = FIXTURE if fixture is None else Path(fixture)
    observation_path = fixture / "observation.bin"
    if not observation_path.is_file():
        observation_path = fixture / "expected_observation.bin"
    mask_path = fixture / "action_mask.bin"
    if not observation_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(
            f"no complete observation/action-mask or exported profile at {fixture}; "
            "pass fixture= explicitly rather than opening a native session"
        )

    import jax.numpy as jnp
    from hytalegym.jax.training.checkpoint import load_policy_checkpoint
    from hytalegym.jax.training.policy import apply_policy

    raw = np.load(checkpoint)
    params, config, metadata = load_policy_checkpoint(checkpoint)
    transfer_contract = metadata.get("transfer_contract")
    if transfer_contract is None and all(
        name in metadata
        for name in (
            "observation_contract_sha256",
            "action_contract_sha256",
            "arsenal_contract_sha256",
            "combat_dynamics_contract_sha256",
        )
    ):
        # Population checkpoint v1 writes its selected portable row using the
        # canonical checkpoint loader's flat metadata shape. Ordinary PPO
        # checkpoints historically nested the same contract. Preserve both
        # without fabricating fields: the bundle layer applies the unchanged
        # current-contract validator to whichever representation was actually
        # stored.
        transfer_contract = metadata

    destination.mkdir(parents=True, exist_ok=True)
    observation = np.fromfile(observation_path, dtype="<f4")
    mask = np.fromfile(mask_path, dtype=np.uint8).astype(bool)
    expected = (config.observation_size, config.action_size)
    actual = (int(observation.size), int(mask.size))
    if actual != expected:
        raise ValueError(
            f"fixture policy surface {actual} does not match checkpoint {expected}"
        )
    shutil.copyfile(observation_path, destination / "observation.bin")
    shutil.copyfile(mask_path, destination / "action_mask.bin")

    recurrent = jnp.zeros((1, config.recurrent_size), dtype=jnp.float32)
    next_recurrent, logits, value = apply_policy(
        params,
        jnp.asarray(observation)[None],
        recurrent,
        jnp.asarray(mask)[None],
    )

    for name in WEIGHTS:
        (destination / f"{name}.bin").write_bytes(
            np.ascontiguousarray(raw[name], dtype="<f4").tobytes())
    for name, array in (("expected_logits", np.asarray(logits)[0]),
                        ("expected_recurrent", np.asarray(next_recurrent)[0])):
        np.ascontiguousarray(array, dtype="<f4").tofile(
            destination / f"{name}.bin")

    return {
        "destination": str(destination),
        "fixture": str(fixture),
        "fixture_observation": str(observation_path),
        "encoder_size": int(config.encoder_size),
        "recurrent_size": int(config.recurrent_size),
        "observation_nonzero": int((observation != 0).sum()),
        "legal_actions": int(mask.sum()),
        # Retained as useful activation/debugging telemetry. It is not training
        # evidence: narrow trained and broad synthetic policies both exist.
        "actor_std": float(np.std(raw["actor_kernel"])),
        "jax_value": float(np.asarray(value)[0]),
        "transfer_contract": transfer_contract,
        # Carried into the deployment manifest as provenance.  Weight
        # dispersion is not evidence of optimization: trained policies can
        # remain narrow, while synthetic diagnostics can be deliberately
        # broad.  The bundle layer validates explicit counters instead.
        "checkpoint_metadata": metadata,
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    checkpoint = Path(argv[0])
    destination = Path(argv[1]) if len(argv) > 1 else checkpoint.parent / "case"
    fixture = Path(argv[2]) if len(argv) > 2 else None
    report = build(checkpoint, destination, fixture=fixture)
    for key, value in report.items():
        print(f"{key:20} {value}")
    print("\nnow run, from experimental/jvm-agent/:")
    print(f'  java -cp "out;$SERVER_JAR" Main "{report["destination"]}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
