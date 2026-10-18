"""Certify a Console-trained checkpoint as deployable, or refuse and say why.

`arena.training.runs.pursuit_run` writes every milestone with

    "live_policy_compatible": False,
    "promotion_status": "not_assessed",
    "transfer_contract": source_metadata.get("transfer_contract"),

which are conservative defaults meaning "nobody has checked", not "this is
broken". Nothing in the pursuit path ever checked, and the third line inherits
from the source checkpoint -- so a lineage that started without a contract
carries `null` forever, and `adk.deploy.bundle` refuses it with

    checkpoint is not deployable under the current policy contract:
    checkpoint has no exact transfer_contract

This is the missing contract assessment. It records exact file lineage for the
derived checkpoint, but does NOT assert training semantic provenance,
competence, or live behavior: it builds the live contract from the Gym,
compares it against the widths the checkpoint was actually trained at, and
stamps it only when they agree. A mismatch is an error, not a warning, because
the whole point of the field is that a policy trained on a different
observation width would otherwise load and produce confident garbage. Shape
and contract agreement alone are not a training-semantic claim.

    python -m adk.deploy.certify IN.npz OUT.npz --agent-role Kweebec_Razorleaf

The input is never modified; certification writes a new checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


CERTIFICATION_LINEAGE_SCHEMA = (
    "hytalerl_checkpoint_certification_lineage_v1"
)
_CHECKPOINT_TENSOR_DOMAIN = b"HYTALERL_CHECKPOINT_TENSORS\0"
_LINEAGE_METADATA_NAMES = frozenset({"config_json", "metadata_json"})


def _load(path: Path) -> tuple[dict, dict, dict]:
    with np.load(path, allow_pickle=False) as raw:
        tensors = {name: raw[name] for name in raw.files}
        config = json.loads(str(raw["config_json"]))
        metadata = json.loads(str(raw["metadata_json"]))
    return tensors, config, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _tensor_content_sha256(tensors: dict[str, np.ndarray]) -> str:
    """Hash every non-JSON checkpoint array independent of NPZ packaging."""

    names = sorted(set(tensors) - _LINEAGE_METADATA_NAMES)
    digest = hashlib.sha256(_CHECKPOINT_TENSOR_DOMAIN)
    digest.update(struct.pack(">II", 1, len(names)))
    for name in names:
        array = np.asarray(tensors[name])
        if array.dtype.hasobject:
            raise ValueError(f"checkpoint tensor {name!r} has object dtype")
        encoded_name = name.encode("utf-8")
        encoded_dtype = array.dtype.str.encode("ascii")
        content = np.ascontiguousarray(array).tobytes(order="C")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", array.ndim))
        for dimension in array.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return digest.hexdigest().upper()


def _training_context(source: Path, metadata: dict) -> dict:
    """Assemble the owner rows the run already recorded but never collected.

    `adk.deploy.compatibility.normalize_training_context` wants `policy_id` and
    one owner per actor with `entity_index`, `profile`, `team_id`, `controller`
    and `trainable`. None of that is missing from a pursuit run -- it is spread
    across `training_execution` in the checkpoint and the run's own
    `launch-spec.json`, and simply never assembled into one field.

    RECORDED, read straight through:
      entity_index  <- training_execution.full_actor_slots
      trainable     <- training_execution.record_policy_slots
      profile       <- launch-spec loadout / opponent
      controller    <- launch-spec target_controller

    DERIVED, and the one judgement here: `team_id` is not written down
    anywhere. A pursuit run is one duel between two opposing actors, so slots
    are numbered onto opposing teams. If a run ever has more than two actors or
    same-team actors this is wrong and must be recorded properly instead.
    """

    spec_path = source.parents[2] / "launch-spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    execution = metadata.get("training_execution") or {}

    slots = int(execution.get("full_actor_slots") or 2)
    trainable_slots = set(execution.get("record_policy_slots") or [0])
    learner_profile = str(spec.get("loadout") or "unknown")
    opponent_profile = str(spec.get("opponent") or learner_profile)
    opponent_controller = str(spec.get("target_controller") or "unknown")

    owners = []
    for index in range(slots):
        trainable = index in trainable_slots
        owners.append({
            "entity_index": index,
            "profile": learner_profile if trainable else opponent_profile,
            "team_id": index,
            "controller": "learner" if trainable else opponent_controller,
            "trainable": trainable,
        })
    return {
        "schema": "hytalerl_policy_training_context_v1",
        "policy_id": 0,
        "owners": owners,
    }


def certify(
    source: Path,
    destination: Path,
    *,
    agent_role: str,
    target_role: str | None = None,
) -> dict:
    """Stamp the live contract onto `source`, or raise if it does not fit."""

    source = Path(source)
    destination = Path(destination)
    same_file = source.resolve() == destination.resolve()
    if not same_file and source.exists() and destination.exists():
        same_file = source.samefile(destination)
    if same_file:
        raise ValueError(
            "certification destination must be different from the source "
            "checkpoint"
        )

    from hytalegym.jax.combat.observation.v3 import (
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
    )
    from hytalegym.jax.training.checkpoint import (
        ARSENAL_POLICY_SURFACE,
        combat_checkpoint_contract_errors,
        current_combat_checkpoint_contract,
    )

    tensors, config, metadata = _load(source)
    checkpoint_observation = int(config["observation_size"])
    checkpoint_action = int(config["action_size"])
    checkpoint_heads = [int(h) for h in config["action_head_sizes"]]

    contract = current_combat_checkpoint_contract(
        policy_surface=ARSENAL_POLICY_SURFACE,
    )

    # The assessment. A checkpoint trained at another width is exactly what the
    # deployment gate exists to stop, so this refuses rather than warns.
    mismatches = []
    if int(contract["observation_size"]) != checkpoint_observation:
        mismatches.append(
            f"observation {checkpoint_observation} != live "
            f"{contract['observation_size']}"
        )
    if int(contract["action_size"]) != checkpoint_action:
        mismatches.append(
            f"action {checkpoint_action} != live {contract['action_size']}"
        )
    live_heads = [int(h) for h in ARSENAL_POLICY_ACTION_HEAD_SIZES]
    if live_heads != checkpoint_heads:
        mismatches.append(f"head sizes {checkpoint_heads} != live {live_heads}")
    if mismatches:
        raise SystemExit(
            "checkpoint does not speak the live contract, refusing to certify:\n  "
            + "\n  ".join(mismatches)
        )

    # `build_deployment_compatibility` reads `transfer_contract.agent_role`, and
    # a wildcard role is reserved for explicitly controlled diagnostics.
    contract["agent_role"] = agent_role
    contract["target_role"] = target_role or agent_role

    errors = combat_checkpoint_contract_errors(
        contract,
        arsenal_runtime_capacity=contract.get("arsenal_runtime_capacity"),
        policy_surface=ARSENAL_POLICY_SURFACE,
    )
    if errors:
        raise SystemExit(
            "the live contract itself failed validation:\n  " + "\n  ".join(errors)
        )

    lineage = {
        "schema": CERTIFICATION_LINEAGE_SCHEMA,
        "source_checkpoint": str(source.resolve()),
        "source_sha256": _sha256(source),
        "config_sha256": _canonical_json_sha256(config),
        "tensors_sha256": _tensor_content_sha256(tensors),
    }
    certified = dict(metadata)
    certified["transfer_contract"] = contract
    if not certified.get("policy_training_context"):
        certified["policy_training_context"] = _training_context(source, metadata)
    certified["promotion_status"] = "certified_contract_only"
    certified["certified_from"] = source.name
    certified["certification_lineage"] = lineage
    # Deliberately NOT set true. This certifies the CONTRACT the weights speak;
    # it is not a competence or live-behaviour claim, and the PPO checkpoints
    # that already carry a transfer_contract leave this false too.
    certified["live_policy_compatible"] = metadata.get(
        "live_policy_compatible", False
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    out = dict(tensors)
    out["metadata_json"] = np.asarray(json.dumps(certified))
    np.savez(destination, **out)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--agent-role", required=True)
    parser.add_argument("--target-role", default=None)
    args = parser.parse_args()

    contract = certify(
        args.source,
        args.destination,
        agent_role=args.agent_role,
        target_role=args.target_role,
    )
    print(f"certified {args.source.name} -> {args.destination}")
    print(f"  observation {contract['observation_size']}  "
          f"action {contract['action_size']}")
    print(f"  agent_role  {contract['agent_role']}")
    print(f"  observation_contract_sha256 "
          f"{contract['observation_contract_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
