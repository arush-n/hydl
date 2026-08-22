"""Export exact recipe-encoder parameters and non-vacuous projection rows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = ROOT / "case" / "recipe-projection"

PARAMETER_FILES = {
    "input_projection": "recipe_encoder_input",
    "output_projection": "recipe_encoder_output",
    "requirement_projection": "recipe_encoder_requirement",
    "candidate_projection": "recipe_encoder_candidate",
}


def _write_array(path: Path, value, dtype) -> None:
    array = np.ascontiguousarray(np.asarray(value), dtype=dtype)
    path.write_bytes(array.tobytes())


def write_recipe_encoder_parameters(destination: Path, params=None) -> dict:
    """Write the exact float32 parameters selected by the environment contract."""

    import jax

    from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
        RECIPE_CANDIDATE_EMBEDDING_SIZE,
        RECIPE_CANDIDATE_INPUT_FEATURE_SIZE,
        RECIPE_CANDIDATE_OUTPUT_FEATURE_SIZE,
        RECIPE_CANDIDATE_REQUIREMENT_FEATURE_SIZE,
        RECIPE_CANDIDATE_SCALAR_SIZE,
        recipe_candidate_encoding_contract_sha256,
        recipe_candidate_encoding_parameter_seed,
    )
    from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
        initialize_recipe_candidate_encoder,
    )
    from hytalegym.jax.world import ACTOR_RECIPE_CANDIDATE_CAPACITY

    contract_sha256 = recipe_candidate_encoding_contract_sha256()
    parameter_seed = recipe_candidate_encoding_parameter_seed()
    selected = (
        initialize_recipe_candidate_encoder(jax.random.key(parameter_seed))
        if params is None
        else params
    )
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for field, prefix in PARAMETER_FILES.items():
        dense = getattr(selected, field)
        for component in ("kernel", "bias"):
            name = f"{prefix}_{component}.bin"
            _write_array(
                destination / name, getattr(dense, component), np.float32
            )
            files[name] = hashlib.sha256(
                (destination / name).read_bytes()
            ).hexdigest().upper()

    candidate_feature_size = (
        3 * RECIPE_CANDIDATE_EMBEDDING_SIZE
        + RECIPE_CANDIDATE_SCALAR_SIZE
    )
    manifest = {
        "schema": "hytalerl_recipe_candidate_encoder_profile_v1",
        "encoding_contract_sha256": contract_sha256,
        "parameter_source": "contract_seed" if params is None else "supplied",
        "parameter_seed_hex": f"{parameter_seed:08X}",
        "candidate_capacity": ACTOR_RECIPE_CANDIDATE_CAPACITY,
        "embedding_size": RECIPE_CANDIDATE_EMBEDDING_SIZE,
        "input_feature_size": RECIPE_CANDIDATE_INPUT_FEATURE_SIZE,
        "output_feature_size": RECIPE_CANDIDATE_OUTPUT_FEATURE_SIZE,
        "requirement_feature_size": RECIPE_CANDIDATE_REQUIREMENT_FEATURE_SIZE,
        "candidate_feature_size": candidate_feature_size,
        "files": files,
    }
    ordered = (
        "schema",
        "encoding_contract_sha256",
        "parameter_source",
        "parameter_seed_hex",
        "candidate_capacity",
        "embedding_size",
        "input_feature_size",
        "output_feature_size",
        "requirement_feature_size",
        "candidate_feature_size",
    )
    (destination / "recipe_encoder_contract.tsv").write_text(
        "".join(f"{name}\t{manifest[name]}\n" for name in ordered),
        encoding="utf-8",
    )
    return manifest


def write_recipe_policy_view(destination: Path, view, *, lane: int) -> None:
    """Write one actor-visible structured row with no execution identity."""

    _write_array(
        destination / "raw_recipe_available.bin",
        np.asarray(view.available)[lane : lane + 1],
        np.uint8,
    )
    _write_array(
        destination / "raw_recipe_candidate_mask.bin",
        np.asarray(view.candidate_mask)[lane],
        np.uint8,
    )
    boolean_fields = {
        "input_mask",
        "input_metadata_required",
        "output_mask",
        "requirement_mask",
        "knowledge_required",
    }
    float_fields = {"time_seconds"}
    for name, value in zip(view.policy._fields, view.policy, strict=True):
        dtype = (
            np.uint8
            if name in boolean_fields
            else np.float32
            if name in float_fields
            else np.int32
        )
        _write_array(
            destination / f"raw_recipe_{name}.bin",
            np.asarray(value)[lane],
            dtype,
        )


def _i32(value: int) -> int:
    return np.asarray(value, dtype=np.uint32).view(np.int32).item()


def _synthetic_view():
    import jax.numpy as jnp

    from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
        RecipeCandidatePolicyView,
    )
    from hytalegym.jax.crafting.contract import (
        IDENTITY_HASH_WORDS,
        MAX_BENCH_REQUIREMENTS,
        MAX_INGREDIENTS,
        MAX_OUTPUTS,
        METADATA_HASH_WORDS,
    )
    from hytalegym.jax.world import (
        ACTOR_RECIPE_CANDIDATE_CAPACITY,
        ActorRecipeCandidatePolicy,
    )

    batch = 3
    candidates = ACTOR_RECIPE_CANDIDATE_CAPACITY
    candidate_shape = (batch, candidates)
    input_shape = candidate_shape + (MAX_INGREDIENTS,)
    output_shape = candidate_shape + (MAX_OUTPUTS,)
    requirement_shape = candidate_shape + (MAX_BENCH_REQUIREMENTS,)

    available = np.asarray([False, True, True], dtype=np.bool_)
    candidate_mask = np.zeros(candidate_shape, dtype=np.bool_)
    candidate_mask[0, 0] = True  # must still encode to exact zero: unavailable row
    candidate_mask[1, [0, 3]] = True
    candidate_mask[2, [1, 2, 15]] = True

    values = {
        "input_mask": np.zeros(input_shape, dtype=np.bool_),
        "input_item_id": np.zeros(input_shape, dtype=np.int32),
        "input_resource_type_id": np.zeros(input_shape, dtype=np.int32),
        "input_quantity": np.zeros(input_shape, dtype=np.int32),
        "input_metadata_required": np.zeros(input_shape, dtype=np.bool_),
        "input_metadata_hash": np.zeros(
            input_shape + (METADATA_HASH_WORDS,), dtype=np.int32
        ),
        "output_mask": np.zeros(output_shape, dtype=np.bool_),
        "output_item_id": np.zeros(output_shape, dtype=np.int32),
        "output_quantity": np.zeros(output_shape, dtype=np.int32),
        "output_metadata_hash": np.zeros(
            output_shape + (METADATA_HASH_WORDS,), dtype=np.int32
        ),
        "requirement_mask": np.zeros(requirement_shape, dtype=np.bool_),
        "requirement_bench_type": np.zeros(requirement_shape, dtype=np.int32),
        "requirement_bench_id_hash": np.zeros(
            requirement_shape + (IDENTITY_HASH_WORDS,), dtype=np.int32
        ),
        "requirement_tier_level": np.zeros(
            requirement_shape, dtype=np.int32
        ),
        "knowledge_required": np.zeros(candidate_shape, dtype=np.bool_),
        "required_memories_level": np.zeros(candidate_shape, dtype=np.int32),
        "time_seconds": np.zeros(candidate_shape, dtype=np.float32),
    }

    def ingredient(lane, candidate, slot, item, resource, quantity, *, meta=()):
        values["input_mask"][lane, candidate, slot] = True
        values["input_item_id"][lane, candidate, slot] = item
        values["input_resource_type_id"][lane, candidate, slot] = resource
        values["input_quantity"][lane, candidate, slot] = quantity
        if meta:
            values["input_metadata_required"][lane, candidate, slot] = True
            values["input_metadata_hash"][lane, candidate, slot] = meta

    def output(lane, candidate, slot, item, quantity, *, meta=()):
        values["output_mask"][lane, candidate, slot] = True
        values["output_item_id"][lane, candidate, slot] = item
        values["output_quantity"][lane, candidate, slot] = quantity
        if meta:
            values["output_metadata_hash"][lane, candidate, slot] = meta

    def requirement(lane, candidate, slot, bench, tier, identity):
        values["requirement_mask"][lane, candidate, slot] = True
        values["requirement_bench_type"][lane, candidate, slot] = bench
        values["requirement_tier_level"][lane, candidate, slot] = tier
        values["requirement_bench_id_hash"][lane, candidate, slot] = identity

    ingredient(1, 0, 0, 101, -1, 2, meta=(7, _i32(0x89ABCDEF)))
    ingredient(1, 0, 1, -1, 5, 64)
    output(1, 0, 0, 301, 4, meta=(_i32(0xFFFFFFFF), 11))
    requirement(1, 0, 0, 2, 3, tuple(range(11, 19)))
    values["knowledge_required"][1, 0] = True
    values["required_memories_level"][1, 0] = 9
    values["time_seconds"][1, 0] = 0.25

    # Candidate three deliberately has no ingredients or bench requirements:
    # its masked means must use a divisor of one and remain exact zero.
    output(1, 3, 2, _i32(0x80000001), 1)
    values["time_seconds"][1, 3] = 8.5

    ingredient(2, 1, 4, _i32(0xF1234567), -1, 1)
    ingredient(2, 1, 27, 4096, 6, 2, meta=(-1, 0))
    output(2, 1, 0, 512, 128)
    requirement(
        2, 1, 2, 0, 12,
        tuple(_i32(0x80000000 + index * 17) for index in range(8)),
    )
    values["required_memories_level"][2, 1] = 1
    values["time_seconds"][2, 1] = 1.0 / 30.0

    ingredient(2, 2, 0, 0, 0, 2_147_483_647)
    output(2, 2, 3, 2_147_483_647, 2_147_483_647)
    values["knowledge_required"][2, 2] = True
    values["time_seconds"][2, 2] = 120.0

    # Candidate fifteen exercises an otherwise empty, legal candidate.
    values["required_memories_level"][2, 15] = -4
    values["time_seconds"][2, 15] = -2.0

    policy = ActorRecipeCandidatePolicy(
        *(jnp.asarray(values[name]) for name in ActorRecipeCandidatePolicy._fields)
    )
    return RecipeCandidatePolicyView(
        available=jnp.asarray(available),
        candidate_mask=jnp.asarray(candidate_mask),
        policy=policy,
    )


def _profile_roots() -> list[Path]:
    roots = [ROOT / "case" / "profile"]
    for parent in (ROOT / "case" / "profiles", ROOT / "case" / "profiles-region"):
        if parent.exists():
            roots.extend(sorted(path for path in parent.iterdir() if path.is_dir()))
    return roots


def _refresh_profile_content_sha256(root: Path) -> str:
    manifest_path = root / "profile.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name == "profile.json":
            continue
        identity.update(path.name.encode("utf-8"))
        identity.update(b"\0")
        identity.update(path.read_bytes())
        identity.update(b"\0")
    content_sha256 = identity.hexdigest().upper()
    manifest["content_sha256"] = content_sha256
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return content_sha256


def backfill_profile_parameters() -> int:
    """Atomically add the current encoder parameters to every existing profile."""

    case_root = (ROOT / "case").resolve()
    profiles = _profile_roots()
    if not profiles:
        raise SystemExit("no JVM perception profiles found")
    identities: dict[str, str] = {}
    for profile in profiles:
        resolved = profile.resolve()
        if case_root not in resolved.parents or not (profile / "profile.json").is_file():
            raise SystemExit(f"refusing non-profile path {profile}")
        scratch = Path(tempfile.mkdtemp(prefix=".recipe-profile-", dir=profile.parent))
        staged = scratch / profile.name
        backup = scratch / (profile.name + ".previous")
        try:
            shutil.copytree(profile, staged)
            write_recipe_encoder_parameters(staged)
            identities[str(profile.relative_to(ROOT))] = (
                _refresh_profile_content_sha256(staged)
            )
            os.replace(profile, backup)
            try:
                os.replace(staged, profile)
            except BaseException:
                os.replace(backup, profile)
                raise
            shutil.rmtree(backup)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps(identities, indent=2))
    return 0


def main() -> int:
    from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
        recipe_candidate_encoding_contract_sha256,
    )
    from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
        encode_recipe_candidates,
        initialize_recipe_candidate_encoder,
    )
    from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
        recipe_candidate_encoding_parameter_seed,
    )
    import jax

    if sys.argv[1:] == ["--backfill-profiles"]:
        return backfill_profile_parameters()
    if sys.argv[1:]:
        raise SystemExit(
            "usage: python -m tools.export.recipe_candidates "
            "[--backfill-profiles]"
        )

    destination = DEFAULT_CASE.resolve()
    case_root = (ROOT / "case").resolve()
    if case_root not in destination.parents:
        raise SystemExit(f"refusing to replace output outside {case_root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".recipe-projection-", dir=case_root))
    try:
        seed = recipe_candidate_encoding_parameter_seed()
        params = initialize_recipe_candidate_encoder(jax.random.key(seed))
        parameter_manifest = write_recipe_encoder_parameters(temporary)
        view = _synthetic_view()
        encoding = encode_recipe_candidates(params, view)
        for lane in range(view.available.shape[0]):
            lane_root = temporary / f"lane-{lane}"
            lane_root.mkdir()
            write_recipe_policy_view(lane_root, view, lane=lane)
            _write_array(
                lane_root / "expected_recipe_available.bin",
                np.asarray(encoding.available)[lane : lane + 1],
                np.uint8,
            )
            _write_array(
                lane_root / "expected_recipe_mask.bin",
                np.asarray(encoding.candidate_mask)[lane],
                np.uint8,
            )
            _write_array(
                lane_root / "expected_recipe_embedding.bin",
                np.asarray(encoding.candidate_embedding)[lane],
                np.float32,
            )
        (temporary / "case.json").write_text(
            json.dumps(
                {
                    "schema": "hytalerl_jvm_recipe_projection_case_v1",
                    "encoding_contract_sha256": (
                        recipe_candidate_encoding_contract_sha256()
                    ),
                    "lanes": int(view.available.shape[0]),
                    "parameter_files": parameter_manifest["files"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        temporary = None
        print(f"wrote non-vacuous recipe projection fixture to {destination}")
        return 0
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
