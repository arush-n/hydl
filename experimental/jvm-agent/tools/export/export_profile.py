"""Export one actual-JAX reset as the JVM perception profile.

Unlike ``export_parity.py augment-structured``, this never inverts a finished
learner row into a merely policy-equivalent raw state.  The profile is consumed
in production, so its resource values, authored ability table, and normalisers
must all come from the same real reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


# This exporter moved from the project root into ``tools/export``. Keep every
# destructive replacement and profile lookup anchored to the JVM-agent root,
# not to a shadow ``tools/export/case`` tree.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # Direct execution sets sys.path[0] to tools/export, which makes this
    # script's own `tools.export.*` helpers unimportable. Keep the documented
    # `python tools/export/export_profile.py` entry point self-contained.
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "case" / "profile"


def _live_widths() -> tuple[int, int]:
    """Observation and action widths the Gym publishes right now.

    These were the literals 8271 and 99. Both went stale -- the observation
    gained `locomotion_stamina_fraction` and the action surface grew to 124 --
    and the drift check then rejected a correct current profile while calling it
    "production profile contract drift", which reads like the export is broken
    rather than the constant.
    """

    from hytalegym.jax.combat.observation.v3 import (
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        arsenal_policy_observation_size,
    )

    return (
        int(arsenal_policy_observation_size()),
        int(sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)),
    )
PROFILE = "iron_sword"
OPPONENT = "iron_mace"
SEED = 0x50455243

# The parity fixture pairs each weapon with the next one in this rotation
# (manifest_flat.json "profiles" vs "opponents"). A profile only composes
# legitimately against states of its own pairing: point it at another weapon's
# state and it fails on authored metadata, which is an artifact rather than a
# defect. Kept here so a per-weapon export cannot silently mispair.
PAIRING = (
    ("iron_sword", "iron_mace"),
    ("iron_mace", "iron_daggers"),
    ("iron_daggers", "iron_spear"),
    ("iron_spear", "iron_battleaxe"),
    ("iron_battleaxe", "iron_shield"),
    ("iron_shield", "iron_shortbow"),
    ("iron_shortbow", "iron_crossbow"),
    ("iron_crossbow", "iron_staff"),
    ("iron_staff", "fire_spellbook"),
    ("fire_spellbook", "bombs"),
    ("bombs", "potions"),
    ("potions", "iron_sword"),
)

# The region fixture's opponents are UNARMED, and its manifest says otherwise.
#
# `manifest_region.json` lists "opponents": [iron_shield, iron_shortbow, ...],
# which is export_parity.py's rotation variable -- computed for both modes but
# never passed to `region_scene`. `region_scene` calls
# `hytale_0_5_7_loadouts(tuple(weapons))` with no `target_profiles=`, and that
# default is `[""] * len(agents)`: a target with no authored abilities at all.
#
# Believing the manifest costs two failures in a row. Emitting target ability
# bindings from the named opponent is rejected outright ("native binding points
# at an unauthored learner slot: target/0"), and before that, exporting a flat
# world with the manifest's pairing produced a raw_ability row differing from
# its own state at 289 of 868 indices.
REGION_OPPONENT = ""
REGION_WEAPONS = ("iron_sword", "iron_shield", "iron_shortbow", "iron_staff")


def _require_cuda_backend(jax_module) -> None:
    """Refuse to publish a production profile from a non-CUDA JAX runtime."""

    backend = jax_module.default_backend()
    devices = tuple(jax_module.devices())
    if backend != "gpu" or not devices or any(
        device.platform != "gpu" for device in devices
    ):
        raise SystemExit(
            "production JVM profiles require WSL/CUDA; "
            f"observed backend={backend!r}, devices={devices!r}"
        )


def _write_profile_metadata(
    temporary, runtime, params, profile_name, opponent_name, seed,
    observation_size, action_size, *, motion_timing_profile, lane=0,
):
    """Write checkpoint-owned text tables plus their content identity.

    Shared by the flat and region exports so the two cannot drift. The content
    hash covers every file in the directory, so it must be computed last.
    """

    import jax

    from hytalegym.jax.combat.arsenal.profiles import (
        hytale_0_5_7_native_profile_bindings,
        hytale_0_5_7_loadouts,
        native_ability_requested_charge_time_seconds,
    )
    from hytalegym.jax.combat.observation.v3.policy import (
        arsenal_policy_contract_sha256,
    )
    from hytalegym.jax.combat.observation.v3.native.codec.decode import (
        _status_programs,
    )
    from hytalegym.rulesets import load_native_status_projections
    from tools.export.recipe_candidates import write_recipe_encoder_parameters
    from tools.export.dodge_motion import write_dodge_motion_parameters

    programs = _status_programs(runtime.loadout)
    program_lines = [
        "# semantic_id\tdamage\thealing\tcooldown\tresource_delta\t"
        "speed_multiplier\tresource_id\tvalue_percent\tinfinite"
    ]
    for semantic_id, program in sorted(programs.items()):
        program_lines.append(
            "\t".join((
                str(int(semantic_id)),
                repr(float(program.damage)),
                repr(float(program.healing)),
                repr(float(program.cooldown)),
                repr(float(program.resource_delta)),
                repr(float(program.speed_multiplier)),
                str(int(program.resource_id)),
                "1" if program.value_percent else "0",
                "1" if program.infinite else "0",
            ))
        )
    (temporary / "status_programs.tsv").write_text(
        "\n".join(program_lines) + "\n", encoding="utf-8")

    omissions = sorted(
        effect_id
        for effect_id, projection in load_native_status_projections().items()
        if projection.projection
        in {"omit_visual_only", "omit_derived_mechanic"}
    )
    (temporary / "status_omissions.txt").write_text(
        "\n".join(omissions) + "\n", encoding="utf-8")

    binding_lines = [
        "# actor\tslot\titem_id\tinteraction_id\tinteraction_type"
        "\trequested_charge_seconds"
    ]
    guard_lines = ["# actor\titem_id\tinteraction_id\tinteraction_type"]
    # ``arsenal_runtime_config`` compacts events into runtime tables, so its
    # loadout no longer carries the authored (B,N,A,E) event clock required by
    # the native charge selector. Rebuild the exact named source loadout here;
    # profile/opponent identities are the same inputs used to make ``runtime``.
    charge_loadout = hytale_0_5_7_loadouts(
        (profile_name,), target_profiles=(opponent_name,))
    for actor, entity_index, binding_profile in (
        ("agent", 0, profile_name),
        ("target", 1, opponent_name),
    ):
        # An empty profile is a real loadout, not a missing one: the region
        # scene arms only the agent. Emitting bindings for an unarmed target
        # names learner slots it never authored, and PerceptionProfile rejects
        # the profile at load rather than at first use.
        if not binding_profile:
            continue
        native = hytale_0_5_7_native_profile_bindings(binding_profile)
        if len(native.abilities) != len(native.authored_ability_slots):
            raise SystemExit(
                f"{actor} native ability bindings lost slot identity")
        for slot, binding in zip(
            native.authored_ability_slots, native.abilities, strict=True
        ):
            binding_lines.append(
                "\t".join((
                    actor,
                    str(int(slot)),
                    native.item_id,
                    binding.interaction_id,
                    binding.interaction_type,
                    repr(native_ability_requested_charge_time_seconds(
                        charge_loadout,
                        int(slot),
                        source_entity_index=entity_index,
                    )),
                ))
            )
        if native.guard is not None:
            guard_lines.append(
                "\t".join((
                    actor,
                    native.item_id,
                    native.guard.interaction_id,
                    native.guard.interaction_type,
                ))
            )
    (temporary / "ability_bindings.tsv").write_text(
        "\n".join(binding_lines) + "\n", encoding="utf-8")
    (temporary / "guard_bindings.tsv").write_text(
        "\n".join(guard_lines) + "\n", encoding="utf-8")

    # The encoder uses fixed random projections selected by the observation
    # contract. Export their exact float32 values; Java must never attempt to
    # reproduce JAX's PRNG from only the seed.
    write_recipe_encoder_parameters(temporary)
    write_dodge_motion_parameters(
        temporary,
        runtime,
        params,
        lane=lane,
        motion_timing_profile=motion_timing_profile,
    )

    identity = hashlib.sha256()
    for path in sorted(temporary.iterdir(), key=lambda value: value.name):
        if not path.is_file():
            continue
        identity.update(path.name.encode("utf-8"))
        identity.update(b"\0")
        identity.update(path.read_bytes())
        identity.update(b"\0")
    manifest = {
        "schema": "hytalerl_jvm_perception_profile_v1",
        "profile": profile_name,
        "opponent": opponent_name,
        "seed": seed,
        "observation_size": int(observation_size),
        "action_size": int(action_size),
        "arsenal_policy_contract_sha256": arsenal_policy_contract_sha256(),
        "jax_backend": jax.default_backend(),
        "content_sha256": identity.hexdigest().upper(),
    }
    (temporary / "profile.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _export_region() -> int:
    """Capture one region profile per lane from a single region reset.

    Mirrors ``export_parity.py region`` exactly -- same weapons, same held item,
    same seed, same reset -- because the profile's authored ability table has to
    be the one its states were captured against. Anything else silently exports
    a flat-world table that cannot compose.
    """

    import jax
    import jax.numpy as jnp

    from hytalegym.jax.combat import default_combat_params
    from hytalegym.jax.combat.observation import COMBAT_FLOAT_FEATURES
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        AGENT_ENTITY,
        align_actor_light_policy_tokens,
        arsenal_policy_action_mask,
        mask_inventory_policy_tokens,
    )
    from hytalegym.jax.combat.observation.v3.tokens.inventory import (
        inventory_policy_tokens_from_state,
    )
    from hytalegym.jax.combat.arsenal.programs.ability import (
        ability_lifecycle_legality_view,
    )
    from hytalegym.jax.combat.types import TARGET_ENTITY
    try:
        from hytalegym.jax.combat.runtime.reset import target_evidence
        from hytalegym.jax.combat.opponents.legacy.controller import _target_phase
    except ModuleNotFoundError:
        from hytalegym.jax.combat._reset import target_evidence
        from hytalegym.jax.combat._target import _target_phase
    from worlds.region import region_scene
    from export_parity import REGION_PROFILES, SEED as PARITY_SEED, capture

    if tuple(REGION_PROFILES) != REGION_WEAPONS:
        raise SystemExit(
            f'region weapons drifted: export_parity has {tuple(REGION_PROFILES)}, '
            f'this exporter expects {REGION_WEAPONS}')

    weapons = tuple(REGION_PROFILES)
    params = default_combat_params(
        microticks=1,
        target_active=True,
    )._replace(
        # The standalone public evidence path does not own the JAX episode's
        # reset-local tick. Profiles 1/2 alternate loaded_dt by that phase and
        # cannot be reproduced from an unrelated absolute world tick. Native-
        # uploadable profiles therefore pin the deterministic profile 0.
        motion_timing_profile_min=jnp.int32(0),
        motion_timing_profile_max=jnp.int32(0),
    )
    scene = region_scene(
        weapons=weapons, held_item="Tool_Pickaxe_Iron", native_evidence=False)
    state, observation, _ = scene.environment.reset(
        jnp.stack([jax.random.key(PARITY_SEED)] * len(weapons)))

    case_root = (ROOT / "case").resolve()
    case_root.mkdir(parents=True, exist_ok=True)
    written = []
    for lane, weapon in enumerate(weapons):
        out = ROOT / "case" / "profiles-region" / weapon
        output = out.resolve()
        if case_root not in output.parents or output == case_root:
            raise SystemExit(f"refusing to write outside {case_root}: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".profile-", dir=case_root))
        try:
            capture(
                temporary, 0, lane, weapon, state, observation, params,
                scene.runtime, AGENT_ENTITY, TARGET_ENTITY,
                COMBAT_FLOAT_FEATURES, target_evidence, _target_phase,
                align_actor_light_policy_tokens, mask_inventory_policy_tokens,
                inventory_policy_tokens_from_state, arsenal_policy_action_mask,
                ability_lifecycle_legality_view,
            )
            _write_profile_metadata(
                temporary, scene.runtime, params, weapon,
                REGION_OPPONENT, PARITY_SEED,
                observation.learner_observation.shape[-1]
                if hasattr(observation, 'learner_observation')
                else _live_widths()[0],
                _live_widths()[1],
                motion_timing_profile=int(
                    np.asarray(
                        state.runtime.combat.motion_timing_profile
                    )[lane]
                ),
                lane=lane,
            )
            if output.exists():
                shutil.rmtree(output)
            os.replace(temporary, output)
            temporary = None
            written.append(weapon)
        finally:
            if temporary is not None and Path(temporary).exists():
                shutil.rmtree(temporary, ignore_errors=True)

    print(f"wrote {len(written)} region profiles: {', '.join(written)}")
    return 0


def main(
    profile_argument: str | None = None,
    opponent_argument: str | None = None,
    output_argument: Path | None = None,
    *,
    region: bool = False,
) -> int:
    # Imports stay inside main so --help/source inspection does not initialize
    # JAX or allocate a device. Run from a neutral cwd with the live Gym first
    # on PYTHONPATH; see INSTRUCTIONS.md.
    import jax
    import jax.numpy as jnp

    from hytalegym.jax.combat import (
        arsenal_runtime_config,
        default_combat_params,
        hytale_0_5_7_loadouts,
    )
    try:
        from hytalegym.jax.combat.runtime.reset import target_evidence
        from hytalegym.jax.combat.opponents.legacy.controller import _target_phase
    except ModuleNotFoundError:
        from hytalegym.jax.combat._reset import target_evidence
        from hytalegym.jax.combat._target import _target_phase
    from hytalegym.jax.combat.observation import COMBAT_FLOAT_FEATURES
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        AGENT_ENTITY,
        align_actor_light_policy_tokens,
        arsenal_policy_action_mask,
        mask_inventory_policy_tokens,
    )
    from hytalegym.jax.combat.observation.v3.tokens.inventory import (
        inventory_policy_tokens_from_state,
    )
    from hytalegym.jax.combat.arsenal.programs.ability import (
        ability_lifecycle_legality_view,
    )
    from hytalegym.jax.combat.types import TARGET_ENTITY
    from hytalegym.jax.training import (
        make_arsenal_ppo_environment,
        open_flat_arsenal_world_capabilities,
    )
    from adk.validation import reset_keys
    from export_parity import capture

    # A CPU-generated profile can be numerically close while violating the
    # execution backend required by the deployment evidence. Fail before any
    # temporary directory or canonical fixture is created.
    _require_cuda_backend(jax)

    # No argument reproduces the pinned production profile exactly as before.
    # A weapon name exports that weapon's profile beside it, so ProfileTest can
    # certify the production path on every weapon rather than only iron_sword.
    profile_name = PROFILE
    opponent_name = OPPONENT
    out = OUT
    if region:
        # A region profile cannot be built the way a flat one is. The region
        # parity states come from `region_scene(..., held_item=
        # "Tool_Pickaxe_Iron")`, which changes the authored ability table
        # outright: exporting the region *pairing* from a flat world produced a
        # raw_ability row differing from its own state at 289 of 868 indices,
        # while the flat pair differs at 0. So this branch reproduces the same
        # scene and captures all four lanes from one reset -- both because it
        # must match, and because the region compile is far too expensive to pay
        # four times.
        return _export_region()
    if profile_argument is not None:
        profile_name = profile_argument
        pairs = dict(PAIRING)
        if profile_name not in pairs:
            raise SystemExit(
                f"unknown profile {profile_name!r}; that rotation pairs only: "
                f"{', '.join(name for name, _ in PAIRING)}")
        opponent_name = pairs[profile_name]
        out = ROOT / "case" / "profiles" / profile_name
    if opponent_argument is not None:
        available = {name for name, _ in PAIRING}
        if opponent_argument not in available:
            raise SystemExit(
                f"unknown opponent {opponent_argument!r}; available profiles: "
                f"{', '.join(sorted(available))}"
            )
        opponent_name = opponent_argument
    if output_argument is not None:
        out = output_argument
        if not out.is_absolute():
            out = ROOT / out

    params = default_combat_params(
        microticks=1,
        target_active=True,
    )._replace(
        # Native upload has no public reset-local JAX tick. Profile 0 is the
        # only timing catalogue entry independent of that unavailable phase.
        motion_timing_profile_min=jnp.int32(0),
        motion_timing_profile_max=jnp.int32(0),
    )
    runtime = arsenal_runtime_config(
        hytale_0_5_7_loadouts((profile_name,), target_profiles=(opponent_name,))
    )
    environment = make_arsenal_ppo_environment(
        params,
        runtime,
        world_capability_provider=open_flat_arsenal_world_capabilities,
    )
    state, observation, _ = environment.reset(reset_keys(SEED, 1))

    case_root = (ROOT / "case").resolve()
    case_root.mkdir(parents=True, exist_ok=True)
    output = out.resolve()
    # This path is handed to shutil.rmtree, so the guard stays -- widened only
    # far enough to allow case/profiles/<weapon>, never case/ itself.
    if case_root not in output.parents or output == case_root:
        raise SystemExit(f"refusing to replace profile outside {case_root}: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".profile-", dir=case_root))
    try:
        capture(
            temporary,
            0,
            0,
            profile_name,
            state,
            observation,
            params,
            runtime,
            AGENT_ENTITY,
            TARGET_ENTITY,
            COMBAT_FLOAT_FEATURES,
            target_evidence,
            _target_phase,
            align_actor_light_policy_tokens,
            mask_inventory_policy_tokens,
            inventory_policy_tokens_from_state,
            arsenal_policy_action_mask,
            ability_lifecycle_legality_view,
        )

        raw_ability = np.fromfile(
            temporary / "raw_ability.bin", dtype=np.float32)
        resources = np.fromfile(
            temporary / "raw_resource_current.bin", dtype=np.float32)
        if raw_ability.shape != (868,) or resources.shape != (14,):
            raise SystemExit(
                "production profile raw widths drifted: "
                f"{raw_ability.shape}/{resources.shape}"
            )
        if not np.array_equal(raw_ability[:14], resources):
            raise SystemExit(
                "production profile is not one coherent state: ability and "
                "mechanics resource rows differ"
            )
        observation_row = np.fromfile(
            temporary / "expected_observation.bin", dtype=np.float32)
        action_mask = np.fromfile(
            temporary / "action_mask.bin", dtype=np.uint8)
        live_observation, live_actions = _live_widths()
        if observation_row.shape != (live_observation,) or (
            action_mask.shape != (live_actions,)
        ):
            raise SystemExit(
                "production profile contract drift: "
                f"{observation_row.shape}/{action_mask.shape} != "
                f"({live_observation},)/({live_actions},)"
            )

        manifest = _write_profile_metadata(
            temporary, runtime, params, profile_name, opponent_name, SEED,
            observation_row.size, action_mask.size,
            motion_timing_profile=int(
                np.asarray(state.runtime.combat.motion_timing_profile)[0]
            ),
        )

        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        temporary = None
        print(
            f"wrote {profile_name} vs {opponent_name} production profile to {output} "
            f"({manifest['content_sha256']})"
        )
        return 0
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate the production JVM perception profile fixture."
    )
    parser.add_argument(
        "profile",
        nargs="?",
        choices=tuple(profile for profile, _ in PAIRING),
        help="flat weapon profile (default: iron_sword)",
    )
    parser.add_argument(
        "--opponent",
        choices=tuple(profile for profile, _ in PAIRING),
        help=(
            "flat opponent profile; defaults to the production parity pairing "
            "for the selected profile"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output directory under case/; defaults to case/profile for the "
            "default pair or case/profiles/<profile> for an explicit profile"
        ),
    )
    parser.add_argument(
        "--region",
        action="store_true",
        help="regenerate the four Region production profiles",
    )
    parsed = parser.parse_args()
    if parsed.region and (
        parsed.profile is not None
        or parsed.opponent is not None
        or parsed.output is not None
    ):
        parser.error(
            "--region cannot be combined with a flat profile, opponent, or output"
        )
    sys.exit(main(
        parsed.profile,
        parsed.opponent,
        parsed.output,
        region=parsed.region,
    ))
