"""Bind the Gym's exact Region runtime so the world-action heads are reachable.

**Measured against the retired 99-bit surface: 32 action bits were dead on
every scene** -- 16 ``block_none_plus_candidates`` and 16
``recipe_none_plus_candidates``. Crafting has since left the policy surface
entirely, so only the block half of that finding still has a head; re-measure
before quoting a bit count. That is
not a defect. Those heads are gated by a legality mask that closes when there
is nothing to place, break or craft *against*, and the flat combat fixture has
no mutable world. Judging place/break/craft behaviour there would file a
behaviour gap against a surface the policy was never offered -- the exact
mistake the reachability rule exists to prevent.

This module builds the reachable case, so the question becomes answerable.

Two things it deliberately does **not** invent:

* **Reach and view angle** come from
  ``native_region_block_candidate_defaults()`` -- authored Adventure
  ``UseDistance`` 5.0 plus the bridge's 2.0 buffer, and a full 360 degree
  sector because ``NativeBlockUse.startExact`` applies no angular gate. The
  Gym's own fixture test uses ``maximum_distance=1.25`` to keep its candidate
  set tight; that is a test convenience, and copying it would measure a
  deliberately crippled reach.
* **The terrain**, which is a published, hash-pinned artifact library rather
  than a generated seed. ``library_semantic_sha256`` is read from the manifest
  and handed back to the loader, which recomputes the library digest and
  refuses a library whose content no longer matches its own manifest.

The binding itself is the Gym's: ``examples/jax_arsenal_world_benchmark.py``
already assembles geometry, navigation, world runtime, action surface and
executor. That file is a script rather than a package module, so it is loaded
by path -- see :func:`loader`.

## The world-action surface splits into two independently gated halves

Reading ``jax_arsenal_world_benchmark.py:287-317``:

* ``candidate_producer_bound`` is **unconditional**. Block candidates come
  from the Region runtime alone, so the 16 ``block_*`` slots are reachable
  with terrain and nothing else.
* ``executor_bound`` requires ``native_item_interaction_recording`` **and**
  ``modification_allowed``; the crafting catalog additionally requires the
  item-interaction recording. So ``use_off_on``,
  ``block_primary_secondary_trigger`` and the 16 ``recipe_*`` slots need
  native evidence recorded against the **currently deployed bridge**.

That split matters because the second half is currently stranded -- see
:func:`evidence_status`. Keeping them separate is what stops a stale artifact
from being reported as a missing capability.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The Gym checkout whose published artifacts this library reads. Derived here
#: rather than imported from ADK: this is the shared layer both ADK and Arena
#: sit on, so it must not depend on either of them.
GYM_ROOT = Path(__file__).resolve().parents[1] / "HytaleRL" / "hytalegym"

HYTALE_ROOT = GYM_ROOT.parent

#: The published Region corpus the Gym's own action-surface gate runs against.
#:
#: **256 train worlds, up from the pilot library's 16.** Every run through
#: 2026-08-17 bound `region-library-pilot-v2`, so `world_count: 16` was not a
#: capacity choice -- it was the entire library, and rotating `selection_key`
#: had nothing left to rotate between.
#:
#: v3-288 ships no block-semantic sidecars, and they cannot be captured: the
#: live server no longer regenerates the terrain those Regions were recorded
#: from. The first attempt to bind it anyway died at update 0
#: (`20260818T031307Z-036f5cfb`) because `multi_actor/region_rollout.py`
#: requires an action-surface runtime, which a fixture happily builds without.
#: The runtime is now constructed either way -- see
#: `region_block_semantic_atlas_without_evidence` -- with every semantic mask
#: false, so the mutable-physics half the rollout needs is present while the
#: block interaction heads close. Verify with the provider check in
#: `arena/tests/test_region_library_pin.py`, not by building a fixture.
LIBRARY = HYTALE_ROOT / "artifacts" / "worldgen" / "region-library-v3-288"
REGION_MANIFEST = LIBRARY / "manifest.json"
#: v3 keeps the traversal sidecar at the library root; the pilot libraries
#: nested it under `traversal-v2/`.
TRAVERSAL_MANIFEST = (
    LIBRARY / "traversal-manifest.json"
    if (LIBRARY / "traversal-manifest.json").is_file()
    else LIBRARY / "traversal-v2" / "traversal-manifest.json"
)
BLOCK_SEMANTIC_MANIFEST = LIBRARY / "block-semantics-v1" / "manifest.json"

#: One directory per bridge generation, named by the leading bytes of the jar
#: it was recorded against. Nothing here is authoritative on its own; the
#: deployed jar decides which generation is usable.
EVIDENCE_HISTORY = HYTALE_ROOT / "artifacts" / "worldgen" / "world-evidence-history"
ITEM_INTERACTION_FIXTURE = "fixtures/native_item_interaction_recording.json"
CRAFTING_FIXTURE = "fixtures/native_crafting_catalog_recording.json"

FIXTURE_BUILDER = HYTALE_ROOT / "examples" / "jax_arsenal_world_benchmark.py"

#: Terrain and code only. Native evidence is deliberately *not* required --
#: its absence closes some heads without making the scene unbuildable.
#:
#: Block semantics are on the same footing: they gate the block-interaction
#: action surface, not the terrain. Their absence closes the place/break heads
#: and leaves the scene, the rollout and every combat head intact.
REQUIRED = (
    REGION_MANIFEST,
    TRAVERSAL_MANIFEST,
    FIXTURE_BUILDER,
)


def block_semantics_available() -> bool:
    """Whether this library can publish a block-interaction action surface.

    False means terrain and traversal are complete but place/break is closed --
    a legitimate movement-only library, not a broken one.
    """

    return BLOCK_SEMANTIC_MANIFEST.is_file()

#: Seeds the Gym's action-surface gate uses, kept so a result here is
#: comparable with theirs rather than being a fresh, unanchored sample.
SELECTION_KEY = 1_197_923_983
ASSIGNMENT_KEY = 1_197_923_984
NODE_SEED = 570_057

#: World's candidate capacity; each of the two candidate heads is
#: ``none`` + this many slots, so 16 -> 17 logits.
CANDIDATE_CAPACITY = 16


def missing() -> tuple[Path, ...]:
    """Which required artifacts are absent.

    A Region scene depends on published data, not only on code, so an empty
    result here is a precondition for any world-action claim: without it a
    dead ``block_*`` bit means "no terrain was loaded", not "the head is
    broken".
    """

    return tuple(path for path in REQUIRED if not path.exists())


def available() -> bool:
    return not missing()


def deployed_bridge() -> str:
    """The jar the Gym currently considers deployed."""

    from hytalegym.worldgen.region.stability import (
        current_native_evidence_jar_sha256,
    )

    return current_native_evidence_jar_sha256().upper()


def _generation_bridge(directory: Path) -> str | None:
    for name in ("recertification-manifest.json", ITEM_INTERACTION_FIXTURE):
        path = directory / name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bridge = value.get("bridge_sha256")
        if isinstance(bridge, str) and bridge:
            return bridge.upper()
    return None


def evidence_generation() -> Path | None:
    """The evidence directory recorded against the **deployed** bridge.

    Never trust a generation's own directory name or README for this -- a
    later jar deployment strands every earlier recording without editing any
    of them. The deployed jar is the only authority.
    """

    if not EVIDENCE_HISTORY.is_dir():
        return None
    current = deployed_bridge()
    for directory in sorted(EVIDENCE_HISTORY.iterdir()):
        if not directory.is_dir():
            continue
        if _generation_bridge(directory) == current:
            item = directory / ITEM_INTERACTION_FIXTURE
            if item.exists():
                return directory
    return None


def evidence_status() -> dict[str, Any]:
    """Whether native item/crafting evidence exists for the deployed bridge.

    A closed ``use_off_on`` or ``recipe_*`` head means nothing until this is
    read: without a current recording those heads are *correctly* closed, and
    calling that a missing capability would be a class-4 result filed as a
    class-2 one.
    """

    current = deployed_bridge()
    generations = {}
    if EVIDENCE_HISTORY.is_dir():
        for directory in sorted(EVIDENCE_HISTORY.iterdir()):
            if directory.is_dir():
                generations[directory.name] = _generation_bridge(directory)
    match = evidence_generation()
    return {
        "deployed_bridge": current,
        "generation": None if match is None else match.name,
        "generations": generations,
        "stranded": match is None and bool(generations),
    }


def loader():
    """Import the Gym's Region fixture builder, which lives outside the package.

    ``examples/jax_arsenal_world_benchmark.py`` is a script, so there is no
    import path to it; the Gym's own gate loads it the same way. Loading by
    path means a moved or renamed file fails here rather than surfacing as an
    unexplained missing capability.
    """

    if not FIXTURE_BUILDER.exists():
        raise FileNotFoundError(
            f"the Gym's Region fixture builder is missing at {FIXTURE_BUILDER}; "
            "without it there is no published binding from a Region artifact "
            "library to the Arsenal world seams"
        )
    name = "adk_region_fixture_builder"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, FIXTURE_BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a module from {FIXTURE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authored_reach() -> tuple[float, float]:
    """``(maximum_distance, view_sector_full_angle_degrees)`` from native.

    Derived, never chosen: ``native_region_block_candidate_defaults()`` adds
    the bridge's ``NATIVE_BLOCK_INTERACTION_DISTANCE_BUFFER`` to the authored
    Adventure ``UseDistance``. The sector is ``math.tau`` radians because the
    native typed-target path applies no angular gate at all.
    """

    from hytalegym.jax.training.world_actions.block_candidates import (
        native_region_block_candidate_defaults,
    )

    defaults = native_region_block_candidate_defaults()
    degrees = math.degrees(defaults.view_sector_full_angle_radians)
    return float(defaults.maximum_distance), min(float(degrees), 360.0)


@dataclass(frozen=True, slots=True)
class LoadedRegion:
    """A Region fixture and its bindings, with no environment built yet.

    Split out so a consumer that constructs its own environment -- the console
    builds through `AgentKit` so that one collector path serves both the flat
    and Region worlds -- does not pay for an environment it discards.
    """

    fixture: Any
    runtime: Any
    maximum_distance: float
    view_sector_full_angle_degrees: float
    batch: int
    #: The evidence generation actually used, or None when the scene was built
    #: from terrain alone. A closed use/trigger/recipe head is expected here.
    evidence: str | None

    @property
    def params(self) -> Any:
        """The fixture's params -- spawn, target offset and floor are rewritten."""
        return self.fixture.params


@dataclass(frozen=True, slots=True)
class RegionScene:
    """A built Arsenal environment with the world-action surface bound."""

    environment: Any
    fixture: Any
    params: Any
    runtime: Any
    maximum_distance: float
    view_sector_full_angle_degrees: float
    batch: int
    #: The evidence generation actually used, or None when the scene was built
    #: from terrain alone. A closed use/trigger/recipe head is expected here.
    evidence: str | None

    @property
    def bound(self) -> dict[str, bool]:
        """Which optional world seams the loader actually published.

        Every one of these fails closed independently, so a dead world-action
        bit has to be read against this map before it is called a defect.
        """

        return {
            name: getattr(self.fixture, name) is not None
            for name in (
                "action_surface_provider",
                "action_surface_executor",
                "action_surface_runtime_initializer",
                "world_runtime_provider",
                "explosion_candidate_provider",
                "inventory_reset_provider",
            )
        }


#: ``ActorBlockActionCandidates.diagnostics`` bits, in bit order. A cleared
#: candidate row always names its own cause here, so a closed ``block_*`` head
#: never has to be explained by inference.
BLOCK_DIAGNOSTICS = (
    "invalid_actor",
    "query_unavailable",
    "los_unavailable",
    "affordance_unavailable",
    "output_capacity",
    "source_incomplete",
    "alias_conflict",
)


def decode_block_diagnostics(diagnostics) -> tuple[str, ...]:
    """Name every diagnostic bit set anywhere in a candidate row."""

    import numpy as np

    value = int(np.bitwise_or.reduce(np.asarray(diagnostics).ravel().astype(np.uint32)))
    return tuple(
        name for index, name in enumerate(BLOCK_DIAGNOSTICS) if value & (1 << index)
    )


def raw_block_candidates(scene: "RegionScene", state, selection: str = "native_camera"):
    """Re-run the block-candidate provider to recover its ``diagnostics``.

    ``state.action_surface.block_candidates`` is the **policy view**, and
    ``diagnostics`` is explicitly a non-policy field
    (``block_tokens.py:887``) -- the policy must not see why a row closed. A
    validator must.

    **Which producer this rebuilds matters, and this function used to get it
    wrong.** ``RegionActionSurfaceConfig.candidate_selection`` defaults to
    ``"native_camera"`` (``jax_arsenal_world_benchmark.py:171``), and
    :func:`region_scene` does not override it -- so the scene runs
    ``make_region_runtime_camera_block_candidate_provider``. This function
    previously always rebuilt ``make_region_runtime_block_candidate_provider``,
    the ``"dense_complete"`` path, and described it as "the same inputs the
    fixture used". It was not: every number taken through it described a
    producer the scene never runs, which is how a reach/capacity sweep came to
    "explain" a closure it could not have caused.

    ``selection`` now defaults to the scene's own default. Pass
    ``"dense_complete"`` deliberately if you want the stencil producer, and do
    not compare its output to the policy view -- they are different algorithms.
    """

    import math

    import jax.numpy as jnp

    from hytalegym.geometry.contract import FLAG_OPAQUE
    from hytalegym.jax.combat.arsenal.environment import (
        make_region_runtime_block_candidate_provider,
    )

    world = state.action_surface_runtime.world
    if selection == "native_camera":
        from hytalegym.jax.combat.arsenal.environment import (
            make_region_runtime_camera_block_candidate_provider,
        )

        provider = make_region_runtime_camera_block_candidate_provider(
            maximum_interaction_distance=scene.maximum_distance,
        )
    elif selection == "dense_complete":
        flags = world.geometry.atlas.cell_flags
        provider = make_region_runtime_block_candidate_provider(
            role_opaque_mask=((flags.astype(jnp.int32) & jnp.int32(FLAG_OPAQUE)) != 0),
            maximum_distance=scene.maximum_distance,
            view_sector_full_angle_radians=math.radians(
                scene.view_sector_full_angle_degrees
            ),
            cell_radius=math.ceil(scene.maximum_distance),
        )
    else:
        raise ValueError(
            f"selection must be native_camera or dense_complete, got {selection!r}"
        )
    candidates, _ = provider(state.runtime, world, scene.params)
    return candidates


def load_region(
    *,
    weapons: tuple[str, ...] = ("iron_sword",),
    target_weapons: tuple[str, ...] | None = None,
    policy_controlled_targets: bool = False,
    target_active: bool = True,
    held_item: str | None = "Tool_Pickaxe_Iron",
    native_evidence: bool | None = None,
    mutation_capacity: int = 16,
    selection_key: int = SELECTION_KEY,
    node_seed: int = NODE_SEED,
    maximum_distance: float | None = None,
    target_separation_range: tuple[float, float] | None = None,
    require_initial_line_of_sight: bool = True,
    working_set_capacity: int = 1,
    environment_diversity: bool = False,
    selection_seeds: tuple[int, ...] | None = None,
) -> LoadedRegion:
    """Load the Region fixture and everything bound to it, without an environment.

    ``target_weapons`` supplies the second entity's authored profile per batch
    row. ``policy_controlled_targets=True`` disables its scripted controller;
    multi-actor collectors can then seat a second policy on that entity.
    ``target_active=False`` also disables the legacy target controller and is
    required for two-live-policy duels.

    ``held_item`` selects which block trigger becomes available: a pickaxe
    exposes the primary (break) trigger, a placeable block the secondary
    (place) one. ``None`` leaves the actor empty-handed, which is the
    unarmed state-change-use case.

    ``node_seed`` picks the spawn node from the region's authored traversal
    graph (21,174 candidates, 4,477 of them navigation-legal per the w1 reset
    pool census). By default batch rows share it: the loader takes one seed, so
    a batch is byte-identical across rows -- 0 of 8271 observation columns
    differ at batch 4.

    ``working_set_capacity`` publishes that many library worlds and spreads the
    batch across them; ``environment_diversity=True`` then draws a separate
    spawn/target pair for every row inside that row's own world, still subject
    to ``target_separation_range`` and the line-of-sight requirement. Together
    they are what stops a policy from memorising one map and one opening
    distance. Both default to the historical single-world single-pair batch.

    ``target_separation_range`` optionally replaces the adjacent-edge target
    reset with a seeded navigation-reachable, actor-valid node in the declared
    horizontal distance band. It changes reset geometry, not runtime physics.
    ``require_initial_line_of_sight=False`` keeps the separated traversal pair
    but leaves visibility to the curriculum; this is useful for privileged
    pursuit fundamentals and is stamped in fixture metadata.

    ``native_evidence`` defaults to *auto*: the item-interaction and crafting
    recordings are attached only when a generation exists for the deployed
    bridge. Pass ``True`` to demand them -- useful when a result is only
    meaningful with the executor bound -- or ``False`` to build the pure
    terrain case deliberately.

    ``maximum_distance`` overrides the native-derived reach. **For control
    experiments only.** Any reach other than :func:`authored_reach` measures a
    world the server does not implement, so a number taken under an override
    is not a fidelity result -- it is evidence about the encoding's limits.
    """

    if policy_controlled_targets and target_weapons is None:
        raise ValueError("policy-controlled targets require target_weapons")

    from hytalegym.jax.combat import (
        arsenal_runtime_config,
        default_combat_params,
        hytale_0_5_7_loadouts,
    )

    absent = missing()
    if absent:
        raise FileNotFoundError(
            "the Region corpus is incomplete; a world-action measurement here "
            "would report unreachability as absence:\n"
            + "\n".join(f"  {path}" for path in absent)
        )

    generation = None if native_evidence is False else evidence_generation()
    if native_evidence is True and generation is None:
        status = evidence_status()
        raise FileNotFoundError(
            "no native world-action evidence was recorded against the deployed "
            f"bridge {status['deployed_bridge'][:16]}...; "
            f"{len(status['generations'])} older generations exist and every "
            "one is stranded. This is stale infrastructure (class 4), not a "
            "missing capability -- the executor code path is present and the "
            "recordings simply predate the current jar."
        )

    module = loader()
    batch = len(weapons)
    params = default_combat_params(microticks=1, target_active=target_active)
    loadouts = hytale_0_5_7_loadouts(tuple(weapons), target_profiles=target_weapons)
    runtime = arsenal_runtime_config(
        loadouts,
        opponent_controller_mask=(False, False) if policy_controlled_targets else None,
    )
    authored, view_degrees = authored_reach()
    reach = authored if maximum_distance is None else float(maximum_distance)

    item_recording = (
        None if generation is None else generation / ITEM_INTERACTION_FIXTURE
    )
    crafting_recording = None
    if generation is not None:
        candidate = generation / CRAFTING_FIXTURE
        crafting_recording = candidate if candidate.exists() else None

    fixture = module.load_arsenal_region_fixture(
        region_manifest=REGION_MANIFEST,
        traversal_manifest=TRAVERSAL_MANIFEST,
        expected_region_sha256=_identity(REGION_MANIFEST),
        expected_traversal_sha256=_identity(TRAVERSAL_MANIFEST),
        batch_size=batch,
        selection_key=selection_key,
        assignment_key=ASSIGNMENT_KEY,
        node_seed=node_seed,
        params=params,
        runtime=runtime,
        target_separation_range=target_separation_range,
        require_initial_line_of_sight=require_initial_line_of_sight,
        working_set_capacity=working_set_capacity,
        environment_diversity=environment_diversity,
        selection_seeds=selection_seeds,
        action_surface_config=module.RegionActionSurfaceConfig(
            block_semantic_manifest=(
                BLOCK_SEMANTIC_MANIFEST if block_semantics_available() else None
            ),
            mutation_capacity=mutation_capacity,
            maximum_distance=reach,
            view_sector_full_angle_degrees=view_degrees,
            native_item_interaction_recording=item_recording,
            native_crafting_catalog_recording=crafting_recording,
            modification_allowed=True,
            held_item_asset_id=held_item,
        ),
    )
    return LoadedRegion(
        fixture=fixture,
        runtime=runtime,
        maximum_distance=reach,
        view_sector_full_angle_degrees=view_degrees,
        batch=batch,
        evidence=None if generation is None else generation.name,
    )


#: The provider seams a Region binds, in the order `region_scene` passes them.
#: Anything building a Region scene through a different route must bind exactly
#: this set -- note `world_capability_provider` and `world_token_provider` are
#: deliberately **not** here even though the fixture publishes them, because
#: `region_scene` does not pass them either. Binding a different set builds a
#: different world and quietly makes two "Region" results incomparable.
REGION_PROVIDER_SEAMS = (
    "geometry_provider",
    "target_navigation_provider",
    "action_surface_provider",
    "action_surface_executor",
    "action_surface_runtime_initializer",
    "world_runtime_provider",
    "explosion_candidate_provider",
    "inventory_reset_provider",
)


def region_providers(loaded: LoadedRegion) -> dict[str, Any]:
    """The bound seams, for a consumer that builds its own environment.

    The console needs these to construct a Region scene through `AgentKit`
    rather than through :func:`region_scene`, so that one collector path serves
    both worlds. Reading the set from one constant is what keeps the two routes
    building the same thing.
    """

    return {name: getattr(loaded.fixture, name) for name in REGION_PROVIDER_SEAMS}


def region_scene(**kwargs) -> RegionScene:
    """Build the Arsenal environment with a mutable Region bound.

    Thin wrapper over :func:`load_region` -- see it for what every argument
    means. Split so a caller that wants the fixture without an environment does
    not pay for one it throws away.
    """

    from hytalegym.jax.training import make_arsenal_ppo_environment

    loaded = load_region(**kwargs)
    fixture = loaded.fixture
    environment = make_arsenal_ppo_environment(
        fixture.params,
        loaded.runtime,
        **region_providers(loaded),
    )
    return RegionScene(
        environment=environment,
        fixture=fixture,
        params=fixture.params,
        runtime=loaded.runtime,
        maximum_distance=loaded.maximum_distance,
        view_sector_full_angle_degrees=loaded.view_sector_full_angle_degrees,
        batch=loaded.batch,
        evidence=loaded.evidence,
    )


def world_action_heads(mask, capacity: int = CANDIDATE_CAPACITY) -> dict:
    """Split a 99-bit legality mask into its named heads.

    The heads that only a bound world can open are ``use_off_on``,
    ``block_primary_secondary_trigger`` and ``block_none_plus_candidates``.
    ``recipe_none_plus_candidates`` was a fourth until crafting left the policy
    surface.
    """

    from hytalegym.jax.combat.observation.v3.policy.surface import (
        action_surface_layout,
    )

    return action_surface_layout(capacity).split_mask(mask)


def _identity(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["library_semantic_sha256"]
