"""Hytale 0.7.x, served by the bridge port in ``hytale-plugin-0.7.2``.

The simulator rules are still the 0.5.7 calibration, so native evaluation
accepts 0.5.7 checkpoints here as a cross-version run; see
``hytale-plugin-0.7.2/PORT-0.7.2.md`` for the gameplay that differs.
"""

from hytalegym.server_versions import ServerProfile

PROFILE = ServerProfile(
    release_line=(0, 7),
    certified_version="0.7.2",
    bridge_jar="hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar",
    bridge_jar_sha256=(
        "49CDB40809D1469610758C8DDAA19EF53EA9F6DAD095292D1044A2FF0DC3221E"
    ),
    crafting_recipe_count=2_047,
    checkpoint_rulesets=("0.5.7",),
    region_library="region-library-pilot-v2-072",
    world_fixture_root="v0_7",
    world_capacity_report="region-library-pilot-v2-072/world-token-capacity.json",
    region_library_sha256="9DCA3C71D43A2C3BF4BE6AFF6E2BD24D2F6608C8BE01D1687A0B259F06DAAB95",
    seed_projection_sha256="408D5FB2DAA797E767AEE23248C6591FC226E1FE764C6258353D60FE49D1104D",
    # 96 keeps every sampled in-core row (max 82 tokens; 44 keeps 34-42 %).
    world_geometry_token_capacity=96,
    world_geometry_capacity_evidence=(
        "artifacts/fidelity_evidence/region_token_capacity/census_3seeds_200.log"
    ),
    # Live 0.7.2 has no window gate on the bow (receipt
    # artifacts/fidelity_evidence/gen7_probes/shortbow_tiers_b4eec032.json).
    projectile_clear_past_window_edge=True,
)
