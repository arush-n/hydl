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
        "460054A8E7508FE5E205AD527AFB08474B61626EE5C2212FC2A484ECFCCAD88D"
    ),
    crafting_recipe_count=2_047,
    checkpoint_rulesets=("0.5.7",),
    region_library="region-library-pilot-v2-072",
    world_fixture_root="v0_7",
    world_capacity_report="region-library-pilot-v2-072/world-token-capacity.json",
    region_library_sha256="9DCA3C71D43A2C3BF4BE6AFF6E2BD24D2F6608C8BE01D1687A0B259F06DAAB95",
    seed_projection_sha256="408D5FB2DAA797E767AEE23248C6591FC226E1FE764C6258353D60FE49D1104D",
)
