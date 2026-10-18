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
        "5FE609E23F7CBAF08A27374FE23C2016D66003C8199DE6E378D8E55F4779ADAF"
    ),
    crafting_recipe_count=2_047,
    checkpoint_rulesets=("0.5.7",),
)
