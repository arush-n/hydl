"""Gen-5 profile: Hytale 0.5.7, the generation today's engine code models.

Data: ``data/`` was extracted from the installed 0.5.x ``Assets.zip``, which is
0.5.10 on this machine (no 0.5.7 archive exists locally; see
``data/MANIFEST.json``). ``evidence_057.json`` lists every 0.5.7 hash the
repository pins and shows the installed files identical (28 asset files, 240
weapon interaction graphs, 41 auxiliary sources). ``calibration_057.json`` holds
the values the engine keeps that the extracted tables do not reproduce, each
classified, so the engine behind this profile reproduces today's rollouts.

Constructs: all defaults (today's behaviour). Deprecated by the owner
(2026-10-17); kept selectable until retired.
"""

from __future__ import annotations

from pathlib import Path

from hytalegym.sim.common.profile import Constructs, SimProfile, StepDuration

_HERE = Path(__file__).resolve().parent

#: Effective content identity (tables plus calibration overlay) for the
#: contract stamp; ``tests/sim/test_registry.py`` fails if the data drifts.
CONTENT_SHA256 = "BF735B65F435F748D113E0E398997A5DD94F77FD9ABBCAFFCF8D7572B7626F30"

PROFILE = SimProfile(
    generation="gen5",
    certified_version="0.5.7",
    release_line=(0, 5),
    data_dir=_HERE / "data",
    constructs=Constructs(),
    # Today's ruleset engine block: 1/30 s nominal, 0.045 s loaded, 3 profiles.
    step_duration=StepDuration(),
    calibration_overlay=_HERE / "calibration_057.json",
    deprecated=True,
)
