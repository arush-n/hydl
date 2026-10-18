"""Gen-5 profile: Hytale 0.5.7, the generation today's engine code models.

Data provenance (relabelled 2026-10-18): ``data/`` is the 0.5.10 extraction. Its
``data/MANIFEST.json`` says ``game_version`` 0.5.10 (the installed ``v0.5`` ``Assets.zip``).
No 0.5.7 archive exists on this machine (a name search with Spotlight and the Hytale
install folders found only the 0.7.2 and the 0.5.10 archives), so the tables cannot be
re-extracted at 0.5.7. What certifies 0.5.7 is per file: ``evidence_057.json`` shows 28
asset files, 240 weapon interaction graphs and 41 auxiliary sources identical to their
0.5.7 pins (the whole-archive hash differs: pinned 1B8802C2, installed 6D14386D).
``calibration_057.json`` holds the values the engine keeps that the extracted tables do
not reproduce, each classified, so the engine behind this profile reproduces today's
rollouts. The identity gate is code-only: a gen-5 render applies no content redirect (the
gen-5 overlay is the identity, asserted by tests/sim/test_render.py), so no table or its
label feeds it, and no measured claim changes with this relabel.

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
