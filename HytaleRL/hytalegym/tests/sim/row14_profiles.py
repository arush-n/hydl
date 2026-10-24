"""Profile for the row-14 differential: the gen-7 profile with the perception_v2 construct switched on.

The gen-7 profile ships ``perception_v2`` off (``hytalegym/sim/common/profile.py``, default False). The row-14 test renders
this profile as a third variant so the learner observation carries the channel's entity fields for the privilege check.
"""

from dataclasses import replace

from hytalegym.sim.gen7.profile import PROFILE as _GEN7

PERCEPTION_V2 = replace(_GEN7, constructs=replace(_GEN7.constructs, perception_v2=True))
