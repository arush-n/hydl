"""Who can run, and what they are.

``agents``      agents discovered from ``agents/`` and what each needs
``profiles``    what an agent *is*, not what test it happens to run
``identities``  declared identities compared with live or staged counterparts
"""

from __future__ import annotations

from console.core.catalog import agents as agents
from console.core.catalog import identities as identities
from console.core.catalog import profiles as profiles


__all__ = [
    "agents",
    "identities",
    "profiles",
]
