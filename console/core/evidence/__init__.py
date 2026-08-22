"""What a run leaves behind.

``store``              persist each console run as a self-describing artifact
``results``            normalize result declarations without learning a lane's
                       metric semantics
``evidence_contract``  validated atomic writes for the shared six-field contract
``diagnostics``        structural extraction for diagnostics already in artifacts
``agent_archive``      synchronise any agent's receipts into the shared archive
"""

from __future__ import annotations

from console.core.evidence import agent_archive as agent_archive
from console.core.evidence import diagnostics as diagnostics
from console.core.evidence import evidence_contract as evidence_contract
from console.core.evidence import results as results
from console.core.evidence import store as store


__all__ = [
    "agent_archive",
    "diagnostics",
    "evidence_contract",
    "results",
    "store",
]
