"""Trivial agents that exist to break the environment, and the checks they trip.

Nothing here learns. A probe is a few lines of policy whose only job is to
drive the environment into states a trained agent reaches rarely or never, so
that a property violation has a chance to show up.

```python
from adk.probes import audit, uniform_legal

report = audit(handle, uniform_legal, steps=128)
print(report.describe())
```

Three pieces:

* :mod:`adk.probes.policies` -- the drivers (`uniform_legal`, `repeat`,
  `prefer`, `head_sweep`, `head_random`, `sequence`).
* :mod:`adk.probes.invariants` -- properties that must hold for every scene
  and every policy. Traced, sync-free, ``True`` means violated.
* :mod:`adk.probes.audit` -- runs a probe and reports what broke, plus the
  whole-run checks (determinism, reset stability) that a single step cannot see.

Collection never synchronizes; reporting does, once, at the end.
"""

from adk.probes.audit import (
    AuditReport,
    Violation,
    audit,
    compare_policies,
    invariant_recorder,
    is_deterministic,
    reset_is_deterministic,
    responds_to_actions,
)
from adk.probes.liveness import Liveness
from adk.probes.staleness import (
    all_observation_groups,
    column_activity,
    constant_columns,
    describe_columns,
    observation_groups,
)
from adk.probes.crosslang import (
    MissingSource,
    RulesetParity,
    all_comparisons,
    combat_ruleset_comparisons,
    describe_parity,
    disagreements,
    dodge_comparisons,
    geometry_comparisons,
    jar_ruleset_parity,
    java_constants,
    world_verb_comparisons,
)
from adk.probes.invariants import (
    INVARIANTS,
    any_violation,
    check,
    invariant_names,
)
from adk.probes.priors import (
    PRIORS,
    SENSOR_RANGE_BLOCKS,
    approach_target,
    blocks,
    guard_when_hurt,
    press_attack,
    run_away,
    shadow_without_damage,
    strike_when_ready,
    wander,
)
from adk.probes.policies import (
    ACTION_LOGIT_WIDTH,
    HEAD_SPANS,
    head_random,
    head_span,
    head_sweep,
    prefer,
    prefer_periodic,
    repeat,
    sequence,
    uniform_legal,
)

__all__ = [
    "ACTION_LOGIT_WIDTH",
    "HEAD_SPANS",
    "INVARIANTS",
    "PRIORS",
    "SENSOR_RANGE_BLOCKS",
    "AuditReport",
    "Liveness",
    "MissingSource",
    "RulesetParity",
    "Violation",
    "all_comparisons",
    "all_observation_groups",
    "any_violation",
    "approach_target",
    "audit",
    "blocks",
    "check",
    "column_activity",
    "combat_ruleset_comparisons",
    "compare_policies",
    "constant_columns",
    "describe_columns",
    "describe_parity",
    "disagreements",
    "dodge_comparisons",
    "geometry_comparisons",
    "guard_when_hurt",
    "head_random",
    "world_verb_comparisons",
    "jar_ruleset_parity",
    "java_constants",
    "head_span",
    "head_sweep",
    "invariant_names",
    "invariant_recorder",
    "is_deterministic",
    "observation_groups",
    "prefer",
    "prefer_periodic",
    "press_attack",
    "repeat",
    "reset_is_deterministic",
    "responds_to_actions",
    "run_away",
    "sequence",
    "shadow_without_damage",
    "strike_when_ready",
    "uniform_legal",
    "wander",
]
