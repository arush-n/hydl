"""Composition task definitions for the ADR-001 neural agent (ADR-001 P2).

The package imports no submodules, because ``flat_arena`` loads the Arena pursuit
runner and that import is heavy. Import the modules you need:

* ``flat_arena``: the two-actor flat pursuit arena, built from the candidate's own
  construction, with the ``built`` interface that the ADK collectors read.
* ``composition``: the pursue, evade, resume task, its executive state and its
  success predicate.
* ``events``: the DAMAGE_RECEIVED read helper and the ablation injector.

Read ``SKILL.md`` for the formulas, the thesis and the citations.
"""
