"""Level 1 skill library for ADR-001 (experimental/neural_agent).

Each module implements the ``Skill`` protocol from ``experimental/neural_agent/
contracts.py`` or the dispatcher that resolves a ``GoalPayload`` to one skill
(ADR-001 D1, seam 1). The package re-exports nothing. Import the submodules:

* ``_common``: shared shape checks, head offsets, mask legality, first-legal choice.
* ``idle``: the no-op skill, catalog index 0.
* ``pursuit``: the pinned pursuit_contact candidate v0001 behind its scoped loader.
* ``evader``: the scripted evader, the shipped variant (see SKILL.md).
* ``library``: ``SkillRegistry`` and ``build_default_registry``.
* ``dispatcher``: ``SkillDispatcher``, the ADK ``LowLevelActor``.
"""
