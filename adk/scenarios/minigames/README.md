# Shaping minigames

These minigames are small, additive objectives for behaviors that the native
combat reward does not distinguish by itself. They read state or transition
events and add a shaping term; they do not replace the native reward and do not
define a complete Arena task or self-play league.

The common builders live in [`framework.py`](framework.py). The registry in
[`__init__.py`](__init__.py) exposes objectives including `sprint`, `flee`,
`spacing`, `footing`, `strike`, `accuracy`, `dodge`, `tracking`, `reach`, and
`checkpoint`. Objectives that require a goal, route, armed opponent, or special
geometry declare that requirement instead of inventing a default.

Use them through `SceneConfig(minigame=...)`, the Console's ordinary scene
runner, or [`adk/tools/run_minigame.py`](../../tools/run_minigame.py). For a
complete two-role pursuit lesson, use Arena's
[`training/contracts/`](../../../arena/training/contracts/README.md) instead.
