# Executable skill stages

`training/skills` is the reusable mechanics layer for two-actor lessons. It turns a declared strategy into executable target motion, physical reset, action masking, learner-owned causal signals, and content-addressed artifacts.

## File guide

| File | Responsibility |
| --- | --- |
| [`program.py`](program.py) | `FactoredActionScope`, `TargetMode`, lane-stratified target motion, flee/standoff compass choices, and fail-closed neutral fallback. |
| [`signals.py`](signals.py) | Pure-JAX pursuit, yaw/pitch/spatial tracking, sustained alignment, attack timing, interval outcomes, and episode-progress signals. |
| [`stage.py`](stage.py) | Radial pair placement, canonical Arsenal reset, learner/target role validation, stage binding, parameter masks, and scripted target action masks. |
| [`__init__.py`](__init__.py) | Public skill-stage exports. |

## Target motion

`TargetMotionProgram` supports stationary, strafe, approach, flee, approach-then-hold, weave, adaptive-gait, standoff, and frozen-policy lanes. Lanes are assigned by deterministic modulo scheduling and are offset so an entire batch does not change direction or gait simultaneously. If a requested row is illegal for the current action surface, the issued row becomes the exact physical neutral and exposes an invalid/support flag.

Frozen-policy lanes require a checkpoint hash and an executable `FactoredActionScope`. The scope must include the physical neutral action so an unsupported policy can fail closed.

## Causal signals

Pursuit and aim signals use counterfactuals to separate learner motion/turning from target-produced motion. Attack timing remains pending until accepted engine lifecycle and attributed damage evidence resolves it; teacher predictions and geometry are diagnostics, not truth labels. Episode progress distinguishes natural terminal, task truncation, no-progress, horizon, and support failure.

## Stage identity

`ArenaSkillStageProgram` binds strategy, target motion, reset, action heads, teacher metadata, and episode law. `bind_arena_skill_stage_identity()` adds combat parameters, runtime configuration, assignment, world, action, and dynamics hashes. The resulting identity explains exactly which physical setup an artifact belongs to.

