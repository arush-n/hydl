# Task framework

The framework turns a named objective into a validated, runnable JAX scene. Its modules are intentionally small and composable.

## File guide

| File | Responsibility |
| --- | --- |
| [`base.py`](base.py) | `Difficulty` and `Task`; validates rungs, loadouts, opponent profiles, world/reward types, and inferred action-head coverage. |
| [`criteria.py`](criteria.py) | Column selectors, time reducers, availability-aware predicates, and boolean combinators. |
| [`goals.py`](goals.py) | Parameterized goal factories such as survival, defeat, visibility, reach, hazards, and terrain change. |
| [`rewards.py`](rewards.py) | `RewardConfig`, named reward signals, potential shaping, event/state/completion terms, and goal-to-reward mappings. |
| [`jax_gate.py`](jax_gate.py) | JIT probes that verify goal output shape/type and reward finiteness before runtime use. |
| [`validate.py`](validate.py) | Synthetic-observation conformance checks for tasks and optional scene construction. |
| [`registry.py`](registry.py) | Explicit task registration, replacement, lookup, and reports. |
| [`__init__.py`](__init__.py) | Public re-exports for `arena.tasks.framework`. |

## Criteria and recording

`col(group, feature)` is a named selector. Built-in reducers collapse the time axis while preserving one result per environment lane. Selectors attach the required observation groups to the criterion, allowing `GameManager` to record only what the scorer needs. Opaque custom callables fall back to all readable groups unless the caller supplies a recorder.

Availability is part of the semantics. For example, a hidden target has zero-filled target values, so a target-health criterion must mask on visibility before reducing. `ever_below()` fails closed when availability is false.

## Goals and rewards

A `Goal` carries a stable kind/label, description, implied action heads, parameters, minimum rollout length, world requirement, and availability requirements. `Task` can use a `Goal` directly as its success callable.

`TaskReward` runs inside the public JAX transition path. Potential shaping uses `discount * potential(next) - potential(current)` with terminal potential zero. Completion evidence is exact only when the underlying signal provides it; for target defeat, that means `episode_success`, not merely a generic terminal flag.

## Validation

The JAX gate compiles a small synthetic probe and refuses incorrect boolean shapes, unreadable groups, unavailable completion evidence, non-finite rewards, and too-short rollout windows. This is why authoring failures happen before a paid rollout rather than halfway through one.

