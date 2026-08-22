# Self-play

This package has two explicit modes. Keeping them separate prevents an immutable historical opponent from accidentally becoming trainable.

## Live duel

[`duel.py`](duel.py) creates exactly two distinct policy IDs on two physical entities. Both are active and trainable, and `make_duel_ppo_trainer()` delegates to HytaleGym’s multi-actor population trainer so each policy retains its own optimizer state while sharing one actor-major arena rollout.

## Frozen-opponent league

The league path is split across these files:

| File | Responsibility |
| --- | --- |
| [`league.py`](league.py) | Rating state, close-rating opponent sampling, Elo updates, snapshots, and frozen assignments. |
| [`production.py`](production.py) | Host-side update scheduling, one live learner, immutable roles, provenance checks, and snapshot publication. |
| [`checkpoint.py`](checkpoint.py) | Content-bound sidecar around the existing population checkpoint v1. |
| [`native_teacher.py`](native_teacher.py) | Installs a contract-checked native behavior policy into a frozen bank row. |
| [`__init__.py`](__init__.py) | Public self-play exports. |

`ProductionLeagueConfig` requires one trainable row, at least one frozen row, and a fixed policy-bank capacity. Each update samples one opponent at the host boundary, broadcasts the static assignment through the batch, runs the existing compiled PPO update, rates exact episode outcomes, and may copy the learner into a reserve row as an immutable snapshot.

## Checkpoint boundary

The population checkpoint remains the owner of policy tensors, optimizer state, counters, assignment, and selected learner export. The league sidecar adds ratings, roles, scheduler position, snapshot lineage, provenance, and the current plan. Resume reconstructs the assignment and derives deterministic schedule/train/reset keys from the persisted seed and update index.

## Outcome rules

Success, death, and simultaneous outcomes are rated; support-classified `other` endings are retained as diagnostics but are not silently converted into wins or losses. Profile/controller changes across snapshots require an explicit transfer reason.

