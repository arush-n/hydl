# `arena.tasks` compatibility namespace

This directory contains a small compatibility package, not a second task implementation.

`arena/tasks/__init__.py` redirects the package search path to [`arena/league/tasks/`](../league/tasks/README.md), then re-exports the public task names (`Task`, `Goal`, `WorldSpec`, `RewardConfig`, registries, and validation helpers).

Use `arena.tasks` in existing callers when that is the established import path. Read and modify the implementation under [`league/tasks/`](../league/tasks/README.md) so there is only one source of task semantics.

