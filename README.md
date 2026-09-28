# HYDL

HYDL is a high-throughput software twin of Hytale for training autonomous game
agents.

It reimplements Hytale's structured game state and rules in JAX so many worlds
can run in parallel inside compiled array code. A Java bridge then runs the
policy against the authoritative Hytale server for validation.

The current policy uses numerical state, action-allowed flags, and target
features. It does not use screenshots or video.

## What is proven

- HytaleGym's combat, locomotion, and world state machine runs in batched,
  JIT-compatible JAX array code.
- A trained HYDL policy has been loaded through the Java bridge and run by an
  authoritative Hytale server. It received structured game state and controlled
  a native actor toward an assigned target using legal actions. This shows that
  the policy deployment and locomotion path works for the tested task surface.
- Exported policies carry their input and action requirements. The runtime checks
  compatibility and action legality before taking control of an actor.
- Training, task definitions, policy execution, and server validation are
  separate parts of the codebase, so an agent can be evaluated independently of
  the simulator that trained it.

This establishes a working policy-to-server deployment path. It does not show
that Hytale and JAX behave identically or that performance transfers unchanged
across tasks and worlds.

## Current gaps

1. Measure how well policies transfer by evaluating the same tasks and starting
   conditions in JAX and Hytale across repeated runs.
2. Expand reliable end-to-end training beyond pursuit; other task interfaces are
   not all runnable end to end yet.
3. Validate more movement and world interactions against Hytale. The current
   policy interface uses structured numerical state rather than screenshots or
   video.
4. Verify compatibility with future Hytale server releases.

## Architecture

```text
Hytale authored data
        |
        v
HytaleGym: batched JAX state machine
        |
        v
Arena: tasks, rewards, policies, training, evaluation
        |
        v
PPO / self-play / rollouts

Hytale server <-> Java bridge <-> policy runtime <-> validation evidence
```

### Main directories

| Directory | Responsibility |
|---|---|
| `HytaleRL/hytalegym/` | JAX environment: combat, motion, collision, observations, actions, geometry, and world stepping. |
| `arena/` | Tasks, rewards, world binding, rollout collection, PPO, and evaluation. |
| `adk/` | Deployment bundles, runtime setup, provenance, and transfer validation. |
| `experimental/jvm-agent/` | Java policy execution, native state projection, action decoding, and server validation. |
| `console/` | Training-job API and run management. |
| `agents/` | Training profiles and policy artifacts. |
| `worlds/` and `npc/` | World providers and NPC/task behavior surfaces. |

The current end-to-end training lesson is pursuit. Other task interfaces exist,
but they are not all runnable end to end.

## Setup

Requirements:

- Python 3.11 or newer
- JAX, NumPy, Gymnasium, and Optax
- A GPU is recommended for training

From the repository root:

```bash
python -m pip install -e ".[dev]"
export PYTHONPATH="$PWD:$PWD/HytaleRL"
```

PowerShell:

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "$PWD;$PWD\HytaleRL"
```

`hytalegym` is shipped in this repository rather than resolved from a package
index, so the `HytaleRL` path is required.

## Train in JAX

Training is owned by the Console so the task, environment, policy, and evidence
contracts are recorded together.

```bash
python -m console.server --help
```

Use the Console API documented in [`console/docs/API.md`](console/docs/API.md)
to run the `basic` pursuit profile. JAX training does not require a running
Hytale server.

For an environment throughput measurement:

```bash
python tools/bench_env_throughput.py --out results
```

## Test the simulator

```bash
python -m pytest \
  HytaleRL/hytalegym/tests/jax/combat/runtime/test_airborne_player_motion.py \
  HytaleRL/hytalegym/tests/jax/combat/motion/test_target_vertical.py \
  HytaleRL/hytalegym/tests/jax/test_geometry.py -q
```

```bash
python -m ruff check HytaleRL/hytalegym/hytalegym
```

## Native validation

Native validation requires an installed Hytale server, matching assets, a
compatible JDK, and bridge/plugin build outputs. Those runtime files are not
distributed here.

Use this lane to answer three questions:

- Does the Java policy execute inside the real server?
- Does the server provide the state and action evidence expected by the policy?
- Does behavior survive the move from the fast twin to the authoritative game?

A policy replay is not a full environment-fidelity result. A locomotion result
is not a combat result. Every native result must state its server version, task
setup, accommodations, and evidence type.

## Boundaries

- The current policy interface is structured state, not vision.
- Native evidence is bounded to specific fixtures and accommodations.
- Full sim-to-server certification has not been issued.
- Hytale server installs, assets, captured worlds, private fixtures, and runtime
  jars are not distributed.
- Decompiled Hytale source code is not distributed.

## License

MIT. Hytale's own assets, server files, and decompiled code are not distributed
under this license.
