# hydl

**Reinforcement-learning agents trained in a JAX reimplementation of Hytale
0.5.7, built to transfer to the real server.**

The premise is that a game server is too slow to train against and too
authoritative to train without. So the combat, locomotion and world rules are
reimplemented in JAX — where thousands of environments step in parallel on a
GPU — and the resulting policy is run against the actual Hytale server through a
JVM bridge. Everything in this repository exists to keep those two worlds
telling the same story: the simulation is derived from the shipped game's own
authored data, and the parts that drift are treated as defects rather than
tuning.

> **Status: research code, pre-1.0.** One training lesson is bound end to end and
> works; five more are drafted but not runnable. The sections below say which is
> which. Nothing here is a supported product.

---

## What is actually in here

| Path | What it is |
|---|---|
| `arena/` | The task, world, evaluation, imitation and training layer. Reward laws, action-scope contracts, PPO collectors, curricula, promotion gates. Agent-neutral by design. |
| `adk/` | The agent development kit that adapts Arena. The dependency points one way: `adk` may import `arena`, never the reverse. |
| `console/` | A FastAPI service that owns training runs — preflight, launch, evidence, replay. Runs are launched through it, not by calling a trainer directly. |
| `agents/` | Concrete agents and their run profiles. `agents/basic` is the one the Console drives. |
| `worlds/` | Turning bounded captured worlds into JAX world providers. |
| `npc/` | NPC behaviour surfaces. |
| `tools/` | Repository gates, including the publication-hygiene check that CI runs. |
| `HytaleRL/hytalegym/` | The JAX gym itself — combat, locomotion, perception, world stepping. |
| `experimental/` | Work that has not earned a place in the trees above. |

**Start here:** [`arena/README.md`](arena/README.md) is the public orientation
layer for the training stack, and the READMEs beneath it cover the task
framework, contracts, rewards, evaluation and self-play.

---

## Installing

Python 3.11 or newer.

```bash
pip install -e .                 # arena, adk, console, npc, worlds
pip install -e ".[console]"      # + FastAPI/uvicorn, to run the Console
pip install -e ".[dev]"          # + pytest, ruff
```

Runtime dependencies are `jax`, `numpy`, `gymnasium` and `optax`.

**`hytalegym` is not resolved from an index.** It ships inside this repository at
`HytaleRL/hytalegym` and is put on the path rather than installed, so that the
gym and the training code always move together. Put the repository root on
`PYTHONPATH`.

A GPU is strongly recommended. JAX on CPU will run the code but not at a scale
where the curricula are meaningful.

---

## Training an agent

Runs go through the Console. That is a deliberate constraint, not a convenience
wrapper — the Console owns preflight validation, cache identity and ownership of
the evidence a run produces, and bypassing it loses all three.

```
POST http://127.0.0.1:8770/api/train/preflight    # launch only if passed: true
POST http://127.0.0.1:8770/api/train?profile_id=basic
```

Do not invoke the trainer module directly. The agent profile and Console API
boundary are described in [`agents/basic/README.md`](agents/basic/README.md)
and [`console/docs/API.md`](console/docs/API.md).

The algorithm is **recurrent PPO**. It is not selected by a flag — the collector
is a PPO loop, and the source checkpoint you pass defines the network shape.

### What is runnable

**One lesson.** `pursuit` — chase a fleeing target — is the only contract bound
to a collector.

| | |
|---|---|
| Reward law | `arena/training/contracts/pursuit.py` |
| Collector | `arena/training/runs/pursuit_run.py` |
| Console stage | `pursuit_tracking` |

Five further lessons (`guard_duel`, `punish_window`, `evasion`,
`ability_landing`, `checkpoint_route`) publish a reward law, an action scope and
a manifest, but have no collector, so the Console will not offer them. Their
registrations in `arena/training/contracts/catalog.py` carry the `blockers=`
that say why.

### Curricula are gated, and the gate runs one rung at a time

`arena/curriculum/pursuit.py` declares two ladders — an objective ladder
(`long_range` → `closing` → `contact` → `recursive`) and a terrain ladder — and
`arena/curriculum/ladder.py` owns the promote / hold / demote decision.

`POST /api/train/ladder/start` launches the **current rung only**. When it
finishes, its own selected success rate is fed to the gate, and what runs next
is whatever the gate returns: the rung above, the same rung again, or the rung
below. Queueing every rung up front is what makes a threshold decorative — two
of the three outcomes are not "the next rung" — so the driver
(`console/core/execution/ladder.py`) never has more than one rung outstanding.
`GET /api/train/ladder` reports where it is and why.

### Running without the captured worlds

The Region world libraries are large and are not published, so a fresh clone has
none. Two things follow, and both are deliberate:

- **The gym still imports.** The geometry-token contract reports its staging
  provenance as absent rather than refusing, so `import arena.training` works on
  a clean checkout. A checkpoint pinned to a staged library is still correctly
  rejected — the two contracts hash differently.
- **Flat-world training still runs.** `world_design` builds a pursuit arena from
  an authored design with no capture step (`worlds/flat.py`), which is also why
  it skips the scene build: about 6 seconds against roughly 380 for a Region
  run. `completely_flat` is a plane on which every point is standable.

Training on captured Hytale terrain needs a library you capture yourself.

---

## What is deliberately *not* in this repository

Being explicit, because the absences are load-bearing:

- **Decompiled Hytale sources.** Never published. CI fails the build if any are
  committed, independently of the ignore rules. See [`LICENSE`](LICENSE).
- **Hytale server installs, plugin jars and bridge build output.** The Gradle
  wrapper jar is the single deliberate exception, because `./gradlew` cannot
  bootstrap without it.
- **Test suites.** Both the Python and Java suites are developer material and are
  excluded. This is why CI runs hygiene and lint but no test job: a test job on a
  tree with no tests would collect zero tests and report green, which reads
  exactly like a green that passed. See the comment at the top of
  [`.github/workflows/hygiene.yml`](.github/workflows/hygiene.yml).
- **Run artifacts, checkpoints and captured world libraries.** Large, generated,
  and reproducible from the code that made them.
- **Contributor-only maintenance records.** They are not part of the user
  documentation; the README files inside each package are the public
  orientation layer.
- **The root `docs/` tree.** Design records, knowledge registers, coverage
  censuses and port critical paths — development history rather than a manual.
  The README files inside each package are the published orientation layer.

If you are working from a clone and something referenced in the docs does not
exist, this list is the likely reason.

---

## Repository gates

CI (`.github/workflows/hygiene.yml`) runs on push to `main` and on pull requests:

- `tools/check_publication_hygiene.py` — asserts excluded paths stay excluded and
  source paths stay trackable
- no decompiled sources committed
- no jars committed beyond the Gradle wrapper
- `ruff check arena agents console tools adk npc worlds`

---

## License

MIT — see [`LICENSE`](LICENSE). The license file also states the position on
Hytale's own assets and decompiled code, which are **not** covered by it and are
not distributed here.
