# Curriculum

`arena.curriculum` describes how an agent moves through increasingly difficult task variants and how a training objective is made reproducible.

## `ladder.py`

Defines:

- `Stage`: one task/difficulty rung with promotion and demotion thresholds.
- `LadderState`: current rung plus evaluation history.
- `Ladder`: ordered promotion logic.

Promotion uses a moving window and requires sustained evidence; a single strong evaluation does not skip rungs. Demotion is also thresholded and cannot move below the first rung. The ladder records rates rather than policy internals, so it can consume results from `GameManager`, Arena training, or another evaluator.

`Ladder.record()` returns `promote`, `hold` or `demote`. **Two of those three
are not "the next rung"**, which is why a caller cannot queue a whole ladder up
front and still claim to be gated: the run after this one is not knowable until
this one has been graded. The console's driver
(`console/core/execution/ladder.py`) keeps exactly one rung outstanding for that
reason.

## `strategy.py`

Defines content-addressed training semantics:

- `SkillAxis` names the capability being trained, such as pursuit, aim, attack timing, or guard timing.
- `TargetMotionContract` describes the target patterns a lesson must expose.
- `EpisodeContract` fixes horizon, no-progress patience, terminal/truncation, bootstrap, and optimizer-chunk rules.
- `ArenaTrainingStrategy` combines objectives, target motion, and episode semantics into a stable manifest and hash.

The strategy hash is intended to travel with an artifact. It prevents a checkpoint trained under one target controller or reward law from being described as if it came from another.

## Relationship to training

Curriculum does not run PPO and does not own policy parameters. It supplies the identity and progression rules consumed by training and evaluation. For executable lesson contracts, continue to [`../training/contracts/`](../training/contracts/README.md) and [`../training/runs/`](../training/runs/README.md).


## The pursuit curriculum lives in `pursuit.py`, not in the loop

There are **two** ladders here and they move independently:

| ladder | rungs | asks |
| --- | --- | --- |
| `objective_ladder()` | `long_range` → `closing` → `contact` → `recursive` | what the evader is being told to DO |
| `terrain_ladder()` | flat plane → `flattest` captured worlds → the rugged `mixed` tail | what ground it has to do it on |

The objective ladder runs first. Terrain difficulty is meaningless until the
agent understands the game it is playing on that terrain.

`GET /api/train/curriculum` serves `pursuit.describe()`, so "what gets harder,
and when" has ONE definition rather than a copy in the console and another in
the loop.

### The bottom rung pays for distance, and nothing for proximity

`long_range` sets `separation_reward` and leaves `contact_reward` at zero, which
is what `EvaderDuelConfig.band_active` reads to mean "there is no standoff
band". Every band-relative ordering guard then compares against a term that does
not exist, which is deliberate: this is the learn-to-move configuration.
`separation` is the only term in the contract keyed on distance rather than on
motion or posture — every other one pays a run and a circle identically — so it
is the only way to express "open the gap and stay alive". From `closing` up the
band comes on and separation goes to zero: the two objectives cannot both lead.

### `describe()` carries what a rung needs to be launchable

`settings` alone is not a runnable rung. The payload therefore also states the
episode length per rung (`horizon_by_stage`), and the role — `objective:
"evade"` with `target_controller: "frozen_source_policy"`, because
`pursuit_run` refuses an evade objective whose opponent does not pursue. Stating
the role once beside the rungs is what stops a caller inferring it from the
`evader_` field prefix and pairing it wrong.

**The horizon is deliberately not a rung.** Shorter episodes favour the evader
and longer ones favour the chaser, so promoting on it would promote one side by
handicapping the other. `HORIZON_BAND` records the range (128 / 256 / 512 ticks
= 4.3 / 8.5 / 17 s at 30 TPS) and the loop moves within it; the ladder never
touches it. A test pins that no rung carries a horizon key.

## `Rung` is why a non-task curriculum can reuse `Ladder`

`Ladder` never touched `stage.task` — it reads a name and four thresholds. That
is now stated as the `Rung` protocol, with two implementations:

| rung | difficulty is | used by |
| --- | --- | --- |
| `Stage` | a `Difficulty` name on an arena `Task` | `ladder_for(task)` |
| `SettingsStage` | an opaque settings mapping | the pursuit curriculum |

`SettingsStage.settings` is opaque on purpose: this package owns WHEN to
advance, the caller owns what advancing means. Threshold validation is shared
through `_validate_thresholds` so the two rung types cannot drift.
