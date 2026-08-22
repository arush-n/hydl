# API

Interactive reference at **http://127.0.0.1:8770/api/docs** while the server
runs. This page covers what the generated docs cannot: what the numbers mean and
which ones lie.

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/api/options` | choices for the UI controls, read from the ADK |
| `GET` | `/api/worldgen/options` | custom-world fields, presets and the native boundary |
| `POST` | `/api/worldgen/preview` | deterministic semantic terrain/design preview |
| `POST` | `/api/worldgen/batch` | up to 512 compact seed or recipe-variation previews |
| `GET` | `/api/worldgen/builds` | validated local V2 compiler receipts |
| `GET` | `/api/worldgen/captures` | exact composite worlds available for voxel inspection |
| `GET` | `/api/worldgen/captures/{id}/volume` | render one authenticated captured composite without regenerating it |
| `POST` | `/api/worldgen/compile` | compile one design to an offline pinned V2 asset pack |
| `GET` | `/api/worldgen/designs` | saved local design manifests |
| `POST` | `/api/worldgen/designs` | validate and atomically save one design |
| `GET` | `/api/worldgen/designs/{id}` | restore one normalized saved design |
| `GET` | `/api/custom/minigames/options` | Arena goals and loadouts accepted by the declarative creator |
| `GET` | `/api/custom/minigames` | saved custom-minigame manifests, newest first |
| `POST` | `/api/custom/minigames` | validate and save one task against one unique WorldGen design |
| `GET` | `/api/custom/minigames/{id}` | restore one saved creator manifest |
| `POST` | `/api/run` | compile and execute one rollout |
| `GET` | `/api/runs` | every stored run, newest first |
| `GET` | `/api/runs/{id}` | one run's full report |
| `GET` | `/api/runs/{id}/replay` | report, trajectory, and lightweight recorded 3D surface |
| `GET` | `/api/compare` | whether two runs are comparable, then what differed |
| `GET` | `/api/profiles` | agent profiles and their timestamped replay bindings |
| `POST` | `/api/profiles/{id}/replays` | bind an existing stored trajectory to a profile |
| `POST` | `/api/train` | start one PPO run on a background worker |
| `GET` | `/api/jobs` | training jobs this process is running or has run |
| `GET` | `/api/jobs/{id}` | one job's live state and latest milestone replay payload |
| `GET` | `/api/jobs/{id}/replay` | latest completed fixed-seed evaluation replay |
| `GET` | `/api/jobs/{id}/metrics` | the per-update metric series, read off disk |
| `GET` | `/api/jobs/{id}/checkpoint` | download the trained weights |
| `POST` | `/api/jobs/{id}/cancel` | stop after the current update |
| `GET` | `/api/adk` | the ADK's description of itself |
| `GET` | `/api/bridge` | native listener reachability (read-only) |
| `GET` | `/api/policy-agent` | counters from the policy running inside the Java server |

The default replay uses the recorded standable cells as a lightweight 3D
surface and exposes the Region seed on the Map. This keeps boot fast and lets
the seed be carried directly into Custom Studio. Add `?blocks=true` only when
the exact captured block shell matters; legacy artifacts are then rehydrated
in memory without rewriting the stored receipt. New rollout payloads may
already include `terrain.blocks`, with parallel coordinates, material ids, and
exposed-face masks. On boot, the newest replayable Region artifact with terrain
is preloaded without launching a rollout.

Replay seed lookup is stable for backend growth: the Console prefers
`terrain.seed`, then the artifact's `spec.seed`, then a top-level backend
`seed`. Seed `0` is valid. A future seed registry can therefore populate the
top-level field without changing the map or Custom Studio handoff.

---

## `GET /api/compare`

The response separates three axes:

- `settings`: key-level `spec` differences;
- `source`: declared snapshot-manifest SHA equality;
- `contracts`: declared contract hashes plus executable observation/action
  shapes.

`verdict` is `single-variable`, `multi-variable`, `different source`, or
`different contract` when those declarations support the conclusion. Exact
replicates return `same configuration`; void runs return `void`. If either
source or contract identity is absent, the axis and verdict are `undeclared`
and `comparable` is false. Omission is never treated as equality.

---

## WorldGen design endpoints

`POST /api/worldgen/preview` accepts the flat field contract returned by
`GET /api/worldgen/options`. Omitted values use the documented defaults;
unknown, non-finite, fractional-integer and out-of-range values return **400**
instead of being silently clamped. The response contains:

- the normalized config and portable grouped design manifest;
- a 49x49 height/slope/semantic-biome grid over independent 32-512 block
  width and depth;
- derived trees, structures, bounded cave graphs, and neutral support anchors;
- relief, water, broad traversability, cave, and content-diversity metrics;
- design warnings and a SHA-256 over the spatial result; and
- an explicit warning that the preview is not native Hytale capture evidence.

Horizontal extent is `world_width` (X) by `world_depth` (Z); the console labels
the second control **world length (Z)**. API clients may send `world_length` as
an alias for `world_depth`, but sending conflicting values is rejected. The
legacy square `world_size` field remains accepted only when neither independent
axis field is present.

`design.extent.native_capture` makes size semantics machine-readable. Native
generation is unbounded, and each exact capture tile contributes a 96x96 usable
core. The plan reports tile counts, coverage, artifact requirements, and whether
the exact port contract is implemented. A one-tile request maps to centered
`region_core` bounds; larger dimensions map to a same-reset composite whose
adjacent captures are exposed as one JAX world. The loader and stitching path
exist, but `exact_capture_artifacts_required` remains true: readiness is not a
claim that a live composite has already been captured. Width/depth do not
duplicate an otherwise identical native asset recipe.

The same normalized config and seed are bit-for-bit reproducible. The digest is
not treated as sufficient evidence of diversity:
`console.tools.audit_worldgen_families` fingerprints actual height fields,
semantic surfaces, structure layouts, and cave graphs across every family.

`POST /api/worldgen/batch` accepts `config`, `start_seed`, `count` (1-512),
`stride`, `resolution` (`17`, `25`, or `33`), `strategy`, and
`variation_strength`. Strategies are `native_seeds`, `plausible_variants`, and
`wild_variants`. Its compact rows can be loaded into the full preview. The
response also carries `native_seed_plan`, including the exact seed list and a
held-out count. It is labelled `not_native_capture=true`.

`POST /api/worldgen/compile` validates the design and writes a local pack under
the Console's configured runtime directory. It
verifies pinned installed Hytale assets and returns both pack and receipt
identities. Fine-grained preview
controls that cannot yet be expressed by the template compiler are returned as
preview-only controls; the endpoint does not deploy, start Hytale, capture a
Region, publish JAX content, or alter legacy `world="hytale"`.

`GET /api/worldgen/builds` reads those receipts and validates their current
content before listing them.

`GET /api/worldgen/captures` lists exact composite manifests available through
the console's configured capture roots. The volume route addresses one by its
published id and renders the captured data without regenerating terrain; a
missing or invalid artifact is reported instead of falling back to the semantic
preview.

`POST /api/worldgen/designs` recomputes and validates the preview before saving.
Its content-addressed id is `<normalized-name>-<digest-prefix>`, so retrying an
identical save is idempotent. Files live under the Console's configured runtime
directory; this endpoint does not author native
assets, launch the server, publish a corpus or change any default pin.

WorldGen's objective field is only an environment-authoring hint for spawn
selection. Rewards, rules, scoring, teams, and minigame termination are outside
these endpoints.

---

## Custom minigame endpoints

`GET /api/custom/minigames/options` returns the finite goal, loadout, reward,
scene, and world-pool contract used by the Custom tab. `POST
/api/custom/minigames` validates those fields, compiles the goal and reward
shapes, normalizes the WorldGen design, and refuses duplicate game names or
reused world recipes before writing a local Git-ignored manifest.

A saved manifest with `jax_ready=true` proves the goal and reward contracts
compile. It does **not** prove a full rollout is ready: newly authored terrain
remains `needs_region_capture` until the required native worlds are captured.
The list and item `GET` routes restore those manifests; they do not regenerate
or deploy terrain.

---

## `POST /api/run`

Body is `console.api.schemas.RunRequest`; every field mirrors
`console.core.execution.runner.RunSpec`. Minimal call:

```bash
curl -s -X POST http://127.0.0.1:8770/api/run \
  -H 'Content-Type: application/json' \
  -d '{"loadout":"iron_mace","ticks":256,"decision_period":8}'
```

Invalid values return **400** with the reason — unknown loadout, unknown combat
parameter, out-of-range period. Validation lives on `RunSpec.validate`, so the
same errors surface when calling from Python.

### Profile ownership

Pass `profile_id` as a query parameter to record explicit agent ownership:
`POST /api/run?profile_id=basic`. The artifact stores that
id, the profile receives the latest replay for the scenario, and both use the
artifact's UTC `written_at` timestamp. The Archive filters on this explicit id;
it does not infer ownership from loadout or policy names.

Existing trajectories can be attached with `POST /api/profiles/{id}/replays`
and a body containing `scenario`, `run_id`, and an optional `label`. The server
reads the timestamp from the stored report rather than trusting the caller's
clock. Training accepts the same `profile_id` query parameter and carries it on
the live job and final artifact.

### Response

```jsonc
{
  "spec":  { ...the resolved RunSpec... },
  "steps": 256,
  "head_names":    [ ...12 action heads... ],
  "target_phases": ["idle","windup","sweep","recovery","cooldown"],
  "summary":    { ... },
  "warnings":   [ "..." ],
  "trajectory": { ...27 fields, each indexed by `steps`... }
}
```

### `summary`

| key | meaning |
| --- | --- |
| `visible_steps` | steps where the policy could perceive the target |
| `requested` / `accepted` | abilities asked for vs granted |
| `landed` | steps where damage was actually dealt |
| `hit_rate` | `landed / accepted` |
| `damage_dealt` / `damage_taken` | health deltas over the episode |
| `reward_total` | summed reward |
| `decision_interval_ms` | `decision_period / 30 * 1000` |
| `simulated_seconds` | `ticks / 30` |
| `wall_seconds` | real compute time, mostly JIT on a new configuration |

### `warnings` — read these first

The endpoint says plainly when a run is degenerate rather than leaving a flat
zero to be mistaken for a result. Emitted when:

- **the agent never perceived the opponent** — every target column is masked,
  not zero
- **no ability was ever accepted** — nothing the run says about damage means
  anything
- **the agent took no damage** — `agent_damage` and `death` are dead, so guard
  and dodge have no signal
- **abilities accepted but none landed** — the measured hit rate is 0–2 per few
  hundred steps, so a single run is noise
- **`decision_period` is 1** — a 33 ms loop; see [DECISION-RATE.md](DECISION-RATE.md)

### `trajectory` — the one field that lies

`observed_distance` is what the **policy** sees. `true_distance` is the
simulator's ground truth. **They disagree when `visible` is 0.**

When `visible[i] == 0`, `observed_distance[i]` is **masked-out data, not a
measurement**. Render it as "no reading". Displaying it as `0.00` reproduces
the exact bug this console exists to catch — a run once scored for hours on a
scene where the agent could not perceive an opponent standing two units away.

The UI honours this: the readout says *no reading* and the 3D sightline is
physically absent.

---

## `POST /api/train`

Body is the same shape as `POST /api/run` plus `updates`, `rollout_steps`,
`update_epochs` and `num_envs`. It returns immediately with a `job_id`; the run
continues on a background worker.

**`num_envs` decides whether the run learns.** It defaults to `32`, not to the
rollout's `BATCH = 2` — that batch exists so the map has something to draw, and
2 envs x 256 steps is 512 steps per update. A checkpoint can only be reloaded
into a handle built at the same width, so the value is recorded in the run
metadata rather than left implicit.

**One job at a time.** A second launch returns `409` naming the job still
holding the device — concurrent JAX runs contend for device memory rather than
sharing it. That refusal is the designed answer, not a failure.

The response echoes the **resolved** spec, not the request body. A launch that
omitted `armed` ran with `RunSpec`'s default, and a record showing `null` there
would claim the run had no value for it.

## `GET /api/jobs/{id}/checkpoint`

The trained weights as one compressed `.npz`.

| status | meaning |
| --- | --- |
| `200` | the weights, `application/octet-stream` |
| `409` | a save was **refused** — the body carries the reason |
| `404` | no weights stored for that job |

**`409` is an answer, not an error.** `handle.save_checkpoint` re-validates the
build, the PPO config and the live contracts before writing, and refuses when
they no longer agree; a checkpoint stamped against contracts that have already
moved is one nobody can load. The reason is kept in `checkpoint-refused.txt`
beside the artifact.

This route reads through to disk when the job is not in memory, so a server
restart does not orphan weights it already wrote. The job *registry* is
per-process; the artifact directory is the durable record.

**A checkpoint only loads against the scene it was trained on.**
`load_checkpoint` compares the AgentSpec digest, the scene identity and the
arsenal runtime capacity, and raises rather than returning weights whose
observation columns would mean something else. Confirmed in both directions: the
same configuration loads in a fresh process; a different loadout or opponent is
refused.

---

## `GET /api/adk`

Scene descriptions with contract SHAs and bound providers, loadout names, action
heads with offsets and sizes, observation groups with column counts, and the
provider-seam list — each read from the ADK rather than restated, so this
endpoint tracks the library automatically.

Current values: **124 logits across 12 heads, 241 named columns across 28 groups,
16 provider seams, 31 loadouts.**

---

## `GET /api/bridge`

```jsonc
{
  "listeners": [
    {"port": 5556, "role": "native",    "listening": true},
    {"port": 5557, "role": "simulator", "listening": true}
  ],
  "any_up": true,
  "note": "Read-only. ..."
}
```

**Reachability only.** The console never opens a `NativeBridgeSession`, because
that takes the native evidence lease and would evict whatever lane holds it. A
listening socket means a process is bound to the port — it does **not** mean the
deployed jar matches the Gym's canonical pin. For that, run:

```bash
pytest adk/tests/test_crosslang_fidelity.py adk/tests/test_bridge_fidelity.py
```

which manages the lease properly and reports "not evaluated" rather than
stealing it.

---

## `GET /api/policy-agent`

Counters published by `HytalePolicyAgent`, the mod that runs a JAX-trained
policy inside the Java server (`experimental/jvm-agent`).

```jsonc
{
  "state": "live",              // live | stale | absent | unreadable
  "path": ".../mods/policy-agent/metrics.json",
  "age_seconds": 2.4,
  "stale_after_seconds": 30.0,
  "metrics": {
    "role": "*", "claimed": 117,
    "policy_ticks": 18243, "policy_ticks_per_second": 230.4,
    "world_tick": 2775, "world_tps": 26.0, "world_ticking": true,
    "skipped_no_observation": 0, "rejected_illegal": 0,
    "dropped_world_verb": 0, "skipped_interactions": 18243,
    "observation_size": 8272, "action_size": 124, "recurrent_size": 128,
    "capture_non_zero_columns": 97, "capture_legal_actions": 45,
    "last_skill": 2, "last_world_move": 6, "last_yaw_delta_degrees": 33.75,
    "last_jump": true, "last_dodge": 4, "last_action_legal": true,
    "last_value": -0.1418
  }
}
```

**A file, not a port.** The mod writes `metrics.json` next to its weights every
5 s and the console reads it. That keeps the console a pure viewer — the same
reason `/api/bridge` refuses to open a session — and the numbers survive the
server going down, which is when the last known state is most worth seeing.
Writes are atomic (temp + `ATOMIC_MOVE`), so a parse failure is real corruption
and reported as `unreadable` rather than silently as `absent`.

**Four states, because the fix differs.** `absent` = no mod deployed anywhere
searched; `stale` = a file exists but nothing is writing it (server stopped, or
the mod never armed); `live`; `unreadable` = corrupt. A stale reading still
returns its full metrics — hiding them would discard the only evidence of what
the agent was doing when it stopped.

By default it looks in the Hytale install's `mods/policy-agent/`. Point it at
another instance — an isolated test server, say — with:

```bash
HYTALE_POLICY_AGENT_DIR=/path/to/policy-agent python -m console.server
```

### Reading the numbers

* `world_tps` below 30 means the **server** is behind, not the policy —
  inference is 0.0116 ms, about 0.035% of a tick.
* `skipped_no_observation` climbing means `PolicyPerception` returned `null`;
  those NPCs fell back to authored behaviour rather than being fed zeros.
* `skipped_interactions` is reported by the active action-sink composition. If
  it climbs, inspect the selected interaction heads and facade availability;
  current source includes combat and World sink layers, so the counter does not
  by itself prove that only `SteeringActionSink` was built or deployed.
* `capture_non_zero_columns` at 0 means the observation source degraded to
  zeros, which still produces a confident action.
