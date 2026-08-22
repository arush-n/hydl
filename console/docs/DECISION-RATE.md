# Decision rate — the agent is tick-quantised

**The agent's unit is the tick, not the millisecond.** It observes on a tick,
decides on a tick, and the engine advances in ticks. Wall-clock time only says
how fast we got through those ticks, and going faster is a pure win with no
effect on what the agent is.

Milliseconds enter in exactly one place: a **deployed** agent on a real Hytale
server has to answer inside that server's tick. That is a production latency
budget. Nothing in training should be tuned against it, and in particular
nothing should be throttled to match human reaction time — see
[Production only](#production-only) at the bottom.

## Two knobs, both in ticks

| knob | range | what one step costs | what the agent sees |
| --- | --- | --- | --- |
| `microticks` | 1–4 | **one** policy evaluation, one env step | one observation per N engine ticks |
| `decision_period` | 1–128 | **N** policy evaluations, N env steps | an observation every tick, acts on every Nth |

They compose. `decision_every_ticks = decision_period * microticks` is reported
in every run summary and is the number that describes the agent.

### Prefer `microticks` for N ≤ 4

Both express "act every N ticks", but they are not the same thing and one is
much cheaper.

`microticks` is the engine's own mechanism. From `combat/env.py`: *"The
edge-triggered action is applied once. Movement intent persists while
`lax.scan` advances up to four engine microticks."* One decision, one
observation, N ticks of world. The scan is bounded at `MAX_MICROTICKS = 4` and
**masked** — `can_tick = microtick_index < params.microticks` — which is how a
single compiled scan serves the whole 1–4 range without recompiling.

`decision_period` is action-repeat bolted on above the environment. The inner
policy is evaluated **every tick** and the result discarded when held, because
branching on a traced step counter would make the trace shape depend on a
traced value; `jnp.where` keeps it constant. So N ticks cost N policy
evaluations and N env steps to produce one decision.

For N > 4, `microticks` runs out and `decision_period` is the only option.

`microticks` must be set as a top-level knob, never through `parameters` — the
scene and the `AgentSpec` both carry it and `build()` rejects a mismatch rather
than silently preferring one. The console validates this up front instead of
letting it fail inside the scene builder.

## It changes what the agent is

Not a cosmetic knob. Measured at 96 ticks, `pulse_ability` with request period
4, same seed:

| `decision_period` | action changes | accepted | landed |
| --- | --- | --- | --- |
| 1 | **95 / 95** | 8 | 1 |
| 8 | **11 / 95** | 5 | 0 |

The 11/95 is the arithmetic check that the throttle works — 95 steps at one
decision every 8 ticks is ~11.9 changes. Ability acceptances fall from 8 to 5
because held actions cannot re-request on the tick the mask happens to open.

**In training, acting every tick is correct**, and it is the default. Both
fighters run at the same rate, so there is no asymmetry to correct for, and
throttling only throws away sample efficiency.

## Interaction with request period

Two different periods, easy to confuse:

- **`period`** — how often the *policy wants* to fire an ability
- **`decision_period`** — how often the policy is *allowed to change its mind*

If `decision_period > period`, the policy cannot express its intended cadence:
requests land only on decision boundaries. Both are reported in ticks.

## Throughput

Nothing throttles the simulation. `TICKS_PER_SECOND = 30` appears in the runner
only to derive display values — there is no sleep and no pacing anywhere in
`console/`. The environment runs as fast as the backend allows.

Measured on CPU, 256-step rollouts, PPO training:

| envs | s/update | env-steps/s | ticks/s per env |
| ---: | ---: | ---: | ---: |
| 8 | 4.01 | 511 | 63.9 |
| 32 | 11.53 | 711 | 22.2 |
| 64 | 16.04 | 1021 | 16.0 |
| 128 | 23.55 | 1391 | 10.9 |

Peak total throughput is at the widest batch; per-env tick rate is highest at
the narrowest. 8 → 128 envs buys 2.72× against a 16× ideal — **17% scaling
efficiency**, which is what a CPU backend saturating its cores looks like.
Large batches are exactly where a GPU should win, which makes the WSL CUDA
crash at ≥32 envs a throughput blocker rather than a curiosity.

<a id="production-only"></a>
## Production only

A real server ticks at a fixed rate, so a deployed policy has a hard latency
budget per tick. `decision_interval_ms` and `simulated_seconds` are reported for
that purpose and are derived from `TICKS_PER_SECOND`. They are the right numbers
to check before deploying to a live server and the wrong numbers to tune a
training run against.

There used to be a warning on every `decision_period == 1` run telling the
reader to raise it to 6–8 ticks for "human reaction latency". It fired on every
default run and was wrong to fire: one decision per tick is correct for
training, and a human's reaction time is not a constraint on a training
environment. It has been removed.

## Related

Sweeping request cadence is separately load-bearing — damage swings with it in
*both* directions, and single-cadence weapon censuses have produced wrong
answers three times. The public capability vocabulary is in
[`adk/capability/README.md`](../../adk/capability/README.md).
