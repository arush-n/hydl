"""Request shapes for the HTTP layer.

Deliberately a mirror of :class:`console.core.execution.runner.RunSpec` rather than a
reuse of it: the dataclass is the engine's contract and may carry types that do
not serialise, while this is the wire contract. Adding a knob means adding the
field in both places, which is the cost of keeping the engine usable without
FastAPI installed.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StreamLogRequest(BaseModel):
    """One point on a metric stream, from any learner in the repo.

    The console does not decide what a metric is: a Dreamer logs world-model
    loss and imagined return, a transformer logs attention entropy, PPO logs
    its four. Whatever keys appear here get charted, and the stream is created
    by its first log line so nothing has to be registered first.
    """

    step: int | None = Field(
        None,
        description="X position. Defaults to the number of points already logged.",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Metric name to value. Non-finite values are recorded as rejected "
            "rather than plotted — a NaN loss is usually the finding, and it "
            "must not look like a gap in the line."
        ),
    )
    meta: dict[str, str] = Field(
        default_factory=dict,
        description="Free-form labels — architecture, checkpoint, git sha.",
    )


class ProfileReplayRequest(BaseModel):
    """Bind an existing stored trajectory to one agent profile."""

    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(min_length=1, max_length=128)
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    label: str | None = Field(None, min_length=1, max_length=160)


class RunRequest(BaseModel):
    """One rollout configuration. Field docs surface in /api/docs."""

    loadout: str = Field("iron_sword", description="Agent combat profile.")
    opponent: str | None = Field(
        "iron_sword", description="Opponent profile, or null for none."
    )
    armed: bool = Field(
        True, description="Bind an opponent ability provider so it fights back."
    )
    perceive: bool = Field(
        True, description="Bind the world capability provider so the agent can see."
    )
    target_active: bool = Field(
        True, description="False makes the target inert — the negative control."
    )

    world: str = Field(
        "open_flat",
        description=(
            "'open_flat' pins every world query true — the permissive control "
            "the Gym's own readiness gates use, with no terrain to draw. "
            "'region' binds real captured geometry and is the only world the "
            "map can show ground for; measured 2.6x the wall clock."
        ),
    )
    zone: str | None = Field(
        None,
        description=(
            "Region only: '<region_seed>:<component>' from "
            "worlds.zones. A zone is a strongly-connected set of "
            "standable nodes, so an agent cannot spawn in a one-way pocket it "
            "can never leave. Null takes the library's default spawn."
        ),
    )

    # These two were added to RunSpec and NOT here, and pydantic drops unknown
    # fields silently -- so the form sent them, the API discarded them, and the
    # runner used its defaults. The controls looked broken because the value
    # never arrived. This mirror is the cost of keeping the engine importable
    # without FastAPI; forgetting it is the failure mode.
    task: str = Field(
        "baseline",
        description=(
            "Named objective from adk.scenarios.tasks — reweights the four "
            "native reward terms. Two runs with different tasks are not two "
            "points on one experiment; the agent is being paid differently."
        ),
    )
    opponent_policy: str | None = Field(
        None,
        description=(
            "Opponent ability policy. Null keeps the Gym's shipped "
            "'first_legal', which is the only one of four measured to land "
            "damage — 'highest_slot' and 'random_slot' get abilities accepted "
            "16 and 36 times and deal 0.0."
        ),
    )

    microticks: int = Field(
        1,
        ge=1,
        le=4,
        description=(
            "Engine ticks advanced per policy decision (1-4). The only real "
            "tick-rate knob in the stack — nothing throttles the simulation "
            "and TICKS_PER_SECOND is display-only. Higher advances more game "
            "time per env step; the agent then acts every N ticks rather than "
            "every tick (33ms at 1, 133ms at 4). Must be set here, not through "
            "`parameters`: the scene and the AgentSpec both carry it."
        ),
    )

    minigame: str | None = Field(
        None,
        description=(
            "Named minigame from adk.scenarios.minigames, added ON TOP of the "
            "native reward. Where `task` reweights the four terms the "
            "environment already computes, this pays for what they cannot "
            "express — distance closed, a route followed, a hit that landed, a "
            "guard that met an actual attack. Additive, so the native terms "
            "stay intact underneath and a shaped run stays comparable to an "
            "unshaped one on those alone. See /api/options for the list."
        ),
    )
    minigame_weight: float = Field(
        1.0,
        description="Scales the minigame's declared weight. 1.0 leaves it alone.",
    )
    minigame_options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Constructor arguments for the minigame — `goal` for reach, `route` "
            "for checkpoint, per-game rates for the rest. A game that needs one "
            "and is not given it is a 400, not a rollout paying a silent "
            "constant."
        ),
    )

    policy: str = Field(
        "pulse_ability", description="One of console.core.execution.runner.POLICY_KINDS."
    )
    slot: int = Field(1, ge=0, le=16, description="Ability slot to request.")
    period: int = Field(12, ge=1, le=512, description="Ticks between ability requests.")

    decision_period: int = Field(
        1,
        ge=1,
        le=128,
        description=(
            "Ticks between fresh decisions; the action is held in between. "
            "1 = every tick (33 ms at 30 TPS, superhuman). 6-8 is roughly "
            "human reaction latency. See docs/DECISION-RATE.md."
        ),
    )

    ticks: int = Field(
        256,
        ge=1,
        le=2048,
        description=(
            "How long the ROLLOUT runs, in ticks (30 = 1s). This was described "
            "as episode length, which it is not: it is how much you watch, not "
            "when the world starts over. See `episode_ticks` for that."
        ),
    )
    episode_ticks: int | None = Field(
        900,
        ge=1,
        le=100_000,
        description=(
            "Cap on one EPISODE, in engine ticks (30 = 1s; 900 = 30s). The "
            "environment ends an episode only on death or a world error, so "
            "without a cap an episode has no upper bound -- measured, a "
            "256-step update completed 0-3 of them and mean_episode_return "
            "reported 0.000 on most updates because it had nothing to average. "
            "Published as `truncated`, never `terminated`, so a learner can "
            "bootstrap through a clock expiry but not through a death. Set "
            "null for the environment's own rule."
        ),
    )
    seed: int = Field(
        0, ge=0, description="PRNG seed. Sweep it — single runs are noisy."
    )
    agent_max_health: float = Field(105.0, gt=0)
    target_max_health: float = Field(61.0, gt=0)
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Any of the 116 CombatParams fields, by name.",
    )


class TrainRequest(RunRequest):
    """One training configuration. Unknown knobs fail before a job starts."""

    model_config = ConfigDict(extra="forbid")

    training_stage: str = Field(
        "pursuit_tracking",
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Registered composable training stage.",
    )
    training_preset: str = Field(
        "large",
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stage-owned defaults; explicit fields override them.",
    )
    runner: Literal["wsl", "local"] = Field(
        "wsl", description="WSL selects CUDA JAX; local is explicit CPU fallback."
    )
    source_policy: str | None = Field(
        None, description="Initial policy checkpoint; the stage may declare a default."
    )
    world_artifact_seed: int | None = Field(
        None,
        ge=0,
        description=(
            "Saved train-split world artifact selected by its native generation "
            "seed. It is materialized before JIT and never regenerated in a scan. "
            "Pins one world, so leave it unset when world_count is above 1."
        ),
    )
    world_count: int = Field(
        1,
        ge=1,
        le=256,
        description=(
            "Saved train-split worlds held resident, with the environments split "
            "across them. 1 trains every lane on one map, which a policy can "
            "memorise; above 1, selection_key chooses which worlds."
        ),
    )
    mixed_terrain_worlds: bool = Field(
        False,
        description=(
            "Compose the resident worlds as half flattest-first and half by the "
            "ordinary digest rank, instead of taking all of them from "
            "selection_key. The key ranks the split rather than filtering it, so "
            "a deliberate mix is not reachable through it at any value. Needs "
            "world_count above 1."
        ),
    )
    # NO `world_library` FIELD, deliberately. Libraries are discovered and
    # listed at GET /api/worlds, but the training runtime still resolves its
    # library from a module constant, so a request field would be accepted and
    # then do nothing -- the exact "accepts what it never forwards" defect the
    # schema test exists to catch. Add it in the same change that teaches the
    # run config to honour it, not before.
    world_order: Literal["digest", "flattest", "mixed"] = Field(
        "digest",
        description=(
            "How to order the library before taking world_count worlds. "
            "'digest' is the loader's own keyed ranking and is indifferent to "
            "terrain. 'flattest' takes the most open, walkable worlds first, "
            "for an agent still learning to move. 'mixed' interleaves the two. "
            "Needs world_count above 1."
        ),
    )
    flattest_worlds: bool = Field(
        False,
        description=(
            "Deprecated alias for world_order='flattest'. Kept so an existing "
            "launch payload keeps working; prefer world_order."
        ),
    )
    environment_diversity: bool = Field(
        False,
        description=(
            "Draw a separate agent and target spawn for every environment inside "
            "its own world, still inside target_separation_range and requiring "
            "initial line of sight. Off means all lanes share one spawn pair and "
            "the batch is byte-identical."
        ),
    )
    selection_key: int | None = Field(
        None,
        description=(
            "Which saved worlds are resident when world_count is above 1. "
            "Above one world the resolver returns this directly and never "
            "consults seed, so two runs that differ only in seed train on the "
            "identical world subset and spawn pool -- vary this to rotate the "
            "subset between runs. Null keeps the run-config default."
        ),
    )
    node_seed: int | None = Field(
        None,
        description=(
            "Picks spawn nodes from each world's authored traversal graph. "
            "Null keeps the run-config default; vary alongside selection_key "
            "so a rotated world subset also gets fresh spawn nodes."
        ),
    )
    sensor_range: float | None = Field(
        64.0,
        gt=0.0,
        description=(
            "Actor perception range in blocks, default 64 (two 32-block "
            "chunks). Pass null for the authored 16.0, which is the fidelity "
            "value. target_separation_range must fit inside it, and the "
            "observed-distance channels are normalised by it, so a checkpoint "
            "trained at one range reads distance differently at another."
        ),
    )
    evaluation_steps: int = Field(
        512,
        ge=1,
        le=512,
        description="Fixed evaluation and episode horizon, capped at 512 ticks.",
    )
    evaluation_updates: list[int] = Field(
        default_factory=lambda: [0, 8, 16, 32, 64],
        description="Sorted training updates evaluated with common random numbers.",
    )
    memory_reset_updates: list[int] | None = Field(
        None,
        description=(
            "Evaluation milestones that also run the recurrent-memory ablation; "
            "disabled by default because each 512-tick ablation adds a cold compile."
        ),
    )
    archive_lanes: int = Field(
        8,
        ge=1,
        le=16,
        description="Representative evaluation lanes archived at every milestone.",
    )
    target_controller: Literal[
        "scripted_flee_weave",
        "scripted_flee_adaptive",
        "scripted_flee_mixed",
        "scripted_flee_standoff",
        "scripted_flee_standoff_mixed",
        "frozen_source_policy",
    ] = Field(
        "scripted_flee_weave",
        description=(
            "Physical opponent controller used by the pursuit lesson. "
            "'scripted_flee_weave' retreats on the skill row, which travels at "
            "run speed and has no slower setting. 'scripted_flee_adaptive' "
            "flees on the gait head at a walk/sprint duty cycle and turns away "
            "from the arena edge. 'scripted_flee_mixed' alternates the two by "
            "lane, so one run measures the learner against both under an "
            "identical policy and spawn distribution. "
            "'scripted_flee_standoff' does not flee: it holds a standoff band "
            "and circles the learner inside it, so the bearing keeps moving "
            "while the range does not and the learner has to track rather than "
            "run down a straight line. 'scripted_flee_standoff_mixed' "
            "alternates fleeing and standoff by lane. Every mode except "
            "'scripted_flee_weave' needs `world_design`, which is what declares "
            "an arena bound."
        ),
    )
    objective: Literal["pursue", "evade"] = Field(
        "pursue",
        description=(
            "Which side of the duel the learner is paid for. 'pursue' is the "
            "catch objective. 'evade' pays the learner for staying near the "
            "opponent, surviving to the horizon and manoeuvring, and charges it "
            "for being caught -- the other half of a recursive pair. 'evade' "
            "requires target_controller 'frozen_source_policy' so the opponent "
            "actually pursues."
        ),
    )
    learner_stamina_fraction: float = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Share of the full stamina bar the learner starts each episode "
            "with. The bar belongs to the agent actor, which is the learner, so "
            "below 1.0 this handicaps exactly the side being trained. Note it "
            "sets the STARTING charge, not the ceiling: regeneration still "
            "clips to the engine maximum, so the bar refills over an episode."
        ),
    )
    learner_stamina_regen_scale: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Multiplier on the authored locomotion-stamina regeneration rate "
            "for the learner. 1.0 is the shipped Hytale rate, so the default "
            "reproduces the game exactly. Below 1.0 a spent bar refills slower, "
            "which makes a stamina handicap last the whole episode rather than "
            "only its first sprint."
        ),
    )
    evader_terminal_reward: float = Field(
        1000.0,
        gt=0.0,
        description=(
            "Magnitude of the evader's survive/caught terminals. Against the "
            "per-tick contact reward this sets how terminal-dominated the "
            "advantages are; the default is roughly 50:1 over a 100-tick "
            "episode. Must exceed a full episode of contact income."
        ),
    )
    evader_contact_reward: float = Field(
        0.20,
        ge=0.0,
        description=(
            "Per-tick reward at the peak of the evader's standoff band."
        ),
    )
    evader_travel_reward: float = Field(
        0.10,
        ge=0.0,
        description=(
            "Paid per block of HORIZONTAL ground the evader covers. Vertical "
            "motion earns nothing, which is deliberate: a policy that hops on "
            "the spot accumulates 3D path length while going nowhere. Capped "
            "below contact_reward at sprint speed so travelling can never "
            "out-earn holding the standoff band."
        ),
    )
    evader_strafe_reward: float = Field(
        0.12,
        ge=0.0,
        description=(
            "Paid per block of the step that runs ACROSS the bearing to the "
            "pursuer. Pays for circling the standoff and pays a straight-line "
            "run almost nothing, which is correct while holding a band and "
            "exactly wrong on a learn-to-move stage. Zero it there."
        ),
    )
    evader_sprint_reward: float = Field(
        0.005,
        ge=0.0,
        description=(
            "Paid per tick spent at sprint speed, on top of the per-block "
            "travel rate. Sprint is forward-only in the gym, so this only pays "
            "an evader that has already turned its body onto its travel "
            "direction and committed to it."
        ),
    )
    evader_turn_reward: float = Field(
        0.02,
        ge=0.0,
        description=(
            "Paid per tick the body swings past the deadband. Seasons a "
            "standoff, but on a learn-to-move stage it pays the spinning that "
            "stops the evader ever holding a heading. Zero it there."
        ),
    )
    evader_airborne_idle_cost: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Charged per tick the evader is airborne having covered no ground. "
            "Pair it with evader_allow_jump: with the head closed the cost can "
            "never be charged and is dead weight, and with the head open a hop "
            "in place is the pathology it exists to price."
        ),
    )
    evader_allow_jump: bool = Field(
        False,
        description=(
            "Open the jump head for BOTH actors. Off on a plane: a jump buys "
            "nothing there and an airborne actor cannot accelerate -- an earlier "
            "lineage stayed airborne on 92% of ticks and covered 2.9 blocks in a "
            "whole episode. Turn it on for a stage with terrain worth jumping "
            "over, and restore airborne_idle_cost alongside it."
        ),
    )
    evader_alive_reward: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Paid on every tick the evader is still alive, so a longer episode "
            "scores higher. evader_terminal_reward pays once at the horizon and "
            "is therefore all-or-nothing -- surviving 250 of 256 ticks and "
            "surviving 6 score the same. This is what expresses 'the longer it "
            "stays alive the better', and it gives a policy that never reaches "
            "the horizon a gradient it can climb."
        ),
    )
    evader_forward_travel_reward: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Paid per block of the step that runs ALONG the body's facing, "
            "clamped non-negative and to the step itself. Movement is "
            "compass-driven, so an actor can travel any direction without "
            "turning -- but authored speed only rewards pointing where you go: "
            "sprint-forward 7.00 against run-strafe 4.40 blocks/s, and sprint "
            "additionally requires forward-dominant travel. travel_reward pays "
            "a strafe and a sprint identically, so it cannot express this."
        ),
    )
    evader_separation_reward: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Paid per block the evader's OWN motion adds to the gap, "
            "normalized by sprint speed and symmetrically clipped, so closing "
            "costs what opening pays. Attributed to the evader rather than to "
            "the raw gap, so a pursuer that wanders off earns it nothing. This "
            "is the only evader term that distinguishes fleeing from circling "
            "-- travel_reward pays ground covered in any direction and "
            "body_away_reward pays where the body points, not where it ends "
            "up. Turn it on for a long-range evasion stage and back off "
            "wherever the standoff band is the objective. Zero disables it."
        ),
    )
    evader_body_away_reward: float = Field(
        0.04,
        ge=0.0,
        description=(
            "Paid for pointing the BODY away from the pursuer: 1.0 at exactly "
            "180 degrees off the bearing, 0.0 staring straight at it. Movement "
            "is body-relative, so an evader facing its pursuer can only "
            "retreat at the reduced backwards multiplier -- this is what makes "
            "fleeing fast, while head yaw stays free to look back. Scaled by "
            "the standoff band as well, so it is only earnable while holding "
            "contact range and a policy that turns and runs earns none of it. "
            "Must stay below contact_reward when the band leads. Zero disables "
            "the term."
        ),
    )
    evader_caught_penalty: float | None = Field(
        None,
        # ge, not gt: 0.0 is a legal and deliberate setting on a stage with no
        # proximity band, where being caught is punished by ENDING the episode
        # and with it the per-tick alive income. gt rejected that at the API
        # boundary with the run config and contract both willing to accept it.
        ge=0.0,
        description=(
            "Subtracted when the evader is caught. Unset, it tracks "
            "evader_terminal_reward, which is the historical behaviour. Set it "
            "separately for a curriculum that RETIRES the survival bonus: the "
            "two used to be one number, so lowering the terminal to hand the "
            "objective over to proximity also lowered the catch penalty, at "
            "the limit leaving an evader that pays nothing for being caught. "
            "Being caught is bad at every rung; only the bonus is scheduled."
        ),
    )
    evader_band_leads: bool = Field(
        True,
        description=(
            "Whether the standoff band is the evader's OBJECTIVE or merely a "
            "nudge. True is the original contract: proximity is what it is for, "
            "so motion, posture and turning each stay below contact_reward. "
            "False is the locomotion-first phase -- keep contact_reward small "
            "and let the per-tick motion terms out-earn it, so the band only "
            "pulls the evader back toward the fight instead of letting it run "
            "to the arena wall and park. False also ungates the body-away term, "
            "because gating posture on the band makes closing the distance the "
            "precondition for being paid to face away."
        ),
    )
    opponent_checkpoint: str | None = Field(
        None,
        description=(
            "A separate frozen checkpoint for the opponent slot. Unset, the "
            "opponent is a copy of `source_policy`, which is what a "
            "single-policy run wants. Set, it is a different policy -- which "
            "is what a recursive pursuer/evader pair wants, each generation "
            "trained against the other's last promoted checkpoint. Must match "
            "the learner's observation and action widths. Only meaningful with "
            "a policy-driven target controller."
        ),
    )
    world_design: str | None = Field(
        None,
        description=(
            "Train on an authored design from `worlds.designs` instead of a "
            "captured Region. The design's seed-independent recipe digest "
            "becomes the run's world identity, so re-rolling a preview seed "
            "cannot reach terrain nobody authored. Skips the Region artifact "
            "hash, so there is no multi-minute scene build."
        ),
    )
    arena_radius: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Design worlds only: half-width of the arena in blocks. Bounds an "
            "otherwise unbounded plane and is what the adaptive evader steers "
            "away from. Must exceed the largest target separation."
        ),
    )
    evader_gait_mix: tuple[float, float, float] | None = Field(
        None,
        description=(
            "(walk, run, sprint) shares of each adaptive-evader duty cycle; "
            "must sum to 1. Measured speeds are 0.063 / 0.194 / 0.238 blocks "
            "per tick, so the default (0.1, 0.6, 0.3) averages 0.194 -- the "
            "same pace `scripted_flee_weave` travels at, making the two "
            "evaders like-for-like on speed and different only in cadence."
        ),
    )
    evader_gait_period_ticks: int | None = Field(
        None,
        ge=1,
        description=(
            "Length of one adaptive walk/sprint duty cycle. Lanes are offset "
            "within it so the batch does not change gait in lockstep."
        ),
    )
    evader_standoff_band: tuple[float, float] | None = Field(
        None,
        description=(
            "Standoff controllers only: (inner, outer) blocks of the band the "
            "evader holds. Inside the inner edge it backs off, beyond the outer "
            "edge it closes, and between them it circles. Must satisfy "
            "0 < inner < outer, and outer must fit inside `arena_radius`."
        ),
    )
    evader_standoff_strafe_ticks: int | None = Field(
        None,
        ge=1,
        description=(
            "Ticks the standoff evader circles one way before reversing. Lanes "
            "are offset within it so the batch does not all turn together."
        ),
    )
    target_separation_range: tuple[float, float] = Field(
        (10.0, 14.0),
        description="Saved-map horizontal reset-distance band in blocks.",
    )
    minimum_baseline_visible_fraction: float = Field(
        0.50,
        ge=0.0,
        le=1.0,
        description="Minimum physical line-of-sight fraction for policy selection.",
    )
    maximum_approximate_kl: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Joint PPO approximate-KL ceiling across all action heads. Omit to "
            "derive it from the live head count, which is the only stable "
            "setting: the metric is a sum over heads, so a fixed number "
            "silently tightens whenever the action surface grows. The old 0.03 "
            "default was calibrated at ten heads and stopped healthy runs at "
            "twelve."
        ),
    )
    stage_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides accepted by the selected environment stage contract.",
    )
    target_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides accepted by the stage's opponent-motion contract.",
    )

    agent_max_health: float = Field(
        105.0, gt=0, description="Maximum health assigned to the learning agent."
    )
    target_max_health: float = Field(
        61.0, gt=0, description="Maximum health assigned to the opponent."
    )
    updates: int = Field(10, ge=1, description="Number of PPO updates to run.")
    num_envs: int = Field(
        16, ge=1, description="Parallel environments used by each PPO update."
    )
    rollout_steps: int = Field(
        128, ge=1, description="Environment steps collected per PPO update."
    )
    update_epochs: int = Field(
        4, ge=1, description="Optimization passes over each rollout batch."
    )
    num_minibatches: int = Field(
        8, ge=1, description="Minibatches used for each optimization epoch."
    )
    encoder_size: int = Field(
        64, ge=1, description="Hidden width of the policy encoder."
    )
    recurrent_size: int = Field(
        128, ge=1, description="Hidden width of the recurrent policy state."
    )
    learning_rate: float = Field(3.0e-4, ge=0, description="Optimizer learning rate.")
    gamma: float = Field(0.99, ge=0, description="Reward discount factor.")
    gae_lambda: float = Field(
        0.95, ge=0, description="Generalized advantage estimation factor."
    )
    clip: float = Field(0.2, ge=0, description="Console alias for clip_epsilon.")
    clip_epsilon: float = Field(
        0.2, ge=0, description="PPO policy-ratio clipping radius."
    )
    value_coef: float = Field(
        0.5, ge=0, description="Console alias for value_coefficient."
    )
    value_coefficient: float = Field(
        0.5, ge=0, description="Weight of the value-function loss."
    )
    entropy_coef: float = Field(
        0.01, ge=0, description="Console alias for entropy_coefficient."
    )
    entropy_coefficient: float = Field(
        0.01, ge=0, description="Weight of the policy entropy bonus."
    )
    max_gradient_norm: float = Field(
        0.5, ge=0, description="Global gradient norm clipping threshold."
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_ppo_aliases(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for alias, canonical in (
            ("clip", "clip_epsilon"),
            ("value_coef", "value_coefficient"),
            ("entropy_coef", "entropy_coefficient"),
        ):
            if alias in data and canonical in data and data[alias] != data[canonical]:
                raise ValueError(f"{alias} and {canonical} must match")
            if alias in data:
                data[canonical] = data[alias]
            elif canonical in data:
                data[alias] = data[canonical]
        return data

    @model_validator(mode="after")
    def _validate_schedule(self):
        points = self.evaluation_updates
        if points != sorted(set(points)) or not points or points[0] != 0:
            raise ValueError("evaluation_updates must be sorted unique and start at 0")
        memory = self.memory_reset_updates
        if memory is not None and (
            memory != sorted(set(memory)) or not set(memory).issubset(points)
        ):
            raise ValueError("memory_reset_updates must be sorted evaluation updates")
        minimum, maximum = self.target_separation_range
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum <= 0.0
            # Equal bounds are a fixed spawn distance, which is a real setting:
            # a constant gap isolates closing skill from the spread. Only an
            # inverted range is wrong.
            or maximum < minimum
        ):
            raise ValueError("target separation range must be finite and ordered")
        return self


class WorldStatusRequest(BaseModel):
    """A keep/deprecate decision about one world.

    `state: null` clears the decision, which is not the same as keeping it:
    kept means judged and fine, cleared means nobody has looked.
    """

    model_config = ConfigDict(extra="forbid")

    state: Literal["kept", "deprecated"] | None = Field(
        None, description="kept, deprecated, or null to forget the decision."
    )
    reason: str = Field(
        "",
        max_length=500,
        description="Why, for whoever reads this list next.",
    )
    split: str = Field(
        "train",
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Which split the world belongs to.",
    )
