"""J1, the learned reactive JEV: a small network over B_t and E_t with no model.

Thesis (ADR-001 D2, COMMITTED). J1 is the primary JEV instance and the first
one deployed. It is "a learned reactive network over B_t and E_t, no model",
and its purpose is to measure what hierarchy alone buys. Given the shared
belief B_t and the executive state E_t, a feed-forward network chooses the
skill, whether to replace the goal in force, and the goal parameters. It has
no world model, no search, and no access to privileged state. Its structural
tests show record validity, shapes, formula values and mask legality. They say nothing about decision
quality (ADR-001 section 6.5).

Formulas. Notation: B lanes; F = BELIEF_FEATURE_COUNT (10); L =
BELIEF_LATENT_WIDTH (64); K = num_skills; G = MAX_GOALS (4); P = PARAM_WIDTH
(4); H = hidden width (default 128). Catalog index 0 is idle.

1. Input, ``j1_inputs``, width D = 2F + L + 5K + 28 (for G = P = 4). Source:
   this module's layout over the contract records SharedBeliefState,
   ExecutiveArrays and GoalPayload (contracts.py).

     x = [ where(known, features, 0) | known | latent |
           onehot_K(skill_index) over G slots | priority | params over G slots |
           goal_valid | log1p(budget) | log1p(version) | onehot_K(current_skill) ]

   Unknown features are zeroed so that an unpublished value cannot reach the
   network. ``feature_known`` carries the distinction. An empty slot has
   skill -1, which one-hot encodes to a zero block.

2. Network, ``apply_j1`` (ReLU trunk, three linear heads). Source: this
   module's design. ADR-001 D2 fixes only the inputs and the ManagedGoal output.

     h1        = relu(x W1 + b1)                 W1 [D, H]
     h2        = relu(h1 W2 + b2)                W2 [H, H]
     z_skill   = h2 W_skill + b_skill            [B, K]
     z_replace = h2 w_replace + b_replace        [B]
     p_head    = h2 W_params + b_params          [B, P]

3. Decision, ``NetworkJev.__call__``:

     A[b, k]     = [k = 0] or OR_g ( goal_valid[b, g] and skill_index[b, g] = k )
     k*_b        = argmax_k ( z_skill[b, k] if A[b, k] else -inf )
     q_b         = sigmoid(z_replace[b])
     terminate_b = [q_b > 0.5] and [k*_b != current_skill_b]
     slot_b      = the matching slot g with the highest priority (lowest g on a
                   tie), or -1 when no slot matches k*_b

   The goal is ``skill_index = k*``, ``params = p_head``,
   ``executive_version = E_t.version`` and ``goal_slot = slot``. Its control is
   skill k*'s row of the instance's ``SkillControlTable`` (``control_for``):
   the skill's contract owns it (SYSTEM_ONE.md section 8.12, item 1). Without a
   table J1 knows no catalogue, so every index gets the ``SkillContract``
   defaults: minimum commitment 0.5 s, maximum duration 10 s and emergency
   events {DAMAGE_RECEIVED, TARGET_DIED, NPC_STUCK}, the control J1 gave every
   goal until 2026-10-17. Source: ADR-001 section 2 names the three controls
   (minimum commitment, maximum duration, emergency-interrupt mask).
   REPLACE_THRESHOLD = 0.5 is this module's choice; ADR-001 sets none of these
   values. ADR-001 section 8, P2 names DAMAGE_RECEIVED only as an evade
   interrupt in a composition task.

4. Initialization, ``init_j1``: He normal for W1 and W2 (std sqrt(2 / fan_in);
   He et al., see Citations); for W_skill and w_replace, fan-in scaling with
   std sqrt(1 / H), the LeCun-normal convention, not cited here; zeros for every
   bias and for W_params. Source: the He citation for the first part, this
   module's choice for the rest. The zero params head is deliberate. The
   behaviour-cloning dataset has no target parameters, so this head is not
   trained and emits zeros until a dataset supplies them.

Citations.
- ADR-001 D2 (``agents/design/ADR-001-three-level-agent.md``): the JEV contract,
  the instance table with the J1 row, and the rule that a missed budget leaves
  the previous goal in force.
- ADR-001 section 8, P2 "Exit for J1": J1 must beat J0 on the same tasks with
  intervals, and must beat its own shuffled-goal control
  (``experimental/hierarchical_planner/DESIGN.md`` section 6).
- Ross, Gordon, Bagnell, *DAgger*, AISTATS 2011,
  https://proceedings.mlr.press/v15/ross11a (ADR-001 section 11, [24], verified
  there). Behaviour cloning on recorded expert decisions compounds errors once
  the learner leaves the expert's states. DAgger is the remedy and is not
  implemented here.
- He, Zhang, Ren, Sun, *Delving Deep into Rectifiers: Surpassing Human-Level
  Performance on ImageNet Classification*, arXiv:1502.01852,
  https://arxiv.org/abs/1502.01852 (initialization). Not in ADR-001 section 11.
  Title, authors and identifier checked on the arXiv abstract page by WebFetch
  on 2026-10-16.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.components import GoalDecision
from adk.architecture.decision_context import DecisionContext, DecisionEvent
from adk.architecture.hierarchy import GoalRuntimeContext
from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    BELIEF_LATENT_WIDTH,
    MAX_GOALS,
    PARAM_WIDTH,
    ExecutiveArrays,
    GoalPayload,
    JevMode,
    SharedBeliefState,
    SkillContract,
    SkillControlTable,
    _require_control_table,
    control_for,
    event_mask,
    make_goal,
)

JEV_NAME: str = "j1"
# Catalog index of the idle skill. The contract's idle_goal uses index 0.
IDLE_SKILL_INDEX: int = 0
# Columns of ExecutiveArrays.budget: [jev_latency_ms, search_expansions,
# executive_calls_per_minute].
BUDGET_WIDTH: int = 3
# The control J1 gave every goal until 2026-10-17, equal to the SkillContract
# defaults. No longer applied: a goal's control is the chosen skill's row of the
# instance's SkillControlTable. Kept because tests and documents cite them.
COMMITMENT_SECONDS: float = 0.5
MAX_DURATION_SECONDS: float = 10.0
EMERGENCY_EVENTS: tuple[DecisionEvent, ...] = (
    DecisionEvent.DAMAGE_RECEIVED,
    DecisionEvent.TARGET_DIED,
    DecisionEvent.NPC_STUCK,
)
REPLACE_THRESHOLD: float = 0.5
DEFAULT_HIDDEN: int = 128


class J1Params(NamedTuple):
    """Parameters of J1, all float32 (see the module formulas).

    Shapes, with D = ``input_width``, H = hidden width, K = num_skills and
    P = PARAM_WIDTH:
        w1: [D, H]. b1: [H].
        w2: [H, H]. b2: [H].
        w_skill: [H, K]. b_skill: [K].
        w_replace: [H]. b_replace: [] (a scalar).
        w_params: [H, P]. b_params: [P].
    """

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w_skill: jax.Array
    b_skill: jax.Array
    w_replace: jax.Array
    b_replace: jax.Array
    w_params: jax.Array
    b_params: jax.Array


def _count(value: Any, name: str) -> int:
    """Return ``value`` as an integer of at least 1. Bools raise TypeError."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return int(value)


def _require_shape(value: Any, expected: tuple[int, ...], name: str) -> None:
    """Raise ValueError unless ``value`` has exactly the ``expected`` shape."""

    shape = tuple(jnp.shape(value))
    if shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {shape}")


def input_width(
    num_skills: int,
    *,
    feature_count: int = BELIEF_FEATURE_COUNT,
    latent_width: int = BELIEF_LATENT_WIDTH,
) -> int:
    """Return D, the width of the vector that :func:`j1_inputs` builds.

    Formula: D = 2F + L + G*K + G + G*P + G + 3 + 1 + K. With G = 4 and P = 4
    this is 2F + L + 5K + 28. With the contract sizes (F = 10, L = 64) and
    K = 4 it is 132.

    Args:
        num_skills: Catalog size K, an integer of at least 1.
        feature_count: F, the number of belief features.
        latent_width: L, the width of the belief latent.

    Returns:
        The input width D, a positive int.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is less than 1.
    """

    skills = _count(num_skills, "num_skills")
    features = _count(feature_count, "feature_count")
    latent = _count(latent_width, "latent_width")
    slot_block = MAX_GOALS * (skills + 1 + PARAM_WIDTH + 1)
    return 2 * features + latent + slot_block + BUDGET_WIDTH + 1 + skills


def init_j1(
    key: jax.Array,
    *,
    num_skills: int,
    feature_count: int = BELIEF_FEATURE_COUNT,
    latent_width: int = BELIEF_LATENT_WIDTH,
    hidden: int = DEFAULT_HIDDEN,
) -> J1Params:
    """Initialize J1 parameters (formulas in the module docstring, section 4).

    Args:
        key: A JAX PRNG key.
        num_skills: Catalog size K, an integer of at least 1.
        feature_count: F, the number of belief features.
        latent_width: L, the width of the belief latent.
        hidden: H, the width of both hidden layers.

    Returns:
        J1Params with the shapes listed on :class:`J1Params`.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is less than 1.
    """

    skills = _count(num_skills, "num_skills")
    width_h = _count(hidden, "hidden")
    width = input_width(
        skills, feature_count=feature_count, latent_width=latent_width
    )
    key_w1, key_w2, key_skill, key_replace = jax.random.split(key, 4)
    f32 = jnp.float32
    return J1Params(
        w1=jax.random.normal(key_w1, (width, width_h), f32) * jnp.sqrt(2.0 / width),
        b1=jnp.zeros((width_h,), f32),
        w2=jax.random.normal(key_w2, (width_h, width_h), f32)
        * jnp.sqrt(2.0 / width_h),
        b2=jnp.zeros((width_h,), f32),
        w_skill=jax.random.normal(key_skill, (width_h, skills), f32)
        * jnp.sqrt(1.0 / width_h),
        b_skill=jnp.zeros((skills,), f32),
        w_replace=jax.random.normal(key_replace, (width_h,), f32)
        * jnp.sqrt(1.0 / width_h),
        b_replace=jnp.zeros((), f32),
        w_params=jnp.zeros((width_h, PARAM_WIDTH), f32),
        b_params=jnp.zeros((PARAM_WIDTH,), f32),
    )


def j1_allowed_skills(executive: ExecutiveArrays, *, num_skills: int) -> jax.Array:
    """Return the skills that a J1 decision may pick (the mask A of section 3).

    Skill k is allowed when it is idle (k = 0), or when some valid executive
    goal names it. An out-of-range index is allowed for no slot, because
    ``jax.nn.one_hot`` returns a zero row for it. Only idle then remains
    possible, so the decision fails safe.

    Args:
        executive: ExecutiveArrays with ``skill_index`` int32[B, G] and
            ``goal_valid`` bool[B, G].
        num_skills: Catalog size K, an integer of at least 1.

    Returns:
        bool[B, K], True where the skill may be chosen.

    Raises:
        TypeError: If ``num_skills`` is not an integer.
        ValueError: If ``num_skills`` is less than 1, or the shapes are wrong.
    """

    skills = _count(num_skills, "num_skills")
    skill_index = jnp.asarray(executive.skill_index, dtype=jnp.int32)
    goal_valid = jnp.asarray(executive.goal_valid, dtype=jnp.bool_)
    _require_shape(goal_valid, jnp.shape(skill_index), "executive.goal_valid")
    if skill_index.ndim != 2 or skill_index.shape[1] != MAX_GOALS:
        raise ValueError(
            f"executive.skill_index must have shape (B, {MAX_GOALS}), "
            f"got {skill_index.shape}"
        )
    hits = jax.nn.one_hot(skill_index, skills, dtype=jnp.bool_)
    hits = hits & goal_valid[..., None]
    allowed = jnp.any(hits, axis=1)
    return allowed.at[:, IDLE_SKILL_INDEX].set(True)


def j1_inputs(
    belief: SharedBeliefState,
    executive: ExecutiveArrays,
    current_goal: GoalPayload,
    *,
    num_skills: int,
) -> jax.Array:
    """Build the J1 input matrix x (formula 1 of the module docstring).

    ``num_skills`` is keyword-only because the one-hot width depends on the
    catalog size, which none of the three records carries.

    Args:
        belief: SharedBeliefState with ``features`` float32[B, F],
            ``feature_known`` bool[B, F] and ``latent`` float32[B, L].
        executive: ExecutiveArrays for the same B lanes.
        current_goal: GoalPayload; only ``skill_index`` int32[B] is read.
        num_skills: Catalog size K, an integer of at least 1.

    Returns:
        float32[B, D], with D = ``input_width(K, F, L)``.

    Raises:
        TypeError: If ``num_skills`` is not an integer.
        ValueError: If ``num_skills`` is less than 1, any shape disagrees
            with B lanes, or the belief is not the contract size: F must equal
            BELIEF_FEATURE_COUNT and L must equal BELIEF_LATENT_WIDTH. The
            ``feature_count`` and ``latent_width`` arguments of ``input_width``
            and ``init_j1`` must therefore keep their defaults for a belief to
            be accepted.
    """

    skills = _count(num_skills, "num_skills")
    features = jnp.asarray(belief.features, dtype=jnp.float32)
    if features.ndim != 2:
        raise ValueError(
            f"belief.features must have shape (B, F), got {features.shape}"
        )
    lanes, feature_count = features.shape
    if feature_count != BELIEF_FEATURE_COUNT:
        raise ValueError(
            f"belief.features must have {BELIEF_FEATURE_COUNT} columns, "
            f"got {feature_count}"
        )
    known = jnp.asarray(belief.feature_known, dtype=jnp.bool_)
    latent = jnp.asarray(belief.latent, dtype=jnp.float32)
    _require_shape(known, (lanes, feature_count), "belief.feature_known")
    if latent.ndim != 2 or latent.shape[0] != lanes:
        raise ValueError(f"belief.latent must have shape (B, L), got {latent.shape}")
    if latent.shape[1] != BELIEF_LATENT_WIDTH:
        raise ValueError(
            f"belief.latent must have {BELIEF_LATENT_WIDTH} columns, "
            f"got {latent.shape[1]}"
        )
    skill_index = jnp.asarray(executive.skill_index, dtype=jnp.int32)
    priority = jnp.asarray(executive.priority, dtype=jnp.float32)
    params = jnp.asarray(executive.params, dtype=jnp.float32)
    goal_valid = jnp.asarray(executive.goal_valid, dtype=jnp.bool_)
    budget = jnp.asarray(executive.budget, dtype=jnp.float32)
    version = jnp.asarray(executive.version, dtype=jnp.float32)
    current = jnp.asarray(current_goal.skill_index, dtype=jnp.int32)
    _require_shape(skill_index, (lanes, MAX_GOALS), "executive.skill_index")
    _require_shape(priority, (lanes, MAX_GOALS), "executive.priority")
    _require_shape(params, (lanes, MAX_GOALS, PARAM_WIDTH), "executive.params")
    _require_shape(goal_valid, (lanes, MAX_GOALS), "executive.goal_valid")
    _require_shape(budget, (lanes, BUDGET_WIDTH), "executive.budget")
    _require_shape(version, (lanes,), "executive.version")
    _require_shape(current, (lanes,), "current_goal.skill_index")
    pieces = [
        jnp.where(known, features, 0.0),
        known.astype(jnp.float32),
        latent,
        jax.nn.one_hot(skill_index, skills, dtype=jnp.float32).reshape(
            lanes, MAX_GOALS * skills
        ),
        priority,
        params.reshape(lanes, MAX_GOALS * PARAM_WIDTH),
        goal_valid.astype(jnp.float32),
        jnp.log1p(budget),
        jnp.log1p(version)[:, None],
        jax.nn.one_hot(current, skills, dtype=jnp.float32),
    ]
    return jnp.concatenate(pieces, axis=-1)


def apply_j1(params: J1Params, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the J1 network (formula 2 of the module docstring).

    Args:
        params: J1Params.
        x: float32[B, D], the output of :func:`j1_inputs`.

    Returns:
        ``(skill_logits, replace_logit, params_head)`` with shapes
        float32[B, K], float32[B] and float32[B, P].

    Raises:
        ValueError: If ``x`` is not 2-D or its width is not D.
    """

    inputs = jnp.asarray(x, dtype=jnp.float32)
    if inputs.ndim != 2 or inputs.shape[1] != params.w1.shape[0]:
        raise ValueError(
            f"x must have shape (B, {params.w1.shape[0]}), got {inputs.shape}"
        )
    hidden_one = jax.nn.relu(inputs @ params.w1 + params.b1)
    hidden_two = jax.nn.relu(hidden_one @ params.w2 + params.b2)
    skill_logits = hidden_two @ params.w_skill + params.b_skill
    replace_logit = hidden_two @ params.w_replace + params.b_replace
    params_head = hidden_two @ params.w_params + params.b_params
    return skill_logits, replace_logit, params_head


def _check_params(params: J1Params) -> tuple[int, int]:
    """Return (hidden width H, catalog size K) after checking every shape."""

    if not isinstance(params, J1Params):
        raise TypeError(f"params must be a J1Params, got {type(params).__name__}")
    for name, value in zip(J1Params._fields, params, strict=True):
        if jnp.asarray(value).dtype != jnp.float32:
            raise TypeError(f"J1Params.{name} must be float32")
    if jnp.ndim(params.w1) != 2:
        raise ValueError(
            f"J1Params.w1 must have shape (D, H), got {jnp.shape(params.w1)}"
        )
    width_h = int(jnp.shape(params.w1)[1])
    skills = int(jnp.shape(params.w_skill)[1]) if jnp.ndim(params.w_skill) == 2 else -1
    if skills < 1:
        raise ValueError("J1Params.w_skill must have shape (H, K) with K >= 1")
    expected = {
        "w1": (jnp.shape(params.w1)[0], width_h),
        "b1": (width_h,),
        "w2": (width_h, width_h),
        "b2": (width_h,),
        "w_skill": (width_h, skills),
        "b_skill": (skills,),
        "w_replace": (width_h,),
        "b_replace": (),
        "w_params": (width_h, PARAM_WIDTH),
        "b_params": (PARAM_WIDTH,),
    }
    for name, shape in expected.items():
        _require_shape(getattr(params, name), shape, f"J1Params.{name}")
    return width_h, skills


class NetworkJev:
    """J1 as a JEV instance that runs inside the jitted hierarchy step.

    Implements ``contracts.JevInstance`` with ``name`` = "j1" and ``mode`` =
    ``JevMode.IN_GRAPH``. It reads B_t and E_t only and ignores
    ``actor_input``, so it never sees privileged state (ADR-001 section 2).

    The hierarchy core calls the manager without an ``executive`` keyword
    (``adk/architecture/hierarchy.py``). A runner must bind the executive
    before the core sees the instance. This class does not do that.

    The core also requires the manager's ``diagnostics`` to have the same
    structure and leaf shapes as the state it holds between steps. Use
    :meth:`diagnostics_template` to initialize them.

    Attributes:
        name: "j1".
        mode: ``JevMode.IN_GRAPH``.
        params: The J1Params in use.
        num_skills: K, read from ``params.w_skill``.
    """

    name: str = JEV_NAME
    mode: JevMode = JevMode.IN_GRAPH

    def __init__(
        self, params: J1Params, *, control: SkillControlTable | None = None
    ) -> None:
        """Bind parameters after checking every shape against the others.

        Args:
            params: J1Params, as produced by :func:`init_j1` or loaded from a
                run.
            control: Keyword only. The goal control of each of the K skills, in
                catalogue order (``SkillRegistry.control_table()``). Default: K
                rows of the ``SkillContract`` defaults, because J1 knows no ids.

        Raises:
            TypeError: If ``params`` is not a J1Params or a leaf is not float32,
                or ``control`` is not a SkillControlTable.
            ValueError: If any shape is inconsistent with the others, or
                ``control`` does not have K rows.
        """

        _, skills = _check_params(params)
        self.params = params
        self.num_skills = skills
        self._control = _require_control_table(
            _contract_default_table(skills) if control is None else control, skills
        )

    def initial_state(self, batch: int) -> jax.Array:
        """Return the reset instance state: float32[B, 0].

        J1 carries no state. The zero-width leaf keeps the leading lane axis
        that the contract requires.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            float32[batch, 0].
        """

        return jnp.zeros((_count(batch, "batch"), 0), dtype=jnp.float32)

    def diagnostics_template(self, batch: int) -> dict[str, jax.Array]:
        """Return zero diagnostics with the structure that ``__call__`` returns.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            ``{"skill_logits": float32[B, K], "replace_prob": float32[B]}``.
        """

        lanes = _count(batch, "batch")
        return {
            "skill_logits": jnp.zeros((lanes, self.num_skills), dtype=jnp.float32),
            "replace_prob": jnp.zeros((lanes,), dtype=jnp.float32),
        }

    def __call__(
        self,
        state: Any,
        belief: SharedBeliefState,
        current_goal: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        *,
        decision_context: DecisionContext,
        goal_runtime: GoalRuntimeContext,
        executive: ExecutiveArrays,
    ) -> GoalDecision:
        """Choose the goal for one decision (section 3 of the module docstring).

        ``actor_input``, ``key``, ``decision_context`` and ``goal_runtime`` are
        accepted for the protocol and not used. J1 reads B_t and E_t only.

        Args:
            state: The instance state, returned unchanged. float32[B, 0].
            belief: SharedBeliefState for B lanes.
            current_goal: GoalPayload in force.
            actor_input: Accepted and ignored.
            key: Accepted and ignored. J1 is deterministic.
            decision_context: Accepted and ignored.
            goal_runtime: Accepted and ignored.
            executive: ExecutiveArrays for B lanes.

        Returns:
            GoalDecision with ``goal`` a ManagedGoal whose payload is a
            GoalPayload (skill_index int32[B], params float32[B, 4],
            executive_version int32[B], goal_slot int32[B]) and the chosen
            skill's row of the control table, ``terminate`` bool[B], and
            ``diagnostics`` as in :meth:`diagnostics_template`.
        """

        x = j1_inputs(belief, executive, current_goal, num_skills=self.num_skills)
        skill_logits, replace_logit, params_head = apply_j1(self.params, x)
        allowed = j1_allowed_skills(executive, num_skills=self.num_skills)
        masked = jnp.where(allowed, skill_logits, -jnp.inf)
        skill = jnp.argmax(masked, axis=-1).astype(jnp.int32)
        replace_prob = jax.nn.sigmoid(replace_logit)
        current = jnp.asarray(current_goal.skill_index, dtype=jnp.int32)
        terminate = (replace_prob > REPLACE_THRESHOLD) & (skill != current)
        minimum, maximum, mask = control_for(self._control, skill)
        goal = make_goal(
            skill,
            params_head,
            executive.version,
            _matching_slot(executive, skill),
            minimum_commitment_seconds=minimum,
            maximum_duration_seconds=maximum,
            emergency_interrupt_mask=mask,
        )
        return GoalDecision(
            state=state,
            goal=goal,
            terminate=terminate,
            diagnostics={"skill_logits": skill_logits, "replace_prob": replace_prob},
        )


def _contract_default_table(num_skills: int) -> SkillControlTable:
    """Return K rows of the SkillContract defaults: the control of a skill that declares none."""

    return SkillControlTable(
        minimum_commitment_seconds=jnp.full(
            (num_skills,), SkillContract.minimum_commitment_seconds, dtype=jnp.float32
        ),
        maximum_duration_seconds=jnp.full(
            (num_skills,), SkillContract.maximum_duration_seconds, dtype=jnp.float32
        ),
        emergency_interrupt_mask=event_mask(
            *(DecisionEvent[name] for name in SkillContract.emergency_events),
            batch=num_skills,
        ),
    )


def _matching_slot(executive: ExecutiveArrays, skill: jax.Array) -> jax.Array:
    """Return the E_t slot for each chosen skill, or -1 (section 3, slot_b).

    Among the valid slots that name the chosen skill, the one with the highest
    priority wins, and the lowest index wins a tie.
    """

    match = jnp.asarray(executive.goal_valid, dtype=jnp.bool_) & (
        jnp.asarray(executive.skill_index, dtype=jnp.int32) == skill[:, None]
    )
    priority = jnp.where(
        match, jnp.asarray(executive.priority, dtype=jnp.float32), -jnp.inf
    )
    best = jnp.argmax(priority, axis=-1).astype(jnp.int32)
    return jnp.where(jnp.any(match, axis=-1), best, jnp.int32(-1))


__all__ = [
    "BUDGET_WIDTH",
    "COMMITMENT_SECONDS",
    "DEFAULT_HIDDEN",
    "EMERGENCY_EVENTS",
    "IDLE_SKILL_INDEX",
    "JEV_NAME",
    "MAX_DURATION_SECONDS",
    "NetworkJev",
    "J1Params",
    "REPLACE_THRESHOLD",
    "apply_j1",
    "init_j1",
    "input_width",
    "j1_allowed_skills",
    "j1_inputs",
]
