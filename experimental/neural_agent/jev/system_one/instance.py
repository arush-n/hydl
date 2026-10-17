"""SystemOneJev: J-S1 as a host-mode JEV instance on two timescales (FORMULAS (M2), (M4), (M8), (M9), (M12)).

Slow path, off the tick (the only trunk calls)::

    construction:           E = catalogue_embeddings(trunk, skill_ids, descriptions)          once per catalogue
    set_executive_state:    h_E = trunk.encode([render_executive(skill_ids, descriptions, E_t)])[0]
                                                                         with the text path only
                            c_l = trunk.encode([l])[0] for each constraint label l not seen before
                            e_t = trunk.encode(new texts) for the goals' intent and when texts not seen
                                  before, one batched call                        heads with goal text (M12)
                            x   = data.slot_inputs(E_t, skill_ids, E, c, text_embeddings=e) with the structured encoder
                            x_E = encode_executive(params, x) = [g_E ; q_top]             (M9), compiled
                            cached by E_t.version: the same version again costs no trunk call
    set_shared_state:       the same, reading the scratchpad's ``executive`` section, whose goals carry
                            their words; set_executive_state reads a goal without words (zero text columns)

The label and text embeddings are cached for the instance's life (one catalogue), so with the structured
encoder alone (``text_path`` False) a new E_t whose labels and texts were seen before costs no trunk call.
The instance reads in the typed mode (e_skill in the slots, the allowed mask from the goals' skills); the
text-only mode of SYSTEM_ONE.md 8.11 is a training and evaluation mode of ``train_supervised``.

Fast path, every decision (``__call__``): no text and no trunk call. One compiled call::

    u_b       = belief_inputs(B_t)_b ;  m = j1_allowed_skills(E_t arrays)
    out       = apply_heads(params, [h_E]*B, u, onehot(k_cur), m, E, [x_E]*B)     h_E, x_E as the heads hold them
    k_b       = argmax_k out.skill_logits[b, k]
    terminate = [out.replace_logit_b > 0] and [k_b != k_cur,b]
    slot_b    = highest-priority valid E_t slot naming k_b, lowest slot on a tie, else -1   (network._matching_slot)
    goal_b    = (k_b, out.params_b, E_t.version, slot_b); a skill other than idle gets J0's goal controls
                (GOAL_MINIMUM_COMMITMENT_SECONDS 0.5 s, MAXIMUM_DURATION_SECONDS 10 s, EMERGENCY_EVENTS);
                idle gets 0 s, no limit and the empty mask, as sim_goal_env.py controls a learned choice.
                J0's 2 s evade commitment belongs to J0's damage rule and is not used.
    diagnostics = {"confidence": out.confidence, "entropy": -sum_k p_k log p_k}, p = softmax(out.skill_logits)
    state'    = state + 1   (decisions made per lane)

Host mode: the instance runs between steps, never inside the compiled core
(``jev/runner.py`` refuses a traced call). It reads B_t (features, feature_known,
latent) and E_t only; actor_input, key, decision_context and goal_runtime are accepted
for the JevInstance contract and not read. Invalid input raises (fail closed); a raise
in the eager session counts as a miss there. A grown catalogue is a new instance with
the same matching-head params: no weight depends on the catalogue size.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.components import GoalDecision
from adk.architecture.decision_context import DecisionContext
from adk.architecture.hierarchy import GoalRuntimeContext
from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.contracts import (
    ExecutiveArrays,
    ExecutiveState,
    GoalPayload,
    JevMode,
    SharedBeliefState,
    event_mask,
    executive_arrays,
    make_goal,
)
from experimental.neural_agent.jev.lm.prompting import validate_skill_ids
from experimental.neural_agent.jev.network import IDLE_SKILL_INDEX, _matching_slot, j1_allowed_skills
from experimental.neural_agent.jev.scripted import (
    EMERGENCY_EVENTS,
    GOAL_MINIMUM_COMMITMENT_SECONDS,
    MAXIMUM_DURATION_SECONDS,
)
from experimental.neural_agent.jev.system_one.data import (
    catalogue_embeddings,
    encode_goal_texts,
    goal_texts,
    render_executive,
    skill_descriptions_of,
    slot_inputs,
)
from experimental.neural_agent.jev.system_one.heads import (
    HeadsParams,
    apply_heads,
    belief_inputs,
    encode_executive,
    heads_dims,
)
from experimental.neural_agent.jev.system_one.trunk import TrunkEncoder
from experimental.neural_agent.shared_state import ExecutiveSection, SharedState

JEV_NAME: str = "js1"
_encode_executive = jax.jit(encode_executive)


@functools.partial(jax.jit, static_argnames=("num_skills",))
def fast_path(
    params: HeadsParams,
    h_executive: jax.Array | None,
    belief: SharedBeliefState,
    current: jax.Array,
    executive: ExecutiveArrays,
    skill_embeddings: jax.Array | None,
    executive_code: jax.Array | None = None,
    *,
    num_skills: int,
) -> tuple[Any, jax.Array, jax.Array, jax.Array]:
    """The per-decision computation (module formulas), compiled; no text and no trunk.

    ``h_executive`` (float32[D]) is None without the text path, ``executive_code`` (float32[2S],
    the cached [g_E ; q_top]) None without the structured encoder.

    Returns:
        ``(goal, terminate, confidence, entropy)``: a ManagedGoal batched [B], bool[B],
        float32[B] and float32[B].
    """

    lanes = current.shape[0]
    u = belief_inputs(belief.features, belief.feature_known, belief.latent)
    h = None if h_executive is None else jnp.broadcast_to(h_executive, (lanes, h_executive.shape[0]))
    code = None if executive_code is None else jnp.broadcast_to(executive_code, (lanes, executive_code.shape[0]))
    allowed = j1_allowed_skills(executive, num_skills=num_skills)
    out = apply_heads(params, h, u, jax.nn.one_hot(current, num_skills), allowed, skill_embeddings, code)
    skill = jnp.argmax(out.skill_logits, axis=-1).astype(jnp.int32)
    log_probs = jax.nn.log_softmax(out.skill_logits, axis=-1)
    entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
    has_goal = skill != IDLE_SKILL_INDEX
    goal = make_goal(
        skill,
        out.params,
        executive.version,
        _matching_slot(executive, skill),
        minimum_commitment_seconds=jnp.where(has_goal, GOAL_MINIMUM_COMMITMENT_SECONDS, 0.0),
        maximum_duration_seconds=jnp.where(has_goal, MAXIMUM_DURATION_SECONDS, jnp.inf),
        emergency_interrupt_mask=jnp.where(
            has_goal[:, None], event_mask(*EMERGENCY_EVENTS, batch=lanes), event_mask(batch=lanes)
        ),
    )
    terminate = (out.replace_logit > 0.0) & (skill != current)
    return goal, terminate, out.confidence, entropy


class SystemOneJev:
    """J-S1 as a ``JevInstance`` in ``JevMode.HOST`` (module formulas).

    Args:
        trunk: A TrunkEncoder, frozen. Called on the slow path only.
        params: HeadsParams whose trunk width is ``trunk.dim``. A fixed head (the ablation)
            must score ``len(skill_ids)`` skills; a matching head scores any catalogue. The
            params' shapes say whether the structured encoder and the text path are used.
        skill_ids: The catalogue in dispatcher order, idle first.
        skill_descriptions: One description per skill, or None to describe each skill by its id.

    Raises:
        TypeError: If ``trunk`` is not a TrunkEncoder or ``params`` not a HeadsParams.
        ValueError: If the catalogue or the descriptions are invalid, or a width disagrees.

    Attributes:
        name: "js1".
        mode: ``JevMode.HOST``.
        params: The HeadsParams in use.
        skill_embeddings: float32[K, D], the cached e_j of the matching head; None for the fixed head.
    """

    name: str = JEV_NAME
    mode: JevMode = JevMode.HOST

    def __init__(
        self,
        trunk: TrunkEncoder,
        params: HeadsParams,
        skill_ids: Sequence[str],
        skill_descriptions: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(trunk, TrunkEncoder):
            raise TypeError("trunk must implement TrunkEncoder (dim and encode)")
        dims = heads_dims(params)
        catalogue = validate_skill_ids(skill_ids)
        descriptions = skill_descriptions_of(catalogue, skill_descriptions)
        if dims.skill_head == "fixed" and dims.num_skills != len(catalogue):
            raise ValueError(f"the fixed head scores {dims.num_skills} skills; the catalogue has {len(catalogue)}")
        if dims.trunk_dim != trunk.dim:
            raise ValueError(f"the heads read a trunk of width {dims.trunk_dim}; this trunk has width {trunk.dim}")
        self.params = params
        self._trunk = trunk
        self._skill_ids = catalogue
        self._descriptions = descriptions
        self._dims = dims
        self._skill_table = None if dims.skill_head == "fixed" else catalogue_embeddings(trunk, catalogue, descriptions)
        self.skill_embeddings = None if self._skill_table is None else jnp.asarray(self._skill_table)
        self._label_embeddings: dict[str, np.ndarray] = {}
        self._text_embeddings: dict[str, np.ndarray] = {}
        self._words: tuple[tuple[str, str], ...] = ()
        self._executive: ExecutiveState | None = None
        self._h_executive: jax.Array | None = None
        self._executive_code: jax.Array | None = None
        self._lowered: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def set_executive_state(self, state: ExecutiveState) -> None:
        """Slow path: set E_t and read it once per version: its stable text (text path) and its
        slots (structured encoder; new constraint labels are encoded once and cached). An
        ExecutiveState carries no words, so heads with goal text read zero text columns here.

        The same version with the same content again is a no-op (no trunk call).

        Raises:
            TypeError: If ``state`` is not an ExecutiveState.
            KeyError: If a goal names a skill outside the catalogue.
            ValueError: If ``state`` reuses the version of the E_t in force with different content.
        """

        if not isinstance(state, ExecutiveState):
            raise TypeError("state must be an ExecutiveState")
        self._read(state, state)

    def set_shared_state(self, shared: SharedState) -> None:
        """Slow path from the scratchpad S_t: read its ``executive`` section (SYSTEM_ONE.md §8.10).

        As :meth:`set_executive_state` on the section's typed fields (once per E_t version); heads with
        goal text also read each slot's ``intent`` and ``when`` (M12): the texts not seen before are
        encoded in one batched trunk call and cached for the instance's life. With goal text, the same
        version with other words is other content.

        Raises:
            TypeError: If ``shared`` is not a SharedState.
            KeyError: If a goal names a skill outside the catalogue.
            ValueError: If the section reuses the version of the E_t in force with different content.
        """

        if not isinstance(shared, SharedState):
            raise TypeError("shared must be a SharedState")
        section = shared.executive
        self._read(section.to_executive_state(), section)

    def _read(self, state: ExecutiveState, source: ExecutiveState | ExecutiveSection) -> None:
        """Read E_t = ``state`` once per version; ``source`` is the record whose goals carry the words (or none)."""

        executive_arrays(state, self._skill_ids, 1)
        words = tuple((getattr(goal, "intent", ""), getattr(goal, "when", "")) for goal in source.goals)
        if self._executive is not None and state.version == self._executive.version:
            if state != self._executive or (self._dims.goal_text and words != self._words):
                raise ValueError(f"executive version {state.version} is already set with different content")
            return
        h_executive = code = None
        if self._dims.text_path:
            text = render_executive(self._skill_ids, self._descriptions, state)
            h_executive = jnp.asarray(self._trunk.encode([text])[0], dtype=jnp.float32)
        if self._dims.slot_width:
            new = sorted({label for goal in state.goals for label in goal.constraints} - set(self._label_embeddings))
            if new:
                self._label_embeddings.update(zip(new, np.asarray(self._trunk.encode(new), dtype=np.float32)))
            texts = None
            if self._dims.goal_text:
                texts = encode_goal_texts(self._trunk, goal_texts(source.goals), self._text_embeddings)
            slots = slot_inputs(source, self._skill_ids, self._skill_table, self._label_embeddings, text_embeddings=texts)
            code = _encode_executive(self.params, jnp.asarray(slots))
        self._h_executive, self._executive_code = h_executive, code
        self._executive, self._words = state, words
        self._lowered = {}

    def initial_state(self, batch: int) -> jax.Array:
        """Return int32[batch] zeros: the decisions made per lane."""

        if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch < 1:
            raise ValueError(f"batch must be a positive integer, got {batch!r}")
        return jnp.zeros((int(batch),), dtype=jnp.int32)

    def __call__(
        self,
        state: Any,
        belief: SharedBeliefState,
        current_goal: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        *,
        decision_context: DecisionContext | None,
        goal_runtime: GoalRuntimeContext | None,
        executive: ExecutiveArrays,
    ) -> GoalDecision:
        """Fast path: decide every lane in one compiled pass, with no text and no trunk call.

        Returns:
            GoalDecision: ``state`` int32[B]; ``goal`` a ManagedGoal with a GoalPayload;
            ``terminate`` bool[B]; ``diagnostics`` ``{"confidence": float32[B],
            "entropy": float32[B]}``.

        Raises:
            TypeError: If a record has the wrong type.
            ValueError: If no E_t is set, the executive arrays were not lowered from it onto
                this catalogue, a current skill is outside the catalogue, or a shape is wrong.
        """

        del actor_input, key, decision_context, goal_runtime
        if not isinstance(belief, SharedBeliefState):
            raise TypeError("belief must be a SharedBeliefState")
        if not isinstance(current_goal, GoalPayload):
            raise TypeError("current_goal must be a GoalPayload")
        if not isinstance(executive, ExecutiveArrays):
            raise TypeError("executive must be an ExecutiveArrays")
        if self._executive is None:
            raise ValueError("no executive state is set; call set_executive_state before deciding")
        counts = np.asarray(state)
        lanes = counts.shape[0] if counts.ndim == 1 else 0
        if lanes < 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError(f"state must be an integer array of shape (B,), got {counts.shape} {counts.dtype}")
        if lanes not in self._lowered:
            lowered = executive_arrays(self._executive, self._skill_ids, lanes)
            self._lowered[lanes] = tuple(np.asarray(leaf) for leaf in (lowered.skill_index, lowered.goal_valid, lowered.version))
        for name, expected in zip(("skill_index", "goal_valid", "version"), self._lowered[lanes]):
            if not np.array_equal(expected, np.asarray(getattr(executive, name))):
                raise ValueError(
                    f"executive.{name} was not lowered from the executive state set on this instance onto this catalogue"
                )
        current = np.asarray(current_goal.skill_index)
        if current.shape != (lanes,) or not np.issubdtype(current.dtype, np.integer):
            raise ValueError(f"current_goal.skill_index must be an integer array of shape ({lanes},)")
        if np.any((current < 0) | (current >= len(self._skill_ids))):
            raise ValueError(f"current skill indices must lie in [0, {len(self._skill_ids)}), got {current.tolist()}")
        goal, terminate, confidence, entropy = fast_path(
            self.params,
            self._h_executive,
            belief,
            jnp.asarray(current, dtype=jnp.int32),
            executive,
            self.skill_embeddings,
            self._executive_code,
            num_skills=len(self._skill_ids),
        )
        return GoalDecision(
            state=jnp.asarray(counts + 1, dtype=jnp.int32),
            goal=goal,
            terminate=terminate,
            diagnostics={"confidence": confidence, "entropy": entropy},
        )


__all__ = ["JEV_NAME", "SystemOneJev", "fast_path"]
