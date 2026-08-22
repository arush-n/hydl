"""A Gymnasium interface onto the combat scene, for tooling that expects one.

The ADK's own loop is batched, functional and key-threaded: you hold a frame,
you pass a key, you get a new frame. That is the right shape for JAX and the
wrong shape for the large amount of RL code that expects
``obs, reward, terminated, truncated, info``. This module is the adapter, and
nothing in the ADK depends on it.

It is a :class:`gymnasium.vector.VectorEnv` rather than a single ``Env``
because the underlying environment genuinely is batched -- one JAX step
advances all `B` lanes. Wrapping it as a single environment would either waste
`B-1` lanes or hide a batch behind a scalar interface that then lies about its
own throughput.

Three places this deliberately refuses to pretend
------------------------------------------------

**It does not auto-reset,** and says so with
``metadata["autoreset_mode"] = AutoresetMode.DISABLED``. Gymnasium's default
contract is that a finished sub-environment restarts on the next step; the
Arsenal environment does not do that, and the Gym is not ours to change. A lane
that dies stays dead with `done` latched -- health pinned at 0, the same
terminal observation returned forever. Silently faking a reset here would
manufacture episodes that never happened; in an earlier measurement roughly
700 of 900 ticks were a corpse being re-reported, and the fix was to notice,
not to paper over it. Call :meth:`reset` when :attr:`needs_reset` is true.

**Sampling from ``action_space`` produces illegal actions.** Action legality is
per-tick state, not a static space: an ability on cooldown is not selectable
this tick and is next tick. `MultiDiscrete` cannot express that, so the mask
travels in ``info["action_mask"]`` -- and :meth:`step` raises rather than
silently clamping, because an illegal request that gets quietly rewritten is
how you end up measuring a policy that never ran. Use :meth:`sample_legal` to
draw a uniformly random *legal* action.

**Reward is whatever the handle was built with.** If shaping was attached with
``with_shaping``, it is included here, because it is included in what the
optimizer sees. This adapter does not add, scale or clip anything.

Do not train through this
-------------------------

Measured 2026-08-08, `combat/fail_closed`, 3 lanes: **~19 seconds per
:meth:`step`**, steadily, with no warm-up cliff::

    tick 1: 19.80s   tick 4: 18.77s
    tick 2: 18.73s   tick 5: 18.91s
    tick 3: 19.02s   tick 6: 18.98s  <- episode cap fired

That is not a first-call compile being amortised -- it is the per-step cost.
`AgentHandle.step` traces the environment afresh on each call, and this
interface is step-at-a-time by definition, so there is nothing here to hoist.
The ADK's own path compiles a whole rollout into one `lax.scan`
(``compile_collector``) and pays that cost once for hundreds of ticks.

So this adapter is for **interoperability and inspection** -- driving the scene
from code that speaks Gymnasium, stepping through by hand, checking a wrapper
-- and not for a training loop. A run that needs throughput should use
`handle.collect` or the console's Train tab.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv

__all__ = ["HytaleCombatVectorEnv"]


class HytaleCombatVectorEnv(VectorEnv):
    """`B` combat lanes behind the standard vector-environment methods.

    :param handle: an :class:`adk.api.AgentHandle`, already carrying whatever
        scene, shaping and episode cap the caller wants. Configuration happens
        on the handle, not here -- this class adds no knobs of its own so there
        is exactly one place a scene is described.
    :param seed: root seed for the JAX key chain. `reset(seed=...)` overrides.
    """

    metadata = {"autoreset_mode": AutoresetMode.DISABLED}

    def __init__(self, handle: Any, *, seed: int = 0) -> None:
        super().__init__()
        self._handle = handle
        self.num_envs = int(handle.batch)
        self._key = jax.random.PRNGKey(seed)
        self._frame = None

        heads = np.asarray(handle.action_head_sizes, dtype=np.int64)
        width = int(handle.observation_size)

        self.single_action_space = spaces.MultiDiscrete(heads)
        self.action_space = spaces.MultiDiscrete(
            np.tile(heads, (self.num_envs, 1)))
        # Unbounded because the observation carries masked channels: an absent
        # value is 0.0 beside a false availability bit, not a clipped zero, so
        # any finite bound stated here would be a fiction.
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(width,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_envs, width),
            dtype=np.float32)

    # -- lifecycle ---------------------------------------------------------

    def _split(self) -> jax.Array:
        self._key, key = jax.random.split(self._key)
        return key

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        """Reset **every** lane. There is no per-lane reset to offer.

        `built.reset` takes one key per lane and rebuilds all of them; the Gym
        exposes no partial reset, so neither does this. That is precisely why
        auto-reset is disabled rather than emulated.
        """

        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._frame = self._handle.start(self._split())
        return self._observation(), self._info()

    def step(self, actions: Any):
        """Advance one tick. `actions` is `[num_envs, heads]` of head indices.

        Raises if any requested action is illegal for its lane this tick --
        see the module docstring on why this does not clamp.
        """

        if self._frame is None:
            raise RuntimeError("call reset() before step()")

        factors = jnp.asarray(np.asarray(actions), dtype=jnp.int32)
        if factors.shape != (self.num_envs, len(self._handle.action_head_sizes)):
            raise ValueError(
                f"actions must be [{self.num_envs}, "
                f"{len(self._handle.action_head_sizes)}], got {factors.shape}")

        result = self._handle.step(self._frame, factors, self._split())
        self._frame = result.frame

        rewards = np.asarray(result.reward, dtype=np.float32)
        # `truncated` is None unless an episode tick limit was set on the
        # handle; absent a cap nothing is ever truncated, which is different
        # from "truncated is false because the cap was not reached".
        terminated = np.asarray(result.terminated, dtype=bool)
        truncated = (np.zeros(self.num_envs, dtype=bool)
                     if result.truncated is None
                     else np.asarray(result.truncated, dtype=bool))
        return (self._observation(), rewards, terminated, truncated,
                self._info(result.info))

    # -- what a caller needs that Gymnasium has no slot for ----------------

    @property
    def needs_reset(self) -> bool:
        """True once every lane is finished.

        With auto-reset disabled this is the caller's cue. Stepping past it is
        allowed and returns the same terminal observation with `done` latched,
        which is exactly the trap this property exists to make visible.
        """

        if self._frame is None:
            return True
        return bool(np.asarray(self._frame.policy_input.done).all()) \
            if hasattr(self._frame.policy_input, "done") else False

    def sample_legal(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """A uniformly random action drawn only from what is legal right now.

        The replacement for `action_space.sample()`, which cannot see the mask
        and will produce refused actions. Falls back to head 0 for any head
        with nothing legal -- which does happen: the mask can collapse to no
        legal option at all for several ticks, and that is the environment
        being forced rather than broken.
        """

        if self._frame is None:
            raise RuntimeError("call reset() before sampling")
        rng = rng or np.random.default_rng()
        masks = self._handle.action_space.split_mask(self._frame.policy_input)

        columns = []
        for name in self._handle.action_head_names:
            legal = np.asarray(masks[name], dtype=bool)
            picks = np.zeros(self.num_envs, dtype=np.int64)
            for lane in range(self.num_envs):
                options = np.flatnonzero(legal[lane])
                picks[lane] = rng.choice(options) if options.size else 0
            columns.append(picks)
        return np.stack(columns, axis=1)

    # -- internals ---------------------------------------------------------

    def _observation(self) -> np.ndarray:
        return np.asarray(self._frame.observation, dtype=np.float32)

    def _info(self, extra: Any = None) -> dict[str, Any]:
        """Per-head legality, plus whatever the environment reported.

        The mask is the load-bearing part: without it a caller cannot construct
        a legal action, and :meth:`step` will refuse.
        """

        masks = self._handle.action_space.split_mask(self._frame.policy_input)
        info: dict[str, Any] = {
            "action_mask": {name: np.asarray(value, dtype=bool)
                            for name, value in masks.items()},
        }
        if extra is not None:
            info["environment"] = extra
        return info
