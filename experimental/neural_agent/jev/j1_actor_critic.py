"""J1 as the PPO policy, with a separate critic, for ``train_j1_ppo``.

``train_j1_ppo.train_j1_ppo`` takes ``apply_fn(params, obs) -> (logits
float32[B, K], value float32[B])``. J1 (``network.py``) has no value head, so
the parameters here are a pair: the unchanged ``J1Params`` and a small critic
with its own weights. Only the J1 half is deployed. :func:`save_j1_params`
writes it in the format that ``eval/arms.py::load_j1_params`` reads.

Formulas, with x = ``j1_inputs(belief, executive, current_goal, num_skills=K)``
of width D (``network.py`` formula 1) and H_c the critic width:

    logits = z_skill of apply_j1(J1Params, x)          (network.py formula 2)
    h1     = relu(x W1 + b1)                           W1 [D, H_c]
    h2     = relu(h1 W2 + b2)                          W2 [H_c, H_c]
    value  = h2 w_value + b_value                      w_value [H_c]

``train_j1_ppo`` masks the logits with ``obs["mask"]``, the J1 mask A of
``network.py`` section 3. The critic reads the same x as the actor and shares
no weights with it. Initialization: ``init_j1`` for the actor; for the critic,
He normal for W1 and W2 (std sqrt(2 / fan_in)), std sqrt(1 / H_c) for
w_value, and zero biases. These are the hidden and value-head rules of
the first build's stand-in MLP in ``train_j1_ppo`` (removed 2026-10-18), width H_c = 64. Source: this
module's choice, copied from that stand-in. The loss reads only the skill
logits, so J1's replace and params heads get a zero gradient and keep their
initial values.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.jev.network import (
    J1Params,
    apply_j1,
    init_j1,
    input_width,
    j1_inputs,
)

#: Width H_c of both critic hidden layers: 64, the former stand-in's default.
CRITIC_HIDDEN: int = 64


class CriticParams(NamedTuple):
    """Critic weights, float32, with D = ``network.input_width(K)``.

    Shapes: w1 [D, H_c], b1 [H_c], w2 [H_c, H_c], b2 [H_c], w_value [H_c],
    b_value [] (a scalar).
    """

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w_value: jax.Array
    b_value: jax.Array


class J1ActorCriticParams(NamedTuple):
    """``(J1Params, CriticParams)``: the J1 policy and its critic."""

    actor: J1Params
    critic: CriticParams


def init_j1_actor_critic(key: jax.Array, num_skills: int) -> J1ActorCriticParams:
    """Initialize the actor with ``init_j1`` and the critic as in the module docstring.

    Args:
        key: A JAX PRNG key.
        num_skills: Catalog size K, an integer of at least 1.

    Returns:
        J1ActorCriticParams. The actor has ``init_j1``'s default hidden width,
        and the critic width is ``CRITIC_HIDDEN``.
    """

    actor_key, critic_key = jax.random.split(key)
    width = input_width(num_skills)
    critic_hidden = CRITIC_HIDDEN
    key_w1, key_w2, key_value = jax.random.split(critic_key, 3)
    f32 = jnp.float32
    critic = CriticParams(
        w1=jax.random.normal(key_w1, (width, critic_hidden), f32) * math.sqrt(2.0 / width),
        b1=jnp.zeros((critic_hidden,), f32),
        w2=jax.random.normal(key_w2, (critic_hidden, critic_hidden), f32)
        * math.sqrt(2.0 / critic_hidden),
        b2=jnp.zeros((critic_hidden,), f32),
        w_value=jax.random.normal(key_value, (critic_hidden,), f32)
        * math.sqrt(1.0 / critic_hidden),
        b_value=jnp.zeros((), f32),
    )
    return J1ActorCriticParams(init_j1(actor_key, num_skills=num_skills), critic)


def apply_j1_actor_critic(
    params: J1ActorCriticParams, obs: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    """Return unmasked ``(logits float32[B, K], value float32[B])``: the trainer's ``apply_fn``.

    Args:
        params: J1ActorCriticParams.
        obs: A ``GoalDecisionEnv`` observation with ``belief``, ``executive``
            and ``current_goal``. K is read from ``params.actor.w_skill``.

    Returns:
        J1's skill logits and the critic's value.
    """

    skills = int(jnp.shape(params.actor.w_skill)[-1])
    x = j1_inputs(
        obs["belief"], obs["executive"], obs["current_goal"], num_skills=skills
    )
    logits, _, _ = apply_j1(params.actor, x)
    critic = params.critic
    hidden = jax.nn.relu(jax.nn.relu(x @ critic.w1 + critic.b1) @ critic.w2 + critic.b2)
    return logits, hidden @ critic.w_value + critic.b_value


def with_actor_from_file(params: J1ActorCriticParams, path: str | Path) -> J1ActorCriticParams:
    """Return ``params`` with the actor replaced by the J1Params saved at ``path``.

    The file is the one ``train_j1.save_params`` or :func:`save_j1_params` writes.
    Seeding PPO from behaviour-cloned parameters keeps the replace and params
    heads that PPO does not train (added 2026-10-17).

    Raises:
        ValueError: If any array's shape differs from the current actor's, which
            covers a different catalogue size or hidden width.
    """

    from experimental.neural_agent.jev.train_j1 import load_params

    actor = load_params(path)
    for name in J1Params._fields:
        current, loaded = getattr(params.actor, name), getattr(actor, name)
        if tuple(loaded.shape) != tuple(current.shape):
            raise ValueError(
                f"{path}: {name} has shape {tuple(loaded.shape)}; the actor expects {tuple(current.shape)}"
            )
    return params._replace(actor=actor)


def save_j1_params(path: str | Path, params: J1ActorCriticParams) -> Path:
    """Write only the J1 half as an ``.npz`` that ``eval/arms.py::load_j1_params`` reads.

    The file holds exactly the ten ``J1Params`` fields as float32 arrays,
    ``np.savez(path, **params.actor._asdict())``.

    Args:
        path: Target path. Its suffix must be ``.npz``.
        params: J1ActorCriticParams.

    Returns:
        The path written.

    Raises:
        ValueError: If the suffix is not ``.npz``.
    """

    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError(f"J1 parameters must be saved as .npz, got {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        **{
            name: np.asarray(value, dtype=np.float32)
            for name, value in params.actor._asdict().items()
        },
    )
    return target


__all__ = [
    "CRITIC_HIDDEN",
    "CriticParams",
    "J1ActorCriticParams",
    "apply_j1_actor_critic",
    "init_j1_actor_critic",
    "save_j1_params",
]
