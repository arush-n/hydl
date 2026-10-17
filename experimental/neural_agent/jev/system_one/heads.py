"""J-S1 heads in JAX: belief adapter, LayerNorm fusion, one GELU layer, parallel heads.

Formulas of ``docs/SYSTEM_ONE.md`` section 8.3 and FORMULAS (M2), (M3), (M3b); lane i,
skill j of a catalogue of K skills, P = PARAM_WIDTH::

    u_i  = [ where(known_i, features_i, 0) ; known_i ; latent_i ]     width 10 + 10 + 64 = 84 (j1_inputs pieces)
    a_i  = W_u u_i + b_u                                                belief adapter, D_a = 64
    z_i  = LayerNorm([h_i ; a_i ; c_i])                                 gain and bias trained, eps 1e-5
    g_i  = GELU(W_g z_i + b_g)                                          width 256, exact GELU x Phi(x)

    matching skill head (default, (M3b)): e_j = trunk embedding of skill j's description, E = [e_1 .. e_K]
        c_i  = onehot(current_i) E = e_current                          the current skill, catalogue-agnostic
        s_ij = (W_q g_i) . (W_k e_j) / sqrt(d_k)                        d_k = 64
    fixed skill head (ablation, (M3)):
        c_i  = onehot(current_i)
        s_i  = W_s g_i + b_s                                            R^K
    s~_ij = where(m_ij, s_ij, -1e9)                                      masked to the allowed skills

    r_i  = w_r . g_i + b_r                                              keep-or-replace logit
    p_i  = W_p g_i + b_p                                                parameters, R^P
    c_i  = sigmoid(w_c . g_i + b_c)                                     confidence that argmax s~_i is right
    V_i  = w_v . g_i + b_v                                              critic, stage C only
    plan, k = 1..K_plan (HYPOTHESIS): matching s_ij^(k) = (W_k^plan g_i) . (W_k e_j) / sqrt(d_k);
                                      fixed    s_i^(k)  = W_k^plan g_i + b_k; masked like s~

With the matching head no weight shape depends on K: a skill added after training is
one more row e_j, scored with no new weight. The spec's fusion carries onehot(current),
whose width is K; the matching head carries the current skill's own embedding e_current
instead, the catalogue-agnostic code of the same input. The fixed head keeps the
spec's onehot. Which head a HeadsParams holds is read from its shapes (:func:`heads_dims`):
the fixed head has w_q, w_k with d_k = 0 columns, the matching head has w_s, b_s with
0 skills.

LayerNorm: Ba, Kiros, Hinton, arXiv:1607.06450. GELU: Hendrycks, Gimpel,
arXiv:1606.08415. Scaled dot product: Vaswani et al., arXiv:1706.03762.
Initialisation (this module's choice, J1's where J1 has the same head): LeCun normal
(std sqrt(1/fan_in)) for W_u, W_s, W_q, W_k, w_r and the plan heads; He normal
(std sqrt(2/fan_in)) for W_g; zeros for every bias and for the params, confidence and
value heads, so c_i = 0.5 until stage B fits it.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.contracts import BELIEF_FEATURE_COUNT, BELIEF_LATENT_WIDTH, PARAM_WIDTH

BELIEF_WIDTH: int = 2 * BELIEF_FEATURE_COUNT + BELIEF_LATENT_WIDTH
ADAPTER_WIDTH: int = 64
HIDDEN_WIDTH: int = 256
KEY_WIDTH: int = 64
MASKED_LOGIT: float = -1.0e9
LAYER_NORM_EPSILON: float = 1.0e-5
SKILL_HEADS: tuple[str, ...] = ("matching", "fixed")


class HeadsParams(NamedTuple):
    """J-S1 head parameters, float32. U belief width, D trunk width, D_a adapter width, H hidden
    width, P = PARAM_WIDTH, K_p plan length (0 = no plan head). Matching head: d_k >= 1, K = 0,
    code width C = D. Fixed head: K >= 1 skills, d_k = 0, C = K. F = D + D_a + C.

    Shapes: w_u [U, D_a], b_u [D_a]; ln_scale [F], ln_bias [F]; w_g [F, H], b_g [H];
    w_s [H, K], b_s [K]; w_q [H, d_k], w_k [D, d_k]; w_r [H], b_r []; w_p [H, P], b_p [P];
    w_c [H], b_c []; w_v [H], b_v []; w_plan [K_p, H, d_k or K], b_plan [K_p, K].
    """

    w_u: jax.Array
    b_u: jax.Array
    ln_scale: jax.Array
    ln_bias: jax.Array
    w_g: jax.Array
    b_g: jax.Array
    w_s: jax.Array
    b_s: jax.Array
    w_q: jax.Array
    w_k: jax.Array
    w_r: jax.Array
    b_r: jax.Array
    w_p: jax.Array
    b_p: jax.Array
    w_c: jax.Array
    b_c: jax.Array
    w_v: jax.Array
    b_v: jax.Array
    w_plan: jax.Array
    b_plan: jax.Array


class HeadsDims(NamedTuple):
    """The sizes a HeadsParams fixes: trunk width D, the fixed head's K (0 for the matching
    head), the matching head's d_k (0 for the fixed head), the plan length and the head kind."""

    trunk_dim: int
    num_skills: int
    key_width: int
    plan: int
    skill_head: str


class HeadsOutput(NamedTuple):
    """One pass of the heads, B lanes, K skills: skill_logits float32[B, K] (masked), replace_logit
    float32[B], params float32[B, P], confidence float32[B], value float32[B], and plan_logits
    float32[B, K_p, K] (masked), or None when K_p = 0."""

    skill_logits: jax.Array
    replace_logit: jax.Array
    params: jax.Array
    confidence: jax.Array
    value: jax.Array
    plan_logits: jax.Array | None


def _count(value: Any, name: str, minimum: int = 1) -> int:
    """Return ``value`` as an int of at least ``minimum``. Bools are refused."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _require_shape(value: Any, expected: tuple[int, ...], name: str) -> None:
    """Raise ValueError unless ``value`` has exactly the ``expected`` shape."""

    shape = tuple(jnp.shape(value))
    if shape != tuple(expected):
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {shape}")


def init_heads(
    key: jax.Array,
    trunk_dim: int,
    num_skills: int | None = None,
    *,
    skill_head: str = "matching",
    belief_width: int = BELIEF_WIDTH,
    adapter_width: int = ADAPTER_WIDTH,
    hidden: int = HIDDEN_WIDTH,
    key_width: int = KEY_WIDTH,
    plan: int = 0,
) -> HeadsParams:
    """Return freshly initialised heads (initialisation in the module docstring).

    Args:
        key: A JAX PRNG key.
        trunk_dim: D, the trunk output width.
        num_skills: K, required by the fixed head only. The matching head does not depend on
            the catalogue, so it is initialised for the trunk width alone.
        skill_head: ``"matching"`` (default) or ``"fixed"`` (the ablation).
        belief_width: U, the width of u (84 for the contract belief).
        adapter_width: D_a.
        hidden: H.
        key_width: d_k of the matching head.
        plan: K_p, the plan length; 0 builds no plan head.

    Raises:
        ValueError: If a width is not a positive integer, ``plan`` is negative, ``skill_head`` is
            unknown, the fixed head has no ``num_skills`` or the matching head is given one.
    """

    dim = _count(trunk_dim, "trunk_dim")
    width_u = _count(belief_width, "belief_width")
    width_a = _count(adapter_width, "adapter_width")
    width_h = _count(hidden, "hidden")
    steps = _count(plan, "plan", minimum=0)
    if skill_head == "fixed":
        skills, width_k = _count(num_skills, "num_skills"), 0
        code = skills
    elif skill_head == "matching":
        if num_skills is not None:
            raise ValueError("the matching head does not depend on the catalogue size; omit num_skills")
        skills, width_k = 0, _count(key_width, "key_width")
        code = dim
    else:
        raise ValueError(f"skill_head must be one of {SKILL_HEADS}, got {skill_head!r}")
    fused = dim + width_a + code
    key_u, key_g, key_s, key_q, key_k, key_r, key_plan = jax.random.split(key, 7)
    f32 = jnp.float32

    def normal(k: jax.Array, shape: tuple[int, ...], fan_in: int, gain: float = 1.0) -> jax.Array:
        return jax.random.normal(k, shape, f32) * jnp.sqrt(gain / fan_in)

    return HeadsParams(
        w_u=normal(key_u, (width_u, width_a), width_u),
        b_u=jnp.zeros((width_a,), f32),
        ln_scale=jnp.ones((fused,), f32),
        ln_bias=jnp.zeros((fused,), f32),
        w_g=normal(key_g, (fused, width_h), fused, gain=2.0),
        b_g=jnp.zeros((width_h,), f32),
        w_s=normal(key_s, (width_h, skills), width_h),
        b_s=jnp.zeros((skills,), f32),
        w_q=normal(key_q, (width_h, width_k), width_h),
        w_k=normal(key_k, (dim, width_k), dim),
        w_r=normal(key_r, (width_h,), width_h),
        b_r=jnp.zeros((), f32),
        w_p=jnp.zeros((width_h, PARAM_WIDTH), f32),
        b_p=jnp.zeros((PARAM_WIDTH,), f32),
        w_c=jnp.zeros((width_h,), f32),
        b_c=jnp.zeros((), f32),
        w_v=jnp.zeros((width_h,), f32),
        b_v=jnp.zeros((), f32),
        w_plan=normal(key_plan, (steps, width_h, width_k or skills), width_h),
        b_plan=jnp.zeros((steps, skills), f32),
    )


def heads_dims(params: HeadsParams) -> HeadsDims:
    """Return the sizes of ``params`` after checking every shape against the others.

    Raises:
        TypeError: If ``params`` is not a HeadsParams.
        ValueError: If a shape is inconsistent, or the params hold both skill heads or neither.
    """

    if not isinstance(params, HeadsParams):
        raise TypeError(f"params must be a HeadsParams, got {type(params).__name__}")
    for name in ("w_u", "w_g", "w_s", "w_q", "w_k", "w_p"):
        if jnp.ndim(getattr(params, name)) != 2:
            raise ValueError(f"HeadsParams.{name} must be a matrix, got shape {jnp.shape(getattr(params, name))}")
    width_a = jnp.shape(params.w_u)[1]
    fused, width_h = jnp.shape(params.w_g)
    skills = jnp.shape(params.w_s)[1]
    dim, width_k = jnp.shape(params.w_k)
    steps = jnp.shape(params.w_plan)[0] if jnp.ndim(params.w_plan) == 3 else -1
    if (skills > 0) == (width_k > 0) or dim < 1 or steps < 0:
        raise ValueError(
            "HeadsParams must hold exactly one skill head (fixed: w_s with K >= 1 columns; matching: "
            f"w_q, w_k with d_k >= 1 columns); got K = {skills}, d_k = {width_k}, D = {dim}, plan {steps}"
        )
    code = dim if width_k else skills
    expected = {
        "b_u": (width_a,),
        "ln_scale": (dim + width_a + code,),
        "ln_bias": (dim + width_a + code,),
        "w_g": (dim + width_a + code, width_h),
        "b_g": (width_h,),
        "w_s": (width_h, skills),
        "b_s": (skills,),
        "w_q": (width_h, width_k),
        "w_r": (width_h,),
        "b_r": (),
        "w_p": (width_h, PARAM_WIDTH),
        "b_p": (PARAM_WIDTH,),
        "w_c": (width_h,),
        "b_c": (),
        "w_v": (width_h,),
        "b_v": (),
        "w_plan": (steps, width_h, width_k or skills),
        "b_plan": (steps, skills),
    }
    for name, shape in expected.items():
        _require_shape(getattr(params, name), shape, f"HeadsParams.{name}")
    del fused
    return HeadsDims(int(dim), int(skills), int(width_k), int(steps), "matching" if width_k else "fixed")


def belief_inputs(features: Any, feature_known: Any, latent: Any) -> jax.Array:
    """Return u = [where(known, features, 0) ; known ; latent], float32[B, 84] (SYSTEM_ONE.md 8.2).

    These are the first 2F + L columns of ``network.j1_inputs`` (the tests check it).

    Raises:
        ValueError: If the shapes are not (B, 10), (B, 10) and (B, 64).
    """

    values = jnp.asarray(features, dtype=jnp.float32)
    lanes = values.shape[0] if values.ndim == 2 else -1
    _require_shape(values, (lanes, BELIEF_FEATURE_COUNT), "features")
    known = jnp.asarray(feature_known, dtype=jnp.bool_)
    _require_shape(known, (lanes, BELIEF_FEATURE_COUNT), "feature_known")
    latent_values = jnp.asarray(latent, dtype=jnp.float32)
    _require_shape(latent_values, (lanes, BELIEF_LATENT_WIDTH), "latent")
    return jnp.concatenate(
        [jnp.where(known, values, 0.0), known.astype(jnp.float32), latent_values], axis=-1
    )


def apply_heads(
    params: HeadsParams,
    h: Any,
    u: Any,
    current_onehot: Any,
    allowed_mask: Any,
    skill_embeddings: Any = None,
) -> HeadsOutput:
    """Run the heads once on B lanes over a catalogue of K skills (module formulas). Jit-able:
    every check reads static shapes.

    Args:
        params: HeadsParams, either skill head.
        h: float32[B, D], the pooled trunk output.
        u: float32[B, U], the belief vector (:func:`belief_inputs`).
        current_onehot: float32[B, K], the one-hot of the skill in force.
        allowed_mask: bool[B, K], the skills the executive allows (``j1_allowed_skills``).
        skill_embeddings: float32[K, D], the trunk embeddings of the skill descriptions. The
            matching head needs them; the fixed head does not read them.

    Returns:
        HeadsOutput with the shapes on :class:`HeadsOutput`.

    Raises:
        ValueError: If a shape disagrees with ``params``, or the matching head gets no embeddings.
    """

    dims = heads_dims(params)
    hidden = jnp.asarray(h, dtype=jnp.float32)
    lanes = hidden.shape[0] if hidden.ndim == 2 else -1
    _require_shape(hidden, (lanes, dims.trunk_dim), "h")
    belief = jnp.asarray(u, dtype=jnp.float32)
    _require_shape(belief, (lanes, params.w_u.shape[0]), "u")
    allowed = jnp.asarray(allowed_mask, dtype=jnp.bool_)
    skills = allowed.shape[1] if allowed.ndim == 2 else -1
    _require_shape(allowed, (lanes, skills), "allowed_mask")
    current = jnp.asarray(current_onehot, dtype=jnp.float32)
    _require_shape(current, (lanes, skills), "current_onehot")
    if dims.skill_head == "fixed":
        _require_shape(allowed, (lanes, dims.num_skills), "allowed_mask (fixed head)")
        code = current
    else:
        if skill_embeddings is None:
            raise ValueError("the matching head needs skill_embeddings float32[K, D]")
        embeddings = jnp.asarray(skill_embeddings, dtype=jnp.float32)
        _require_shape(embeddings, (skills, dims.trunk_dim), "skill_embeddings")
        keys = embeddings @ params.w_k
        code = current @ embeddings

    adapted = belief @ params.w_u + params.b_u
    fused = jnp.concatenate([hidden, adapted, code], axis=-1)
    mean = jnp.mean(fused, axis=-1, keepdims=True)
    variance = jnp.var(fused, axis=-1, keepdims=True)
    z = (fused - mean) / jnp.sqrt(variance + LAYER_NORM_EPSILON) * params.ln_scale + params.ln_bias
    g = jax.nn.gelu(z @ params.w_g + params.b_g, approximate=False)
    if dims.skill_head == "fixed":
        logits = g @ params.w_s + params.b_s
    else:
        logits = (g @ params.w_q) @ keys.T / np.sqrt(dims.key_width)
    plan_logits = None
    if dims.plan > 0:
        if dims.skill_head == "fixed":
            plan = jnp.einsum("bh,phk->bpk", g, params.w_plan) + params.b_plan
        else:
            plan = jnp.einsum("bh,phd,kd->bpk", g, params.w_plan, keys) / np.sqrt(dims.key_width)
        plan_logits = jnp.where(allowed[:, None, :], plan, MASKED_LOGIT)
    return HeadsOutput(
        skill_logits=jnp.where(allowed, logits, MASKED_LOGIT),
        replace_logit=g @ params.w_r + params.b_r,
        params=g @ params.w_p + params.b_p,
        confidence=jax.nn.sigmoid(g @ params.w_c + params.b_c),
        value=g @ params.w_v + params.b_v,
        plan_logits=plan_logits,
    )


__all__ = [
    "ADAPTER_WIDTH",
    "BELIEF_WIDTH",
    "HIDDEN_WIDTH",
    "HeadsDims",
    "HeadsOutput",
    "HeadsParams",
    "KEY_WIDTH",
    "LAYER_NORM_EPSILON",
    "MASKED_LOGIT",
    "SKILL_HEADS",
    "apply_heads",
    "belief_inputs",
    "heads_dims",
    "init_heads",
]
