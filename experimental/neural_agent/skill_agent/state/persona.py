"""Who an agent is: its traits, as continuous scalars (``docs/VIRTUAL_HUMAN.md``; owner, 2026-10-21: "unique scalar
values and not discrete options ... ideally everything should be a combination ... shared throughout the entire
architecture to emulate human-like behavior").

A persona is a vector float32[P] over ``contract/tokens_v3.PERSONA_TRAITS``, each trait in [-1, 1] with 0 the
population's mean. The LLM's persona record holds it, set at an agent's birth and never per tick; the decision layer
reads it and forwards it; the neural layer reads it as its ``persona`` token, which also conditions every other token
(``EncoderConfig.condition_types``), and its emotion dynamics take their gains from it (``state/affect.py``).

:func:`sample_personas` draws a population whose traits correlate as people's do. The Big Five correlate at about
0.1 to 0.3 (van der Linden et al. 2010) through two higher factors, stability (conscientiousness, agreeableness, low
neuroticism) and plasticity (extraversion, openness) (DeYoung et al. 2002); economic preferences correlate at the
same size (Falk et al. 2018: patience and risk 0.21, altruism and positive reciprocity 0.33). So each trait loads on
four latent axes (stability, plasticity, honesty, reciprocity) plus its own variance, and the motor signature loads
on none. :func:`big_five` is the derived narrative view, for the LLM and the mood's baseline: the traits are defined
as behaviour, and the Big Five are read from them, not the reverse (trait-to-play-style links are weak in games:
Tekofsky et al. 2013).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.tokens_v3 import PERSONA_TRAITS

AXES: tuple[str, ...] = ("stability", "plasticity", "honesty", "reciprocity")
"""The latent axes of :func:`sample_personas`."""

_LOADINGS: dict[str, dict[str, float]] = {
    "risk_tolerance": {"plasticity": 0.35, "stability": -0.15},
    "patience": {"stability": 0.4},
    "persistence": {"stability": 0.45},
    "aggression": {"stability": -0.3, "honesty": -0.2},
    "social_value_orientation": {"honesty": 0.4, "reciprocity": 0.35},
    "inequity_aversion": {"honesty": 0.35},
    "honesty_humility": {"honesty": 0.6},
    "trust": {"reciprocity": 0.35, "plasticity": 0.15},
    "positive_reciprocity": {"reciprocity": 0.5},
    "negative_reciprocity": {"honesty": -0.3, "stability": -0.2},
    "sociability": {"plasticity": 0.5},
    "conformity": {"stability": 0.3, "plasticity": -0.15},
    "curiosity": {"plasticity": 0.5},
    "deliberation": {"stability": 0.35, "plasticity": 0.15},
    "vigilance": {"stability": 0.25},
    "negative_reactivity": {"stability": -0.55},
    "positive_reactivity": {"plasticity": 0.45},
    "urgency": {"stability": -0.45},
}
"""Each trait's loadings on ``AXES`` (the rest 0). A loading of 0.4 to 0.6 on a shared axis gives correlations of 0.1
to 0.3 between traits, the published size; the motor signature and the play-style traits without evidence load on
none."""


def loading_matrix() -> np.ndarray:
    """float32[P, len(AXES)]: :data:`_LOADINGS` in ``PERSONA_TRAITS`` order."""

    matrix = np.zeros((len(PERSONA_TRAITS), len(AXES)), np.float32)
    for row, trait in enumerate(PERSONA_TRAITS):
        for axis, value in _LOADINGS.get(trait, {}).items():
            matrix[row, AXES.index(axis)] = value
    return matrix


def neutral_persona(batch: int) -> jax.Array:
    """float32[B, P]: every trait at the population's mean."""

    return jnp.zeros((batch, len(PERSONA_TRAITS)), jnp.float32)


def sample_personas(key: jax.Array, count: int, *, spread: float = 0.5) -> jax.Array:
    """float32[count, P]: a population whose traits correlate through ``AXES``.

    Each trait is ``tanh(spread * (L a + sqrt(1 - |L|^2) e))`` with a and e standard normal: unit variance before the
    squash, so ``spread`` sets how far agents differ (0.5 puts about 95% of each trait inside [-0.76, 0.76]).
    """

    loadings = jnp.asarray(loading_matrix())
    shared, own = jax.random.split(key)
    axes = jax.random.normal(shared, (count, len(AXES)), jnp.float32)
    unique = jax.random.normal(own, (count, len(PERSONA_TRAITS)), jnp.float32)
    residual = jnp.sqrt(jnp.clip(1.0 - jnp.sum(loadings ** 2, axis=1), 0.0, 1.0))
    return jnp.tanh(spread * (axes @ loadings.T + unique * residual))


def trait(persona: jax.Array, name: str) -> jax.Array:
    """``persona[..., name]``."""

    return persona[..., PERSONA_TRAITS.index(name)]


def big_five(persona: jax.Array) -> dict[str, jax.Array]:
    """The derived Big Five view, each in [-1, 1] (the trait research's mapping, ``docs/VIRTUAL_HUMAN.md``):
    extraversion from sociability and positive reactivity; neuroticism from negative reactivity and urgency;
    agreeableness from social value orientation and honesty less negative reciprocity; conscientiousness from
    persistence, deliberation and patience; openness from curiosity."""

    def mean(*names: str, minus: tuple[str, ...] = ()) -> jax.Array:
        total = sum(trait(persona, name) for name in names) - sum(trait(persona, name) for name in minus)
        return total / (len(names) + len(minus))

    return {"openness": trait(persona, "curiosity"),
            "conscientiousness": mean("persistence", "deliberation", "patience"),
            "extraversion": mean("sociability", "positive_reactivity"),
            "agreeableness": mean("social_value_orientation", "honesty_humility", minus=("negative_reciprocity",)),
            "neuroticism": mean("negative_reactivity", "urgency")}


__all__ = ["AXES", "big_five", "loading_matrix", "neutral_persona", "sample_personas", "trait"]
