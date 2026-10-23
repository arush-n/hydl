"""Who an agent is: its traits, as continuous scalars (``docs/VIRTUAL_HUMAN.md``; owner, 2026-10-21: "unique scalar
values and not discrete options ... ideally everything should be a combination ... shared throughout the entire
architecture to emulate human-like behavior").

A persona is a vector float32[P] over :data:`TRAITS`, the self section of the variable registry
(``contract/variables.py``; the trait list of ``docs/VIRTUAL_HUMAN.md`` section 15.6 and the motor signature's bias),
each in [-1, 1] with 0 the population's mean. The LLM's persona record holds it, set at an agent's birth and never per
tick; it lives in the state layer's variables (``state/registry.py``), where the decision layer and the LLM read all of
it. The body reads the traits it has a mechanism for (section 15.1) as its ``persona`` token, which also conditions
every other token (``EncoderConfig.condition_types``), and the emotion rules take their gains from it
(``state/affect.py``).

:func:`sample_personas` draws a population whose traits correlate as people's do. The Big Five correlate at about
0.1 to 0.3 (van der Linden et al. 2010) through two higher factors, stability (conscientiousness, agreeableness, low
neuroticism) and plasticity (extraversion, openness) (DeYoung et al. 2002); economic preferences correlate at the
same size (Falk et al. 2018: patience and risk 0.21, altruism and positive reciprocity 0.33). So each trait loads on
four latent axes (stability, plasticity, honesty, reciprocity) plus its own variance, and the motor signature loads
on none. The loadings of the motivation and value traits are our estimates [E], set by the trait each resembles
(boredom proneness and distractibility with low stability; the values with plasticity and honesty).
:func:`big_five` is the derived narrative view, for the LLM and the mood's baseline: the traits are defined as
behaviour, and the Big Five are read from them, not the reverse (trait-to-play-style links are weak in games:
Tekofsky et al. 2013).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import variables

TRAITS: tuple[str, ...] = tuple(name for name in variables.names("self") if variables.BY_NAME[name].source == "birth")
"""A persona's entries: the ``self`` variables set at birth (section 15.6), in registry order. The ``self`` section
also holds the activity counters (``rule``, section 21), which are not traits."""

AXES: tuple[str, ...] = ("stability", "plasticity", "honesty", "reciprocity")
"""The latent axes of :func:`sample_personas`."""

_LOADINGS: dict[str, dict[str, float]] = {
    "patience": {"stability": 0.4},
    "risk_taking": {"plasticity": 0.35, "stability": -0.15},
    "positive_reciprocity": {"reciprocity": 0.5},
    "negative_reciprocity": {"honesty": -0.3, "stability": -0.2},
    "altruism": {"honesty": 0.4, "reciprocity": 0.35},
    "trust": {"reciprocity": 0.35, "plasticity": 0.15},
    "effort_sensitivity": {"stability": -0.35},
    "impulsiveness": {"stability": -0.45},
    "self_control": {"stability": 0.45},
    "distractibility": {"stability": -0.3},
    "self_efficacy": {"stability": 0.3, "plasticity": 0.2},
    "boredom_proneness": {"stability": -0.3, "plasticity": 0.1},
    "negative_emotionality": {"stability": -0.55},
    "positive_emotionality": {"plasticity": 0.45},
    "openness_to_change": {"plasticity": 0.5},
    "conservation": {"plasticity": -0.35, "stability": 0.2},
    "self_enhancement": {"honesty": -0.4},
    "self_transcendence": {"honesty": 0.4, "reciprocity": 0.2},
    "vigilance": {"stability": 0.25},
}
"""Each trait's loadings on ``AXES`` (the rest 0). A loading of 0.4 to 0.6 on a shared axis gives correlations of 0.1
to 0.3 between traits, the published size; the motor signature and the play-style traits without evidence load on
none."""


def loading_matrix() -> np.ndarray:
    """float32[P, len(AXES)]: :data:`_LOADINGS` in ``TRAITS`` order."""

    matrix = np.zeros((len(TRAITS), len(AXES)), np.float32)
    for row, trait in enumerate(TRAITS):
        for axis, value in _LOADINGS.get(trait, {}).items():
            matrix[row, AXES.index(axis)] = value
    return matrix


def neutral_persona(batch: int) -> jax.Array:
    """float32[B, P]: every trait at the population's mean."""

    return jnp.zeros((batch, len(TRAITS)), jnp.float32)


def sample_personas(key: jax.Array, count: int, *, spread: float = 0.5) -> jax.Array:
    """float32[count, P]: a population whose traits correlate through ``AXES``.

    Each trait is ``tanh(spread * (L a + sqrt(1 - |L|^2) e))`` with a and e standard normal: unit variance before the
    squash, so ``spread`` sets how far agents differ (0.5 puts about 95% of each trait inside [-0.76, 0.76]).
    """

    loadings = jnp.asarray(loading_matrix())
    shared, own = jax.random.split(key)
    axes = jax.random.normal(shared, (count, len(AXES)), jnp.float32)
    unique = jax.random.normal(own, (count, len(TRAITS)), jnp.float32)
    residual = jnp.sqrt(jnp.clip(1.0 - jnp.sum(loadings ** 2, axis=1), 0.0, 1.0))
    return jnp.tanh(spread * (axes @ loadings.T + unique * residual))


def trait(persona: jax.Array, name: str) -> jax.Array:
    """``persona[..., name]``."""

    return persona[..., TRAITS.index(name)]


def big_five(persona: jax.Array) -> dict[str, jax.Array]:
    """The derived Big Five view, each in [-1, 1], for the LLM's narrative and the mood's baseline. Each factor is the
    mean of the traits nearest it [E]: openness from openness to change; conscientiousness from self-control and
    patience, less impulsiveness and distractibility; extraversion from positive emotionality; agreeableness from
    altruism and trust, less negative reciprocity; neuroticism from negative emotionality and impulsiveness (the NEO-PI-R
    places impulsiveness under neuroticism; recalled)."""

    def mean(*names: str, minus: tuple[str, ...] = ()) -> jax.Array:
        total = sum(trait(persona, name) for name in names) - sum(trait(persona, name) for name in minus)
        return total / (len(names) + len(minus))

    return {"openness": trait(persona, "openness_to_change"),
            "conscientiousness": mean("self_control", "patience", minus=("impulsiveness", "distractibility")),
            "extraversion": trait(persona, "positive_emotionality"),
            "agreeableness": mean("altruism", "trust", minus=("negative_reciprocity",)),
            "neuroticism": mean("negative_emotionality", "impulsiveness")}


__all__ = ["AXES", "TRAITS", "big_five", "loading_matrix", "neutral_persona", "sample_personas", "trait"]
