"""Dense factored policy layout for learner observation v3.

The previous flat ``policy.py`` exposed imported public constants as module
attributes in addition to its explicit ``__all__``. Mirror that module surface
so direct attribute imports remain compatible after the package split.
"""

from hytalegym.jax.combat.observation.v3.policy import distribution as _distribution
from hytalegym.jax.combat.observation.v3.policy import layout as _layout
from hytalegym.jax.combat.observation.v3.policy import look_deltas as _look_deltas

globals().update(
    {name: value for name, value in vars(_layout).items() if not name.startswith("_")}
)
globals().update(
    {name: getattr(_distribution, name) for name in _distribution.__all__}
)
globals().update(
    {name: getattr(_look_deltas, name) for name in _look_deltas.__all__}
)

__all__ = [*_layout.__all__, *_distribution.__all__, *_look_deltas.__all__]
