"""Does a fixture discriminate skill, or reward survival?

The gate before keeper-scale training: if idle, uniform-legal and a scripted
cadence score about what a trained policy scores, the fixture teaches nothing
and the trained number is meaningless.

**The metric is the whole difficulty.** A run-level total is confounded by
episode count whenever the arms die at different rates -- and dying more is
exactly what a weak arm does. Auto-reset lets a weak arm play many short
episodes and accumulate a *larger* total, so the confound points the wrong way
and the wrong conclusion looks like evidence.

Measured on the arsenal fixture: ``iron_crossbow`` under uniform-legal scored
**45.0 damage** against a scripted **30.0**, which reads as a degenerate
fixture. The arms had **53.7 versus 3.0** episode resets. Per episode that is
``0.84`` versus ``10.0`` -- the scripted arm is ~12x better and the verdict
**inverts**.

So every comparison here is per episode, and ``Arm`` refuses to be built
without the episode count that makes it interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass


#: What each archetype's *outcome* actually is.  Scoring a defense or support
#: archetype on damage is a metric error, not a fixture finding -- they are not
#: built to deal damage, so "DEGENERATE on damage" says nothing about the
#: fixture.  The first run of this gate marked ``iron_shield``, ``potions`` and
#: ``stoneskin_wand`` degenerate for exactly that reason.
#:
#: ``survival`` = ticks before the first ``done``; ``damage`` = damage dealt.
#: Support archetypes have **no measured objective yet** -- effect uptime needs
#: a status channel this harness does not read, and guessing one would repeat
#: the mistake above.
ARCHETYPE_OBJECTIVE = {
    "iron_sword": "damage",
    "iron_shortbow": "damage",
    "iron_crossbow": "damage",
    "adamantite_spear": "damage",
    "crude_spear": "damage",
    "bombs": "damage",
    "iron_shield": "survival",
    "potions": "status_uptime",
    "stoneskin_wand": None,   # no observable outcome yet -- see below
}

#: ``status_uptime`` = fraction of in-episode ticks with a status slot active
#: that was **not** active at spawn.
#:
#: Raw uptime is useless: every profile carries an always-on effect whose
#: ``remaining_seconds`` is FLT_MAX (3.40282e38), pinning uptime at 100%.
#: Scoring only *acquired* slots separates cleanly --
#: ``potions`` 0% idle -> 44% acting (max remaining 5.05 s), while
#: ``iron_sword`` stays 0% in both arms as a negative control.
#:
#: ``stoneskin_wand`` has **no measurable objective on this scene**. Its
#: ability starts (18 rising edges) but acting is indistinguishable from idle
#: on every channel read: damage, projectiles, acquired status,
#: ``damage_resistance_multiplier``, and survival (episode 202 ticks in both
#: arms, hp_end 13.0 in both) -- while ``iron_sword`` in the same scene goes
#: 220 -> 462 ticks of survival when it acts, so the control works.
#:
#: **Class 5, not class 1.** The most likely innocent explanation is scene
#: composition: a buff with no valid target in a bare 1v1. Establish whether
#: the effect is self-targeted before treating this as a defect.


@dataclass(frozen=True, slots=True)
class Arm:
    """One policy's result on a fixture.

    ``episodes`` is mandatory. A total without it cannot be compared to
    anything, and accepting one is how the degenerate-fixture call was nearly
    filed backwards.
    """

    name: str
    total: float
    episodes: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an arm must be named")
        if self.episodes < 0:
            raise ValueError(f"arm {self.name!r} has negative episodes")

    @property
    def per_episode(self) -> float:
        """The only figure worth comparing across arms.

        Zero episodes means the window never terminated; the whole run is then
        one unfinished episode, so the total stands as-is rather than dividing
        by zero.
        """

        return self.total / self.episodes if self.episodes else self.total


def discriminates(candidate: Arm, baselines, *, margin: float = 1.5) -> bool:
    """Does ``candidate`` beat every trivial baseline by ``margin``?

    ``margin`` is a ratio, not a difference, so it is scale-free across
    fixtures. The default 1.5 means "half again as good as the best trivial
    policy" -- below that, a learned policy is not demonstrating skill the
    fixture can measure.
    """

    baselines = tuple(baselines)
    if not baselines:
        raise ValueError("discrimination needs at least one baseline arm")
    if margin <= 1.0:
        raise ValueError(f"margin must exceed 1.0, got {margin}")

    best = max(arm.per_episode for arm in baselines)
    if best <= 0.0:
        # Every trivial policy scored nothing, so any positive score separates.
        return candidate.per_episode > 0.0
    return candidate.per_episode >= best * margin


def report(candidate: Arm, baselines) -> str:
    """A verdict line plus the numbers behind it, both normalisations shown."""

    baselines = tuple(baselines)
    rows = [f"{'arm':>16}{'total':>10}{'episodes':>10}{'per_episode':>13}"]
    rows.append("-" * len(rows[0]))
    for arm in (*baselines, candidate):
        rows.append(
            f"{arm.name[:16]:>16}{arm.total:>10.2f}{arm.episodes:>10}"
            f"{arm.per_episode:>13.3f}"
        )

    best = max(baselines, key=lambda a: a.per_episode)
    verdict = "DISCRIMINATES" if discriminates(candidate, baselines) else "DEGENERATE"
    rows.append("")
    rows.append(
        f"{verdict}: {candidate.name} {candidate.per_episode:.3f}/ep vs best "
        f"trivial {best.name} {best.per_episode:.3f}/ep"
    )
    if verdict == "DEGENERATE":
        rows.append(
            "  A trivial policy nearly matches the candidate. Do not train on "
            "this fixture -- the resulting number would not measure skill."
        )
    return "\n".join(rows)
