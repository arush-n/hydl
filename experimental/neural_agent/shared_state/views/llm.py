"""The LLM's view of S_t: the briefing (``briefing.py``) rendered by ``render.py``.

``view_for_llm(state)`` keeps its name and its one-argument form: the executive harness and the composition
driver call it. The first view (2026-10-16 to 2026-10-18) listed each section as stored; since 2026-10-18 the
view is the briefing: per-NPC rows in words, the goals with how the decision layer read them, the questions
to the LLM, the newest rejections, and what happened since the LLM's own version. ``options`` chooses the
format (markdown by default, json, compact), the tick length for the seconds shown, the lanes (a squad) and
whether the delta block is included.
"""

from __future__ import annotations

from .briefing import briefing
from .render import ViewOptions, render


def view_for_llm(state, options: ViewOptions | None = None) -> str:
    """Render the briefing of ``state`` for the LLM (markdown unless ``options`` says otherwise)."""

    options = options or ViewOptions()
    return render(briefing(state, lanes=options.lanes, rejections=options.rejections), options)


__all__ = ["view_for_llm"]
