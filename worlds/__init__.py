"""The saved-world region library ADK and Arena both sit on.

``terrain`` reads the published region corpus, ``zones`` addresses a region by
selection key, and ``region`` loads a Region fixture and everything bound to it.
These used to live under ``adk`` while Arena reached across into them; they are
here so neither tree has to import the other.

Nothing in this package may import ``adk`` or ``arena``.
"""
