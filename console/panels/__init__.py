"""Built-in dashboard panels.

Every module here calls :func:`console.core.panels.panel` at import time and is
picked up by :func:`console.core.panels.load_builtin`. Dropping a new module in
this directory adds a panel to the dashboard -- no route, no HTML, no JS.

These ship as much to be read as to be used: they are the worked examples of
each view primitive.
"""
