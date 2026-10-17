"""Evaluation for the composition arms of experimental/neural_agent (ADR-001 sections 8 and 9).

The modules here measure the hierarchy that the ADK composes:

* ``metrics``: pure numpy functions over recorded per-tick arrays of shape [T, B].
* ``arms``: the comparison arms (flat, j0, shuffled, and the not-yet-written j1 and j_lm).
* ``traces``: the trace dataset written from a hierarchical run for the learned J1.
* ``run_composition``: the command-line run that writes ``metrics.csv`` and ``REPORT.md``.

This package imports no sibling module at import time. The sibling modules
(``belief``, ``skills``, ``jev``, ``tasks``) are imported inside the functions
that need them. Formulas and citations are in ``README.md`` in this folder.
"""
