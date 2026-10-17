"""The lower layer of the three-level agent: neural skills that execute the middle layer's requests.

``BUILD.md`` in this folder is the build plan; ``AGENT_BRIEF.md`` is phase A. The package re-exports
nothing. Import the submodules:

* ``surface``: the gen-7 token types and action heads in six scopes (A1).
* ``catalogue``: the skill catalogue parsed into typed records, verb contracts, descriptions and
  activity programs; a JSON dump (A2).
* ``references``: the resolver from a request's world point, entity handle or cell and face to the
  agent-frame vector and the marked token (A4).
"""
