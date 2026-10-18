"""The lower layer of the three-level agent: neural skills that execute the middle layer's requests.

``BUILD.md`` in this folder is the build plan; ``AGENT_BRIEF.md`` is phase A; ``README.md`` is the guide
to creating, registering and using skills and the reasons for each component. The package re-exports
nothing. Import the submodules:

* ``surface``: the gen-7 token types and action heads in six scopes (A1).
* ``catalogue``: the skill catalogue parsed into typed records, verb contracts, descriptions and
  activity programs; a JSON dump (A2).
* ``references``: the resolver from a request's world point, entity handle or cell and face to the
  agent-frame vector and the marked token (A4).
* ``protocol``: catalogue schemas lowered to the middle layer's request records and its check (A3).
* ``combiner``: the fixed greedy combiner, stub bodies and the in-graph active-request table (A5).
* ``parity``: today's one goal per lane as one request through the combiner path (A6).
* ``training.traces``: the loader for native NPC traces from the live server (A7).
* ``training.bc_dataset`` and ``training.bc_smoke``: the behaviour-cloning dataset from those traces and its
  smoke (B2).
* ``library``: the skill registry: create, register and use verbs and programs.
* ``programs``: the runner that turns an activity program into requests and reads the statuses back.
* ``scratchpad``: the combiner on S_t: requests in from the requests section, statuses out to skills.
* ``joint_check``: the composition arms with and without the combiner on S_t behind the program executor (B3).
* ``encoder``: the token-set encoder, the one trunk every body reads its tokens through (B1).
* ``bench_tick``: the per-tick latency bench of resolver, combiner and encoder (the 30 Hz gate).
* ``freezing``: one frozen trunk and one adapter per verb; adding a verb never changes a frozen one (B4).
"""
