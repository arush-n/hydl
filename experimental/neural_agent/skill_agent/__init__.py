"""The lower layer of the three-level agent: neural skills that execute the middle layer's requests.

``README.md`` in this folder is the guide to creating, registering and using skills and the reasons for each
component; ``docs/BUILD.md`` is the build plan; ``docs/AGENT_BRIEF.md`` was phase A. The package re-exports nothing.
Import the submodules, which are sorted into subfolders by purpose (each subfolder has a README):

``contract``: what a skill is and what it speaks.

* ``contract.surface``: the gen-7 token types and action heads in six scopes (A1).
* ``contract.catalogue``: the skill catalogue parsed into typed records, verb contracts, descriptions and activity
  programs; a JSON dump (A2).
* ``contract.protocol``: catalogue schemas lowered to the middle layer's request records and its check (A3).
* ``contract.references``: the resolver from a request's world point, entity handle or cell and face to the
  agent-frame vector and the marked token (A4).

``runtime``: running skills, fixed and exact.

* ``runtime.combiner``: the fixed greedy combiner, stub bodies and the in-graph active-request table (A5).
* ``runtime.library``: the skill registry: create, register and use verbs and programs.
* ``runtime.programs``: the runner that turns an activity program into requests and reads the statuses back.
* ``runtime.scratchpad``: the combiner on S_t: requests in from the requests section, statuses out to skills.
* ``runtime.sim_adapter``: the combiner's choices onto the heads a simulator implements; drops counted.
* ``runtime.parity``: today's one goal per lane as one request through the combiner path (A6).
* ``runtime.joint_check``: the composition arms with and without the combiner on S_t behind the program executor (B3).

``model``: the networks.

* ``model.encoder``: the token-set encoder, the one trunk every body reads its tokens through (B1).
* ``model.freezing``: one frozen trunk and one adapter per verb; adding a verb never changes a frozen one (B4); the
  trunk once per tick (``encode_contexts``) and the trained verb as a body (``VerbBody``).
* ``model.generalist``: the request-conditioned generalist policy (``docs/ARCHITECTURE_RESEARCH.md``).
* ``model.skill_module``: a skill's own module on the shared frozen trunk, and the trunk.
* ``model.observation_adapter``: the legal observation to the trunk's token sets (self and entity; body frame).

``neural_skills``: one folder per neural skill (``neural_skills/README.md``): its plan, teacher, task and launchers.

* ``neural_skills.spec``: what a skill declares (``SkillSpec``) and the verb ids (``VERBS``).
* ``neural_skills.walk_to`` and ``neural_skills.look_at``: each skill's ``teacher``, ``task``, ``skill`` (its spec) and
  ``train`` (its own training strategy, which trains its own module).

``perception``: what the skill layer perceives (``docs/PERCEPTION.md``).

* ``perception.registry``: the content registry built from a server version's assets: items, blocks, gather
  types, description vectors.

``teaching``: what scripted teachers share, and their certificates.

* ``teaching.teachers``: the head layout, ``TeacherOutput``, the ``TEACHERS`` registry and the observed-state rule.
* ``teaching.certificates``: the teacher certificate (T1 success, T2 efficiency, T3 obedience, failure sorting) and
  its CLI.
* ``teaching.certify_live``: T4, the walk_to teacher in closed loop on the live 0.7.2 server.

``tasks``: the episodes skills are trained and measured on.

* ``tasks.simple_tasks``: the flat npc_walk environment with explicit spawns, and run_episode.

``training``: data and trainers.

* ``training.traces``: the loader for native NPC traces from the live server (A7).
* ``training.bc_dataset`` and ``training.bc_smoke``: the behaviour-cloning dataset from those traces and its smoke
  (B2).
* ``training.blocks``: the building blocks every trainer shares.
* ``training.skill_dagger``: DAgger for one skill's module on the frozen trunk, run by each skill's strategy.
* ``training.distill``: the A2 reference trainer, walk_to and look_at distilled together into one generalist.
* ``training.bars``: a distillation report scored on A2's bars, on held success.

``bench``: per-tick latency, the 30 Hz gate.

* ``bench.bench_tick``: the per-tick latency bench of resolver, combiner and encoder.
* ``bench.bench_bodies``: the per-tick cost of trained bodies, with the trunk per call or once per tick.
* ``bench.bench_generalist``: the generalist's tick (proof A1).
"""
