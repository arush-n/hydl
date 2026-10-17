"""J-S1, our own System One decision model (``docs/SYSTEM_ONE.md`` section 8).

J-S1 is one more JEV instance, host mode: it reads the rendered executive state
E_t, the belief B_t and the skill catalogue, and emits skill, replace, params and
a confidence in one pass through parallel heads on a pooled trunk output.

Modules: ``trunk`` (``TrunkEncoder``, ``HashedBagTrunk``, ``TorchTransformerTrunk``),
``heads`` (JAX heads), ``instance`` (``SystemOneJev``), ``data`` (supervised rows
and executive variants), ``train_supervised`` (stage A), ``calibrate`` (stage B),
``probe_latency`` (G2). Nothing is imported here, so importing the package loads
neither JAX training code nor a model. Read ``SKILL.md`` in this folder.
"""
