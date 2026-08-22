# Training diagnostics

Diagnostics measure information and timing; they do not change policy parameters, action masks, optimizer state, or checkpoints.

## `decision_interval.py`

`DecisionIntervalProgram` lowers one recurrent policy decision into a semi-Markov physical interval. The learner issues a row once, then neutral rows are used while an accepted ability request is active. A pure-JAX `while_loop` accumulates physical rewards, elapsed ticks, discounts, bootstrap rules, target decisions, and authoritative lifecycle evidence.

This is useful when policy cadence and physical-engine cadence differ. It is not a second simulator and it performs no host callbacks or device transfers.

## `timing_probe.py`

The binary timing separability probe fits a small logistic readout on group-disjoint rows. It compares timing information in raw policy-visible features with timing information in a frozen policy latent. It reports AUROC, AUPRC, balanced NLL, margins, support exclusions, negative-label categories, and predeclared gates.

The probe is diagnostic only: passing it does not train or alter the policy. It answers whether the timing label is present and separable, not whether the agent has learned the desired behaviour.

