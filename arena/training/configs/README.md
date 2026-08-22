# Training configurations

This directory holds checked-in configuration examples, not the implementation of the corresponding training environments.

## `target-tracking-v1.json`

This JSON describes a 512-tick target-tracking/combat-fundamentals setup. Its important choices are:

- tracking and approach shaping are enabled;
- the action scope is restricted to fundamentals;
- basic attacks and terminal victory/damage terms are disabled in this particular profile;
- yaw/pitch success and harmful-turn margins are explicit;
- discount and reward clipping are fixed in the artifact.

The JSON is an input/configuration artifact. The executable reward logic lives in [`../rewards/combat_fundamentals.py`](../rewards/combat_fundamentals.py), while the pursuit-specific collector and transforms live in [`../contracts/pursuit.py`](../contracts/pursuit.py).

