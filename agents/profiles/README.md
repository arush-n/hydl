# Agent profiles

This directory contains declarative JSON profiles for the concrete agent
families. A profile supplies names, network/checkpoint choices, and launch
defaults; it is not a checkpoint and does not execute a run by itself.

The Console-facing profile is consumed by [`agents/basic/`](../basic/README.md).
Other profiles support PPO, native, and WorldGen workflows. Profile resolution
ultimately feeds Arena/HytaleGym configuration and is validated before a job is
launched.

Start with the project-level training instructions in
[`README.md`](../../README.md).
