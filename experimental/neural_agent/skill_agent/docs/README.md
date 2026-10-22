# skill_agent/docs/ — the lower layer's design record

Read `BUILD.md` first. Paths in these documents are relative to `skill_agent/` (for example `runtime/combiner.py`,
`docs/BUILD.md`); files outside the package are cited by repo path. A measurement goes into
`experimental/neural_agent/docs/LEDGER.md` (append-only) before it is claimed here.

| Document | Purpose | Edit rule |
|---|---|---|
| `VIRTUAL_HUMAN.md` | The architecture agreed with the owner on 2026-10-21: the agent as a virtual human; the layers (LLM, decision, the state layer, the body); the state layer (in memory, queried and formatted, read by every layer; game-tied variables from the game's state, others written by the top two layers); regulators (future); which kind of memory lives where; traits, motivation and control (future); what remains for the state layer and the encoder; social interaction and non-game mechanics (proposed); section 13 is the plan of work and the decisions waiting on the owner. | RULED and AGREED lines quote the owner; mark open questions as they are settled, with the date. |
| `BUILD.md` | The build plan: the owner's rulings (section 0), the layout (1), reusable resources (2), the architecture (3), the gen-7 verification checklist (4), the requirement map (5), the proofs E1 to E14 (6), the order of work (7). | Update rows in place with the date; RULED lines quote the owner. |
| `TRAINING_PLAN.md` | The training plan the owner approved: hypotheses, the bar, the method, the teacher certificate T1 to T7, stages S1 to S4, checks C1 to C9, risks. | Status lines say what is true now. |
| `PERCEPTION.md` | The skill layer's input contract (2026-10-19): what agents perceive, as properties and relations from an offline registry of the game's assets, open to mods; the owner's decisions and the build plan. | As `BUILD.md`. |
| `PERCEPTION_ASKS.md` | What the skill layer asks the simulator layer to publish (build step 4 of `PERCEPTION.md`): seven fields with their file references and legal basis, a contract request for the observation's owners. | Mark each ask's status in place, with the date. |
| `IDEAS.md` | Ideas written down at the owner's request, not approved (2026-10-19, night): the core layer's direction (the LLM thinks, the decision layer builds workflows, the neural layer does the basics and blends them like a human, on one shared blackboard: the LLM and the decision layer feed each other, the neural layer takes only the decision layer's requests and reports back to both), the guerrilla design test, behaviour scalars, vague requests, cohort training, cover perception, capability cards. | Add ideas; an idea moves to `BUILD.md` or `TRAINING_PLAN.md` only on the owner's yes. |
| `ARCHITECTURE_RESEARCH.md` | The research behind the request-conditioned generalist, with proofs A1 to A4 and their status. | As `BUILD.md`. |
| `FIDELITY.md` | The gen-7 fidelity ledger: a current-status roll-up above append-only entries. | Append entries; refresh the roll-up in the same commit. |
| `AGENT_BRIEF.md` | Phase A's instruction set (done, 2026-10-17), kept as the record. | Historical: do not extend. |
| `SKILL.md` (gitignored) | Working notes for this folder. | Local. |
