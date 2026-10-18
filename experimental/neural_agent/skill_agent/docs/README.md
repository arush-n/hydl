# skill_agent/docs/ — the lower layer's design record

Read `BUILD.md` first. Paths in these documents are relative to `skill_agent/` (for example `runtime/combiner.py`,
`docs/BUILD.md`); files outside the package are cited by repo path. A measurement goes into
`experimental/neural_agent/docs/LEDGER.md` (append-only) before it is claimed here.

| Document | Purpose | Edit rule |
|---|---|---|
| `BUILD.md` | The build plan: the owner's rulings (section 0), the layout (1), reusable resources (2), the architecture (3), the gen-7 verification checklist (4), the requirement map (5), the proofs E1 to E14 (6), the order of work (7). | Update rows in place with the date; RULED lines quote the owner. |
| `TRAINING_PLAN.md` | The training plan the owner approved: hypotheses, the bar, the method, the teacher certificate T1 to T7, stages S1 to S4, checks C1 to C9, risks. | Status lines say what is true now. |
| `ARCHITECTURE_RESEARCH.md` | The research behind the request-conditioned generalist, with proofs A1 to A4 and their status. | As `BUILD.md`. |
| `FIDELITY.md` | The gen-7 fidelity ledger: a current-status roll-up above append-only entries. | Append entries; refresh the roll-up in the same commit. |
| `AGENT_BRIEF.md` | Phase A's instruction set (done, 2026-10-17), kept as the record. | Historical: do not extend. |
| `SKILL.md` (gitignored) | Working notes for this folder. | Local. |
