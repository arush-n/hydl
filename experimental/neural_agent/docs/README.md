# docs/ — the design record of experimental/neural_agent

Read in this order if you are new: `MIDDLE_LAYER.md` (where things stand), `SYSTEM_ONE.md` sections 8.12
and 9.9 to 9.13 (the decisions that shape the middle layer today), `LEDGER.md` (every measurement with its
command and verdict), then the rest as needed. `CURRENT_STATE.md` is the dated snapshot of the whole package.

| Document | Purpose | Edit rule |
|---|---|---|
| `CURRENT_STATE.md` | Dated snapshot (2026-10-18): what is built, measured, running and pending, per level and for the simulator port. | Rewrite at each snapshot; keep the date. |
| `MIDDLE_LAYER.md` | The consolidated design, status table (measured / built / designed / failed per gate) and plan of the decision layer. The entry point. | Update rows in place with the date. |
| `SYSTEM_ONE.md` | Our own System One decision model: what a JEV is, precedents, the design and formulas (M1 to M13), the owner's rulings, the reading experiments and their verdicts (sections 8 and 9). Large, by design. | Append dated subsections; do not rewrite history. |
| `LEDGER.md` | Append-only experiment ledger: baseline, command, result, verdict, each entry dated. | Append only, with `>>`; never edit an entry. |
| `THESIS.md` | Research question, the three levels, the two state objects, hypotheses H1 to H16, milestones, metrics, evidence status. | Quotes ADR-001 (local-only record) verbatim; tests check the quotes. |
| `ARCHITECTURE.md` | Package map, data flow between levels, the JEV contract, the four-arm protocol, implemented-versus-claimed checklist, provenance table with sha256 pins. | Pins must match the files (`test_na_docs.py`). |
| `FORMULAS.md` | Every formula the package applies, sections A to M, with sources. | Numbered; cited from the other docs. |
| `ORCHESTRATION.md` | The orchestration rules the ADK applies to goals (lowering E_t, manager due, commitment, switch, action age, the latency budget) with confirmed/inferred tags; moved from the top-level README on 2026-10-18. | As FORMULAS. |
| `REFERENCES.md` | Citations: section 1 copies ADR-001 section 11 verbatim; section 3 the later references (DAgger, negation, structured extraction, ...). | Every `[KEY]` cited in the docs must be defined here (`test_na_docs.py`). |
| `SKILL.md` (gitignored) | The documentation pass's working notes. | Local. |

Rules that apply to every document here: the ADR-001 decision record is local-only and pinned by sha256
(`ARCHITECTURE.md` section 8); `contracts.py` is pinned the same way; private inputs under `agent_skills/`
are referenced by path, never copied; numbers come with their command and date; a claim is marked
measured, built, designed or failed, never implied.

Tests: `tests/experimental/neural_agent/test_na_docs.py` (citations defined, ADR and contracts pins,
quoted passages verbatim, documented widths), `test_na_docs_folders.py` (parent documents pin the hashes;
owned folders carry a dated `SKILL.md`).
