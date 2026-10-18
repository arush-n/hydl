# skill_library/ — Level 1 stand-ins for the composition task

Three skills on the 12-head action surface, enough to measure composition: a pinned pursuit candidate, a
scripted evader and idle. **The real neural skill library is built in `../skill_agent/`** (the lower-layer
session; `../skill_agent/BUILD.md`), over the 139-row catalogue, with contracts, a request stream and
behaviour-cloned bodies. Until it replaces these, every composition number is about the middle layer's
decisions, not the skills.

| File | Lines | What it is |
|---|---:|---|
| `library.py` | 124 | The skill catalogue: an ordered tuple with idle always at index 0 (`idle_goal` selects index 0). |
| `dispatcher.py` | 211 | `SkillDispatcher`, the ADK `LowLevelActor` that runs the active skill under the runtime mask. |
| `pursuit.py` | 177 | The pinned `pursuit_contact` candidate v0001 behind its scoped loader. |
| `evader.py` | 204 | Scripted evader: a fixed rule on the 12 heads, restricted to the mask. |
| `idle.py` | 124 | The physical neutral on every head. |
| `descriptions.py` | 37 | Plain-language descriptions of the skills (`SKILL_DESCRIPTIONS`), what the decision layer and the generator read. |
| `_common.py` | 285 | Head layout, shape and mask checks shared by the skills. |

Gate: `test_na_skills.py`. The skill contracts (commitment, duration, emergency events, the control table
the instances read) are in `../contracts.py` (`SkillContract`, `control_table`, `default_control_table`).
