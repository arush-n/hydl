/* Shared state and a dirty-flag frame scheduler.
 *
 * The old client redrew the 3D scene, the timeline and all three charts on
 * every scrub event. Scrubbing fires continuously, so dragging the slider
 * re-plotted charts whose data had not changed -- hundreds of times a second.
 *
 * Here each surface declares what it depends on. Moving the playhead marks
 * only `frame`; loading a run marks `data`. Work is coalesced into one
 * requestAnimationFrame, so a burst of input costs one repaint, not thirty.
 */

export const TPS = 30;

/* Terminal job states. MUST stay in step with `DONE` in
   console/core/execution/jobs.py -- a state missing here makes the UI poll a
   run that already ended, and a state missing there makes the server think the
   device is still held. "stopped" is a run a training guardrail ended early;
   it used to be reported as "finished", which made a run that died at update 3
   of 256 indistinguishable from one that completed. */
export const TERMINAL_JOB_STATES = ["finished", "failed", "cancelled", "stopped"];

export const isJobLive = (job) => !TERMINAL_JOB_STATES.includes(job?.status);

/* Lives here rather than in a panel because more than one panel shows an
   elapsed time, and two copies would drift in formatting. */
export function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

export const state = {
  data: null,     // last /api/run payload
  step: 0,        // playhead
  playing: false,
  history: [],
  runs: [],       // stored artifacts, newest first
  profiles: [],
  activeProfileId: "", // scopes archived runs without guessing agent ownership
  jobs: [],       // training jobs this server knows about
  activeJobId: "",
  jobMetrics: [], // per-update metrics for the job on screen
  jobMetricsFor: null,
  jobReplay: null,
  jobReplayFor: null,
  jobReplayLoading: null,
  jobReplayStep: 0,
  liveConnected: false,
  filters: new Set(),   // event kinds hidden
};

const surfaces = [];
let dirty = new Set();
let queued = false;

/** Register a surface. `deps` are the state keys that should redraw it. */
export function surface(name, deps, draw) {
  surfaces.push({ name, deps: new Set(deps), draw });
}

/** Mark one or more aspects changed; the repaint is coalesced. */
export function invalidate(...aspects) {
  for (const a of aspects) dirty.add(a);
  if (queued) return;
  queued = true;
  requestAnimationFrame(flush);
}

function flush() {
  queued = false;
  const changed = dirty;
  dirty = new Set();
  for (const s of surfaces) {
    for (const dep of s.deps) {
      if (changed.has(dep)) {
        try { s.draw(); } catch (err) { console.error(`[${s.name}]`, err); }
        break;
      }
    }
  }
}

export function setData(payload) {
  state.data = payload;
  state.step = 0;
  invalidate("data", "frame");
}

export function setStep(i) {
  const n = state.data ? state.data.steps : 1;
  const next = Math.max(0, Math.min(i, n - 1));
  if (next === state.step) return;
  state.step = next;
  invalidate("frame");
}

/* What to call the two actors. The trajectory names them by slot -- `agent_*`
   and `target_*` -- because the same sim runs duels, dodge drills and pursuit.
   A pursuit run declares which side the learner plays, so read the role off the
   run instead of calling every learner "agent". Runs that declare nothing keep
   the slot names, which is the only honest label for them. */
export function roles() {
  const objective = state.data && state.data.spec && state.data.spec.objective;
  if (objective === "evade") return { learner: "evader", opponent: "chaser" };
  if (objective === "pursue") return { learner: "chaser", opponent: "evader" };
  return { learner: "agent", opponent: "target" };
}

export const $ = (id) => document.getElementById(id);
