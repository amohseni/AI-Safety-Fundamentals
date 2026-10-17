// The cleaning robot of page 8.2, as a deterministic MDP built on gridworld.js.
// Pure JavaScript, no DOM, so Node tests can import it.
//
// The room is a 5 by 4 grid. A camera on the wall above the top-left cell sees the three left columns;
// a robot standing in the top-left cell blocks its view entirely. The robot starts on its dock.
// Actions: wait; clean the mess under the robot (takes 3 steps); move up, right, down, or left;
// drag the mess under the robot one cell along with it; end the shift (only on the dock).
// Moving into the bin tips it, spilling trash onto the robot's cell. Moving into the vase breaks it.
// Moving into the plant knocks it over, leaving soil (a mess) where it stood.
// The shift lasts T steps.
import { finiteHorizon, rollout, grid, DIRS } from './gridworld.js';

export const W = 5, H = 4, T = 30, CLEAN_STEPS = 3, MAX_MESSES = 3;
export const G = grid(W, H);
const at = (x, y) => G.idx(x, y);
export const LAYOUT = {
  dock: at(0, 2), block: at(0, 0), bin: at(4, 0), vase: at(3, 2), plant: at(4, 3),
  messes: [at(1, 1), at(2, 3)],
  inView: (i) => G.xy(i)[0] <= 2,
};
export const ACTIONS = ['wait', 'clean', 'up', 'right', 'down', 'left', 'drag up', 'drag right', 'drag down', 'drag left', 'end shift'];
export const A_WAIT = 0, A_CLEAN = 1, A_MOVE = 2, A_DRAG = 6, A_END = 10;

// Parts the reader can put in the reward, with what each counts.
export const PARTS = [
  { id: 'cleaned', name: 'Each mess cleaned' },
  { id: 'step', name: 'Each step of work' },
  { id: 'vase', name: 'Breaking the vase' },
  { id: 'clear', name: 'Each step with no mess visible to the camera' },
  { id: 'blocked', name: 'Each step the camera is blocked' },
  { id: 'tip', name: 'Each time the bin is tipped' },
];

const popcount = (m) => { let c = 0; while (m) { c += m & 1; m >>>= 1; } return c; };
const has = (m, i) => ((m >>> i) & 1) === 1;

export function visibleMesses(s) {
  if (s.pos === LAYOUT.block) return 0;
  let c = 0;
  for (let i = 0; i < G.n; i++) if (has(s.mask, i) && LAYOUT.inView(i)) c++;
  return c;
}

// Builds the MDP for a room with or without the plant. States are coded as integers:
// code = (((mask * 20 + pos) * 2 + vase) * 2 + plant) * 2 + done, where bit i of mask marks a mess in cell i.
export function decode(code) {
  const done = code % 2; code = (code - done) / 2;
  const plant = code % 2; code = (code - plant) / 2;
  const vase = code % 2; code = (code - vase) / 2;
  const pos = code % G.n; const mask = (code - pos) / G.n;
  return { mask, pos, vase, plant, done };
}
const encode = (mask, pos, vase, plant, done) => (((mask * G.n + pos) * 2 + vase) * 2 + plant) * 2 + done;

export function buildRoom({ plant = false } = {}) {
  let mask0 = 0;
  for (const m of LAYOUT.messes) mask0 |= 1 << m;
  const nA = ACTIONS.length;
  const codes = [encode(mask0, LAYOUT.dock, 1, plant ? 1 : 0, 0)];
  const index = new Map([[codes[0], 0]]);
  const next = [], dur = [], fs = { cleaned: [], step: [], vase: [], clear: [], blocked: [], tip: [], plant: [] };
  const view = []; for (let i = 0; i < G.n; i++) if (LAYOUT.inView(i)) view.push(i);
  // Per-step parts are summed over the steps an action takes, discounted like the rewards that follow.
  const span = [0, 1, 1 + GAMMA, 1 + GAMMA + GAMMA * GAMMA];
  const push = (to, d, f) => {
    let j = -1;
    if (to !== null) { j = index.get(to); if (j === undefined) { j = codes.length; codes.push(to); index.set(to, j); } }
    next.push(j); dur.push(d);
    for (const k in fs) fs[k].push(f ? f[k] || 0 : 0);
  };
  for (let si = 0; si < codes.length; si++) {
    const { mask, pos, vase, plant: pl, done } = decode(codes[si]);
    if (done) { for (let a = 0; a < nA; a++) push(null, 1, null); continue; }
    let vis = 0;
    if (pos !== LAYOUT.block) for (const i of view) if ((mask >>> i) & 1) vis++;
    const nm = popcount(mask), here = ((mask >>> pos) & 1) === 1;
    const base = (d) => ({ step: span[d], clear: vis === 0 ? span[d] : 0, blocked: pos === LAYOUT.block ? span[d] : 0 });
    const [x, y] = G.xy(pos);
    for (let a = 0; a < nA; a++) {
      if (a === A_WAIT) { push(codes[si], 1, base(1)); continue; }
      if (a === A_END) { push(pos === LAYOUT.dock ? encode(mask, pos, vase, pl, 1) : null, 1, null); continue; }
      if (a === A_CLEAN) {
        if (here) push(encode((mask & ~(1 << pos)) >>> 0, pos, vase, pl, 0), CLEAN_STEPS, { ...base(CLEAN_STEPS), cleaned: 1 });
        else push(null, 1, null);
        continue;
      }
      const isDrag = a >= A_DRAG, [dx, dy] = DIRS[isDrag ? a - A_DRAG : a - A_MOVE];
      const nx = x + dx, ny = y + dy;
      if (!G.inside(nx, ny)) { push(null, 1, null); continue; }
      const t = at(nx, ny);
      const solid = t === LAYOUT.bin || (t === LAYOUT.vase && vase) || (t === LAYOUT.plant && pl);
      if (isDrag) {
        if (!here || solid || ((mask >>> t) & 1)) push(null, 1, null);
        else push(encode(((mask & ~(1 << pos)) | (1 << t)) >>> 0, t, vase, pl, 0), 1, base(1));
        continue;
      }
      if (t === LAYOUT.bin) {
        if (here || nm >= MAX_MESSES) push(null, 1, null);
        else push(encode((mask | (1 << pos)) >>> 0, pos, vase, pl, 0), 1, { ...base(1), tip: 1 });
      } else if (t === LAYOUT.vase && vase) push(encode(mask, t, 0, pl, 0), 1, { ...base(1), vase: 1 });
      else if (t === LAYOUT.plant && pl) {
        if (nm >= MAX_MESSES) push(null, 1, null);
        else push(encode((mask | (1 << t)) >>> 0, pos, vase, 0, 0), 1, { ...base(1), plant: 1 });
      } else push(encode(mask, t, vase, pl, 0), 1, base(1));
    }
  }
  const nS = codes.length;
  const absorbing = new Uint8Array(nS);
  const states = codes.map(decode);
  states.forEach((s, i) => { if (s.done) absorbing[i] = 1; });
  const F = {};
  for (const k in fs) F[k] = Float64Array.from(fs[k]);
  return { nS, nA, next: Int32Array.from(next), dur: Uint8Array.from(dur), states, index, absorbing, F, plantStart: plant };
}

// Reward table for weights { cleaned, step, vase, clear, blocked, tip }.
export function rewardTable(room, weights) {
  const R = new Float64Array(room.nS * room.nA);
  for (const p of PARTS) {
    const w = weights[p.id] || 0;
    if (!w) continue;
    const f = room.F[p.id];
    for (let i = 0; i < R.length; i++) R[i] += w * f[i];
  }
  return R;
}

// A slight preference for sooner reward breaks ties toward acting early.
export const GAMMA = 0.999;

export function solve(room, weights) {
  const R = rewardTable(room, weights);
  const sol = finiteHorizon(room, R, T, { gamma: GAMMA, absorbing: room.absorbing });
  const path = rollout(room, sol.policy, 0, T);
  return { R, sol, path, summary: summarize(room, weights, path) };
}

// What happened over the shift: reward earned (undiscounted), events, and the true task score.
export function summarize(room, weights, path) {
  let reward = 0;
  const count = { cleaned: 0, tip: 0, vase: 0, plant: 0, blocked: 0, dragged: 0, hiddenByDrag: 0 };
  for (const { s, a } of path) {
    if (a < 0) break;
    const i = s * room.nA + a, d = room.dur[i];
    for (const p of PARTS) {
      const f = room.F[p.id][i];
      // Per-step parts are stored discounted over the action's steps; count them as whole steps here.
      reward += (weights[p.id] || 0) * (p.id === 'step' || p.id === 'clear' || p.id === 'blocked' ? (f > 0 ? d : 0) : f);
    }
    const F = room.F;
    count.cleaned += F.cleaned[i]; count.tip += F.tip[i]; count.vase += F.vase[i]; count.plant += F.plant[i];
    if (room.states[s].pos === LAYOUT.block) count.blocked += room.dur[i];
    if (a >= A_DRAG && a < A_END) count.dragged++;
  }
  const last = room.states[path[path.length - 1].s];
  const messesLeft = popcount(last.mask);
  let outOfView = 0;
  for (let i = 0; i < G.n; i++) if (has(last.mask, i) && !LAYOUT.inView(i)) outOfView++;
  const truth = -messesLeft - 10 * (last.vase ? 0 : 1) - 10 * (room.plantStart && !last.plant ? 1 : 0);
  return { reward, truth, messesLeft, outOfView, vaseBroken: !last.vase, plantKnocked: room.plantStart && !last.plant,
    ended: !!last.done, endsBlocking: last.pos === LAYOUT.block, ...count };
}

export const messCells = (s) => { const out = []; for (let i = 0; i < G.n; i++) if (has(s.mask, i)) out.push(i); return out; };
