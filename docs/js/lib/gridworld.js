// Gridworld and graph solver, shared by pages 7.3, 8.2, 8.3, and 8.4.
// Pure JavaScript; the solvers use no DOM, so Node tests can import this file.
// The drawing helpers at the end take a canvas and read colors from the CSS tokens.
//
// API
//
// Deterministic MDPs in table form
//   An MDP is { nS, nA, next, dur }:
//     next: Int32Array(nS * nA), next[s * nA + a] is the state reached by action a in state s, or -1 if a is unavailable;
//     dur:  Uint8Array(nS * nA) or null, the number of time steps the action takes (default 1).
//   Rewards are passed separately as R: Float64Array(nS * nA), the reward for taking a in s.
//
//   enumerate({ start, nA, step, key }) -> { nS, nA, next, dur, states, index, info }
//     Builds the MDP of all states reachable from `start` by breadth-first search.
//     step(state, a) returns null when a is unavailable, or an object { next, dur = 1, ...extra };
//     key(state) returns a number or string that identifies a state.
//     states[i] is the i-th state object (states[0] is start), index maps key -> i,
//     info[s * nA + a] keeps the object step returned, so callers can read features from it.
//
//   valueIteration(mdp, R, { gamma = 0.95, tol = 1e-9, maxIter = 10000 }) -> { V, Q, iters }
//     Discounted optimal values. An action of duration d discounts what follows by gamma^d.
//
//   finiteHorizon(mdp, R, T, { gamma = 1, absorbing = null }) -> { V, Q0, policy(s, t) }
//     Optimal non-stationary policy for a horizon of T steps (an action of duration d needs d steps left).
//     V is a Float64Array((T + 1) * nS) indexed by t * nS + s, with V at t = T equal to 0.
//     absorbing: optional Uint8Array(nS); an absorbing state earns nothing from then on.
//     policy(s, t) returns the best action (lowest index among ties within 1e-9), or -1 if none is available.
//
//   bestActions(Qrow, tol = 1e-9) -> array of the actions whose value is within tol of the best.
//
//   rollout(mdp, policy, s0, T) -> [{ s, a, t }, ...]
//     Follows policy(s, t) from s0 until T steps have passed or no action is available.
//
// Graph MDPs with rewards on states (page 8.4)
//   graphValues(succ, r, gamma) -> Float64Array V, exact (policy iteration)
//     succ[s] lists the states one move from s (include s itself to allow staying);
//     V(s) = r(s) + gamma * max over s' in succ[s] of V(s'). Every state must have at least one successor.
//   power(succ, samples, gamma) -> Float64Array, Turner et al.'s POWER, (1 - gamma) / gamma * mean of (V(s) - r(s)).
//
// Grid helpers
//   DIRS: [[0, -1], [1, 0], [0, 1], [-1, 0]] for up, right, down, left (y grows downward); DIR_NAMES matches.
//   grid(width, height) -> { width, height, n, idx(x, y), xy(i), inside(x, y) }
//
// Drawing helpers (browser only)
//   fitCanvas(canvas, cssWidth, cssHeight) -> ctx, sized for the device pixel ratio and scaled to CSS pixels.
//   cellLayout(cssWidth, width, height, { pad = 1 }) -> { cell, ox, oy, w, h, center(x, y), rect(x, y) }
//   drawGrid(ctx, layout, g, { wall = () => false }) draws the cells as thin rules and walls as filled squares.

// ---------------- Deterministic MDPs ----------------

export function enumerate({ start, nA, step, key }) {
  const states = [start], index = new Map([[key(start), 0]]);
  const nextList = [], durList = [], info = [];
  for (let i = 0; i < states.length; i++) {
    const s = states[i];
    for (let a = 0; a < nA; a++) {
      const res = step(s, a);
      if (!res) { nextList.push(-1); durList.push(1); info.push(null); continue; }
      const k = key(res.next);
      let j = index.get(k);
      if (j === undefined) { j = states.length; states.push(res.next); index.set(k, j); }
      nextList.push(j); durList.push(res.dur || 1); info.push(res);
    }
  }
  const nS = states.length;
  return { nS, nA, next: Int32Array.from(nextList), dur: Uint8Array.from(durList), states, index, info };
}

export function valueIteration(mdp, R, { gamma = 0.95, tol = 1e-9, maxIter = 10000 } = {}) {
  const { nS, nA, next, dur } = mdp;
  let V = new Float64Array(nS), W = new Float64Array(nS);
  const Q = new Float64Array(nS * nA);
  const gd = new Float64Array(nS * nA);
  for (let i = 0; i < nS * nA; i++) gd[i] = Math.pow(gamma, dur ? dur[i] : 1);
  let iters = 0;
  for (; iters < maxIter; iters++) {
    let delta = 0;
    for (let s = 0; s < nS; s++) {
      let best = -Infinity;
      for (let a = 0; a < nA; a++) {
        const i = s * nA + a, j = next[i];
        if (j < 0) { Q[i] = -Infinity; continue; }
        const q = R[i] + gd[i] * V[j];
        Q[i] = q;
        if (q > best) best = q;
      }
      if (best === -Infinity) best = 0;
      W[s] = best;
      const d = Math.abs(best - V[s]);
      if (d > delta) delta = d;
    }
    [V, W] = [W, V];
    if (delta < tol) { iters++; break; }
  }
  return { V, Q, iters };
}

export function finiteHorizon(mdp, R, T, { gamma = 1, absorbing = null } = {}) {
  const { nS, nA, next, dur } = mdp;
  const V = new Float64Array((T + 1) * nS);
  const gp = []; for (let d = 0; d <= 255; d++) gp.push(Math.pow(gamma, d));
  for (let t = T - 1; t >= 0; t--) {
    const row = t * nS;
    for (let s = 0; s < nS; s++) {
      if (absorbing && absorbing[s]) { V[row + s] = 0; continue; }
      let best = -Infinity;
      for (let a = 0; a < nA; a++) {
        const i = s * nA + a, j = next[i];
        if (j < 0) continue;
        const d = dur ? dur[i] : 1;
        if (t + d > T) continue;
        const q = R[i] + gp[d] * V[(t + d) * nS + j];
        if (q > best) best = q;
      }
      V[row + s] = best === -Infinity ? 0 : best;
    }
  }
  const qAt = (s, a, t) => {
    const i = s * nA + a, j = next[i];
    if (j < 0) return -Infinity;
    const d = dur ? dur[i] : 1;
    if (t + d > T) return -Infinity;
    return R[i] + gp[d] * V[(t + d) * nS + j];
  };
  const policy = (s, t) => {
    if (t >= T || (absorbing && absorbing[s])) return -1;
    let best = -Infinity, ba = -1;
    for (let a = 0; a < nA; a++) { const q = qAt(s, a, t); if (q > best + 1e-9) { best = q; ba = a; } }
    return ba;
  };
  return { V, qAt, policy };
}

export function bestActions(Qrow, tol = 1e-9) {
  let best = -Infinity;
  for (const q of Qrow) if (q > best) best = q;
  const out = [];
  Qrow.forEach((q, a) => { if (q >= best - tol && q > -Infinity) out.push(a); });
  return out;
}

export function rollout(mdp, policy, s0, T) {
  const path = [];
  let s = s0, t = 0;
  while (t < T) {
    const a = policy(s, t);
    if (a < 0) break;
    path.push({ s, a, t });
    const i = s * mdp.nA + a;
    t += mdp.dur ? mdp.dur[i] : 1;
    s = mdp.next[i];
  }
  path.push({ s, a: -1, t });
  return path;
}

// ---------------- Graph MDPs with state rewards ----------------

export function graphValues(succ, r, gamma, { maxIter = 1000 } = {}) {
  // Policy iteration with exact evaluation: under a deterministic policy every path ends in a cycle,
  // so values follow from the cycle's discounted reward sum and then back along the path.
  const n = succ.length;
  const pi = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    let b = succ[s].length ? succ[s][0] : s;
    for (const t of succ[s]) if (r[t] > r[b]) b = t;
    pi[s] = b;
  }
  const V = new Float64Array(n);
  const state = new Uint8Array(n); // 0 unvisited, 1 on the current path, 2 done
  for (let it = 0; it < maxIter; it++) {
    state.fill(0);
    for (let s0 = 0; s0 < n; s0++) {
      if (state[s0]) continue;
      const path = [];
      let s = s0;
      while (state[s] === 0) { state[s] = 1; path.push(s); s = pi[s]; }
      let k = path.length;
      if (state[s] === 1) {
        // s is on the current path: the cycle is path[c..].
        const c = path.indexOf(s), m = path.length - c;
        for (let i = 0; i < m; i++) {
          let sum = 0, g = 1;
          for (let j = 0; j < m; j++) { sum += g * r[path[c + ((i + j) % m)]]; g *= gamma; }
          V[path[c + i]] = sum / (1 - g);
          state[path[c + i]] = 2;
        }
        k = c;
      }
      for (let i = k - 1; i >= 0; i--) { const u = path[i]; V[u] = r[u] + gamma * V[pi[u]]; state[u] = 2; }
    }
    let changed = false;
    for (let s = 0; s < n; s++) {
      let b = pi[s], bv = V[b];
      for (const t of succ[s]) if (V[t] > bv + 1e-12) { b = t; bv = V[t]; }
      if (b !== pi[s]) { pi[s] = b; changed = true; }
    }
    if (!changed) break;
  }
  return V;
}

export function power(succ, samples, gamma) {
  const n = succ.length, P = new Float64Array(n);
  for (const r of samples) {
    const V = graphValues(succ, r, gamma);
    for (let s = 0; s < n; s++) P[s] += V[s] - r[s];
  }
  for (let s = 0; s < n; s++) P[s] *= (1 - gamma) / gamma / samples.length;
  return P;
}

// ---------------- Grid helpers ----------------

export const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
export const DIR_NAMES = ['up', 'right', 'down', 'left'];

export function grid(width, height) {
  return {
    width, height, n: width * height,
    idx: (x, y) => y * width + x,
    xy: (i) => [i % width, Math.floor(i / width)],
    inside: (x, y) => x >= 0 && y >= 0 && x < width && y < height,
  };
}

// ---------------- Drawing helpers ----------------

export function fitCanvas(canvas, cssWidth, cssHeight) {
  const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function cellLayout(cssWidth, width, height, { pad = 1, maxCell = 80 } = {}) {
  const cell = Math.min(maxCell, Math.floor((cssWidth - 2 * pad) / width));
  const w = cell * width, h = cell * height;
  const ox = pad, oy = pad;
  return {
    cell, ox, oy, w, h, cssHeight: h + 2 * pad,
    center: (x, y) => [ox + (x + 0.5) * cell, oy + (y + 0.5) * cell],
    rect: (x, y) => [ox + x * cell, oy + y * cell, cell, cell],
  };
}

export function drawGrid(ctx, L, g, { wall = () => false, colors }) {
  ctx.lineWidth = 1;
  ctx.strokeStyle = colors.rule;
  for (let y = 0; y < g.height; y++) for (let x = 0; x < g.width; x++) {
    const [rx, ry, c] = L.rect(x, y);
    if (wall(x, y)) { ctx.fillStyle = colors.quiet; ctx.fillRect(rx, ry, c, c); }
    else ctx.strokeRect(rx + 0.5, ry + 0.5, c - 1, c - 1);
  }
}
