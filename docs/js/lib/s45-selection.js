// Page 5.2: three learning rules on three actions, the replicator equation, and the mean flow of
// softmax policy gradient. Pure JavaScript, no DOM.
import { rng } from './mlp.js';

// Rock-paper-scissors with payoffs in [0, 1]: win 1, tie 0.5, loss 0. Row = own action.
export const RPS = [[0.5, 0, 1], [1, 0.5, 0], [0, 1, 0.5]];

// Expected payoff of each action. game = { kind: 'fixed', pay: [p1, p2, p3] } or { kind: 'rps' },
// where in rock-paper-scissors the learner plays against opponents who choose like it does.
export function payoffs(game, x) {
  if (game.kind === 'fixed') return game.pay.slice();
  return RPS.map((row) => row.reduce((s, a, j) => s + a * x[j], 0));
}

// Right-hand sides. Replicator: dx_i/dt = x_i (pi_i - mean pi).
export function replicatorRHS(game, x) {
  const p = payoffs(game, x), m = p.reduce((s, v, i) => s + v * x[i], 0);
  return x.map((xi, i) => xi * (p[i] - m));
}
// Softmax policy gradient in the small-step limit: d theta/dt = g with g_i = x_i (pi_i - mean pi),
// so dx_i/dt = x_i (g_i - sum_j x_j g_j).
export function pgRHS(game, x) {
  const g = replicatorRHS(game, x), gm = g.reduce((s, v, i) => s + v * x[i], 0);
  return x.map((xi, i) => xi * (g[i] - gm));
}

// Fourth-order Runge-Kutta from x0 over [0, T]; returns points every dtOut.
export function integrate(rhs, game, x0, T, dt = 0.01, dtOut = 0.05) {
  let x = x0.slice(); const out = [{ t: 0, x: x.slice() }];
  const add = (a, b, h) => a.map((v, i) => v + h * b[i]);
  const every = Math.max(1, Math.round(dtOut / dt));
  const n = Math.round(T / dt);
  for (let s = 1; s <= n; s++) {
    const k1 = rhs(game, x), k2 = rhs(game, add(x, k1, dt / 2)), k3 = rhs(game, add(x, k2, dt / 2)), k4 = rhs(game, add(x, k3, dt));
    x = x.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    const z = x.reduce((a, b) => a + b, 0); x = x.map((v) => Math.max(v, 0) / z);
    if (s % every === 0) out.push({ t: s * dt, x: x.slice() });
  }
  return out;
}

const softmax = (th) => { const m = Math.max(...th); const e = th.map((v) => Math.exp(v - m)); const z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
const draw = (r, x) => { const u = r(); return u < x[0] ? 0 : u < x[0] + x[1] ? 1 : 2; };

// One run of a learning rule with step size h for T units of time (T / h steps).
// Payoffs must lie in [0, 1] for Cross's rule. Returns points every dtOut units of time.
export function learn(rule, game, x0, h, T, seed, dtOut = 0.05) {
  const r = rng(seed);
  let x = x0.slice(), th = x0.map((v) => Math.log(v));
  const out = [{ t: 0, x: x.slice() }];
  const n = Math.round(T / h), every = Math.max(1, Math.round(dtOut / h));
  for (let s = 1; s <= n; s++) {
    const i = draw(r, x);
    const lastOpp = game.kind === 'fixed' ? -1 : draw(r, x);
    const u = game.kind === 'fixed' ? game.pay[i] : RPS[i][lastOpp];
    if (rule === 'cross') {
      // Cross (1973): move toward the chosen action in proportion to its payoff.
      x = x.map((v, j) => v + h * u * ((j === i ? 1 : 0) - v));
    } else if (rule === 'exp') {
      // Exponential weights: every action's score grows by the payoff it would have earned this round
      // (against the same opponent); probabilities are a softmax of the scores.
      const j = game.kind === 'fixed' ? -1 : lastOpp;
      th = th.map((v, k) => v + h * (j < 0 ? game.pay[k] : RPS[k][j]));
      x = softmax(th);
    } else {
      // Softmax policy gradient (REINFORCE): theta += h u (e_i - x).
      th = th.map((v, j) => v + h * u * ((j === i ? 1 : 0) - x[j]));
      x = softmax(th);
    }
    if (s % every === 0) out.push({ t: s * h, x: x.slice() });
  }
  return out;
}

// Largest distance between a learning path and a reference curve at matching times.
// The reference curve must be sampled every dtRef units of time, starting at 0.
export function maxGap(path, ref, dtRef = 0.05) {
  let g = 0;
  for (const p of path) {
    const k = Math.round(p.t / dtRef);
    if (k >= ref.length) break;
    const d = Math.hypot(...p.x.map((v, i) => v - ref[k].x[i]));
    if (d > g) g = d;
  }
  return g;
}

// Largest distance from a point of the path to the nearest point of a reference curve, ignoring
// time: how far the path strays from the curve's shape.
export function curveGap(path, ref) {
  let g = 0;
  for (const p of path) {
    let m = Infinity;
    for (const q of ref) { const d = Math.hypot(p.x[0] - q.x[0], p.x[1] - q.x[1], p.x[2] - q.x[2]); if (d < m) m = d; }
    if (m > g) g = m;
  }
  return g;
}

// The tool's readout and the page's numbers: run `runs` learning runs (seeds 1, 2, ...) and return the
// paths and the average of curveGap from the replicator curve (and from the policy-gradient curve).
// Reference curves run to 3T so a slow path is measured against the whole curve.
export function runAndMeasure(rule, game, x0, h, T, runs) {
  const rep = integrate(replicatorRHS, game, x0, 3 * T, 0.01, 0.02);
  const pg = rule === 'pg' ? integrate(pgRHS, game, x0, 3 * T, 0.01, 0.02) : null;
  const paths = [];
  let gRep = 0, gPG = 0;
  for (let s = 1; s <= runs; s++) {
    const p = learn(rule, game, x0, h, T, s);
    paths.push(p);
    const sub = p.filter((_, i) => i % 4 === 0);
    gRep += curveGap(sub, rep) / runs;
    if (pg) gPG += curveGap(sub, pg) / runs;
  }
  return { paths, gRep, gPG: pg ? gPG : null };
}
