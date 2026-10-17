// Models for page 8.4. Pure JavaScript, no DOM, so Node tests can import them.
//
// 1. Random goals on a graph (Turner et al. 2021). A graph has nodes and directed edges; every node may also stay
//    where it is. A reward function gives each node a number, drawn independently from one distribution.
//    For each sampled reward function the optimal policy is computed exactly (gridworld.graphValues), and we
//    record which first move from the start node is optimal (ties split evenly) and each node's POWER.
// 2. An incomplete proxy (Zhuang and Hadfield-Menell 2020). L attributes share a budget; the person's utility
//    counts all L, the proxy only the first J. The agent climbs the proxy by projected gradient steps.
import { graphValues } from './gridworld.js';
import { rng, gaussian } from './mlp.js';

export function defaultGraph() {
  // Positions are fractions of the drawing's width and height; label says where the name goes
// (above, aboveleft, left, or right).
  const nodes = [
    { id: 'start', name: 'start', x: 0.12, y: 0.6, label: 'left' },
    { id: 'dead', name: 'dead end', x: 0.26, y: 0.08, label: 'right' },
    { id: 'hub', name: 'hub', x: 0.5, y: 0.6, label: 'aboveleft' },
  ];
  const edges = [['start', 'dead'], ['start', 'hub']];
  for (let i = 0; i < 5; i++) {
    const a = (-75 + i * 37.5) * (Math.PI / 180);
    const id = `a${i + 1}`;
    nodes.push({ id, name: '', x: 0.5 + 0.4 * Math.cos(a), y: 0.6 + 0.42 * Math.sin(a), pa: a });
    edges.push(['hub', id], [id, 'hub']);
  }
  return { nodes, edges };
}
export const SHUTDOWN = { id: 'off', name: 'shutdown', x: 0.3, y: 1, label: 'left' };

// Successor lists (with staying allowed everywhere) for a graph, plus the index of each node id.
export function successors(graph) {
  const ix = new Map(graph.nodes.map((n, i) => [n.id, i]));
  const succ = graph.nodes.map((_, i) => [i]);
  for (const [a, b] of graph.edges) if (ix.has(a) && ix.has(b) && a !== b) succ[ix.get(a)].push(ix.get(b));
  return { succ, ix };
}

export const DISTS = {
  uniform: { name: 'Uniform', draw: (r, n) => Float64Array.from({ length: n }, () => r()) },
  normal: { name: 'Normal', draw: (r, n) => Float64Array.from({ length: n }, () => gaussian(r)) },
  sparse: { name: 'Sparse', draw: (r, n) => { const v = new Float64Array(n); v[Math.floor(r() * n)] = 1; return v; } },
};

// Draws `count` reward vectors over `n` nodes. The same seed and n give the same rewards.
export function sampleRewards(dist, n, count = 1000, seed = 1) {
  const r = rng(seed * 7919 + n);
  return Array.from({ length: count }, () => DISTS[dist].draw(r, n));
}

// For each sampled reward: the optimal first moves from the start. Returns the share of samples for which each
// successor of the start is optimal (ties split evenly), and POWER for every node.
export function analyze(graph, samples, gamma, startId = 'start') {
  const { succ, ix } = successors(graph);
  const s0 = ix.get(startId), n = succ.length;
  const share = new Map(succ[s0].map((t) => [t, 0]));
  const P = new Float64Array(n);
  for (const r of samples) {
    const V = graphValues(succ, r, gamma);
    let best = -Infinity;
    for (const t of succ[s0]) if (V[t] > best) best = V[t];
    const tied = succ[s0].filter((t) => V[t] >= best - 1e-9 * Math.max(1, Math.abs(best)));
    for (const t of tied) share.set(t, share.get(t) + 1 / tied.length);
    for (let s = 0; s < n; s++) P[s] += V[s] - r[s];
  }
  for (const [t, v] of share) share.set(t, v / samples.length);
  for (let s = 0; s < n; s++) P[s] *= (1 - gamma) / gamma / samples.length;
  return { share, power: P, succ, ix };
}

// ---- Incomplete proxy ----
export const SHAPES = {
  diminishing: { name: 'Diminishing returns', u: (x) => 1 - Math.exp(-x), du: (x) => Math.exp(-x) },
  linear: { name: 'Linear', u: (x) => x, du: () => 1 },
};

// Projection onto { x >= floor, sum(x) <= budget }.
function project(y, floor, budget) {
  const x = y.map((v) => Math.max(floor, v));
  if (x.reduce((a, b) => a + b, 0) <= budget) return x;
  let lo = 0, hi = Math.max(...y) - floor + 1;
  for (let k = 0; k < 100; k++) {
    const mid = (lo + hi) / 2;
    const s = y.reduce((a, v) => a + Math.max(floor, v - mid), 0);
    if (s > budget) lo = mid; else hi = mid;
  }
  return y.map((v) => Math.max(floor, v - hi));
}

// Runs the agent. Attributes start at 0; the budget is L (so there are unused resources at first).
// Each step moves a distance `eta` along the proxy's gradient, then projects back into the feasible set.
// Between equally good referenced attributes the step is shared equally, so a symmetric start stays symmetric.
// Stops when no attribute moves by more than 1e-7 in a step, or after maxSteps.
export function allocate({ L = 4, J = 3, shape = 'diminishing', floor = -3, maxSteps = 4000, eta = 0.05 } = {}) {
  const sh = SHAPES[shape], budget = L;
  let x = new Array(L).fill(0);
  const U = (v) => v.reduce((a, xi) => a + sh.u(xi), 0);
  const P = (v) => v.slice(0, J).reduce((a, xi) => a + sh.u(xi), 0);
  const traj = [{ x: x.slice(), true: U(x), proxy: P(x) }];
  for (let t = 0; t < maxSteps; t++) {
    const g = x.map((xi, i) => (i < J ? sh.du(xi) : 0));
    const norm = Math.hypot(...g) || 1;
    const nx = project(x.map((xi, i) => xi + (eta * g[i]) / norm), floor, budget);
    const moved = Math.max(...nx.map((v, i) => Math.abs(v - x[i])));
    x = nx;
    traj.push({ x: x.slice(), true: U(x), proxy: P(x) });
    if (moved < 1e-7) break;
  }
  return traj;
}
