// The page 7.3 policy network and the interventions on its first hidden layer.
// Pure JavaScript, no DOM. The parameters, subspaces, and probes come from
// docs/data/7-3/policy.json, written by tools/s73 (train.py, analyze.py, export.mjs).
//
// The network: 316 inputs -> 128 ReLU -> 128 ReLU -> 4 action scores; the agent takes the
// action with the highest score. All interventions act on the first hidden layer (h1),
// after its activation function, at every step:
//   remove goal:  h1 <- h1 - G(h1 - mean), where G projects onto the goal subspace;
//   swap goal:    h1 <- h1 + G(h1' - h1), where h1' is h1 for the same state with another goal;
//   set doors:    h1 <- h1 + D(h1'' - h1), where h1'' is h1 for the same state with other door states
//                 and D projects onto the door subspace.
import * as M from './s73-maze.js';

// Decodes a base64 string of little-endian float32 values.
function unpack(s) {
  const bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
const rows = (flat, d) => Array.from({ length: flat.length / d }, (_, i) => flat.subarray(i * d, (i + 1) * d));

// json: the contents of docs/data/7-3/policy.json.
export function load(json) {
  const d = json.sizes[1];
  return {
    sizes: json.sizes,
    W: json.W.map(unpack), b: json.b.map(unpack),
    mean: unpack(json.mean),
    goalBasis: rows(unpack(json.goal), d), // orthonormal directions, most variance first
    doorBasis: rows(unpack(json.door), d),
    kGoal: json.kGoal, kDoor: json.kDoor,
    goalProbe: { W: unpack(json.goalProbe.W), b: unpack(json.goalProbe.b), n: json.goalProbe.n },
    doorProbe: json.doorProbe.map((p) => ({ W: unpack(p.W), b: unpack(p.b), n: p.n })),
  };
}

function affine(W, b, x, nOut, relu) {
  const nIn = x.length, y = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) {
    let s = b[i];
    const row = i * nIn;
    for (let j = 0; j < nIn; j++) s += W[row + j] * x[j];
    y[i] = relu && s < 0 ? 0 : s;
  }
  return y;
}

export const hidden1 = (net, x) => affine(net.W[0], net.b[0], x, net.sizes[1], true);
export function scoresFrom1(net, h1) {
  const h2 = affine(net.W[1], net.b[1], h1, net.sizes[2], true);
  return affine(net.W[2], net.b[2], h2, net.sizes[3], false);
}
export function argmax(v) { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b; }

// Adds P v to h in place, where P projects onto the span of the first k rows of `basis`.
function addProjection(h, basis, k, v, sign = 1) {
  for (let r = 0; r < k; r++) {
    const u = basis[r];
    let c = 0;
    for (let j = 0; j < h.length; j++) c += u[j] * v[j];
    c *= sign;
    for (let j = 0; j < h.length; j++) h[j] += c * u[j];
  }
}
const diff = (a, b) => { const d = new Float32Array(a.length); for (let i = 0; i < a.length; i++) d[i] = a[i] - b[i]; return d; };

// One step's computation with interventions. cfg: { goalMode: 'as-is' | 'remove' | 'swap', goal2,
// doorBelief: null | [upper, lower] door states, kGoal, kDoor }.
export function compute(net, env, st, goal, cfg = {}) {
  const B = M.blocked(env);
  let h = hidden1(net, M.encode(env, st, goal, undefined, B));
  const kG = cfg.kGoal ?? net.kGoal, kD = cfg.kDoor ?? net.kDoor;
  if (cfg.goalMode === 'remove') addProjection(h, net.goalBasis, kG, diff(h, net.mean), -1);
  else if (cfg.goalMode === 'swap' && cfg.goal2 >= 0) {
    const h2 = hidden1(net, M.encode(env, st, cfg.goal2));
    addProjection(h, net.goalBasis, kG, diff(h2, h));
  }
  if (cfg.doorBelief) {
    const envB = { ...env, doors: cfg.doorBelief.slice() };
    const hb = hidden1(net, M.encode(envB, st, goal));
    addProjection(h, net.doorBasis, kD, diff(hb, h));
  }
  const scores = scoresFrom1(net, h);
  return { h, scores, action: argmax(scores) };
}

export function policy(net, env, goal, cfg = {}) {
  return (st) => compute(net, env, st, goal, cfg).action;
}

function softmaxProbe(p, h) {
  const z = affine(p.W, p.b, h, p.n, false);
  const m = Math.max(...z);
  let s = 0;
  for (let i = 0; i < z.length; i++) { z[i] = Math.exp(z[i] - m); s += z[i]; }
  for (let i = 0; i < z.length; i++) z[i] /= s;
  return z;
}
// Probability over the 63 cells of being the goal, read from h1 by the goal probe.
export const readGoal = (net, h) => softmaxProbe(net.goalProbe, h);
// For each door, probabilities of open, closed, locked, read from h1 by the door probes.
export const readDoors = (net, h) => net.doorProbe.map((p) => softmaxProbe(p, h));
