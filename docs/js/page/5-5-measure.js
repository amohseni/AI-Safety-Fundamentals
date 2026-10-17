// Measurements for page 5.5, panel 1 (no DOM, so tools/test-page-5-5.mjs runs the same code).
import { MLP, datasets } from '../lib/mlp.js';

// Five starting points for width 3 that reach 100% on the training points within 4,000 steps,
// also when the initial parameters are perturbed by 1e-9 or 1e-7 (out of 40 tried; see
// tools/test-page-5-5.mjs). The perturbation check matters because browsers can differ in the
// last bits of Math.tanh and Math.exp, and some runs are sensitive to that. Seed 9 passes the
// perturbation check in Node but ended at 97% in the headless Chromium used for testing, so it is left out.
export const SEEDS = [1, 3, 12, 22, 31];
export const STEPS = 4000;
export const DATA = datasets.spirals();
// New points from the same spirals, used to compare runs where the task is defined.
export const FRESH = datasets.spirals(2000, 99);

export const makeNet = (seed) => new MLP([2, 3, 3, 1], { seed, lr: 0.03 });

export function corr(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}

// Fraction of the fresh points that two networks put in the same class.
export function agreement(n1, n2, X = FRESH.X) {
  let same = 0;
  for (const x of X) same += (n1.predict(x) > 0.5) === (n2.predict(x) > 0.5) ? 1 : 0;
  return same / X.length;
}

// Responses of the units of hidden layer `layer` (1 or 2) on the fresh points: units x points.
export function responses(net, layer, X = FRESH.X) {
  const S = X.map((x) => net.stages(x)[2 * layer]);
  return S[0].map((_, u) => S.map((s) => s[u]));
}

// For each unit of run 1, the largest |correlation| with any unit of run 2
// (so a unit that reappears with its sign flipped or under another number still counts as a match).
export function bestMatches(R1, R2) {
  return R1.map((u) => Math.max(...R2.map((v) => Math.abs(corr(u, v)))));
}

export const median = (xs) => { const s = xs.slice().sort((a, b) => a - b), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
