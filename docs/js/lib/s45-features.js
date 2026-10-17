// Page 5.3: regression on p random ReLU features, fitted by minimum-norm least squares or ridge.
// Pure JavaScript, no DOM.
import { rng, gaussian } from './mlp.js';
import { minNormSolve } from './s45-linalg.js';

export const N_TRAIN = 40;
export const P_MAX = 1000;
export const truth = (x) => Math.sin(Math.PI * x);

// Each input x in [-1, 1] is first written as a point on an arc of the unit circle,
// e(x) = (cos 0.75 pi x, sin 0.75 pi x). A feature is a ReLU unit with random parameters applied to
// that point: phi_j(x) = ReLU(w_j . e(x) + b_j) / sqrt(p). With x alone, one-dimensional ReLU
// features are piecewise linear with kinks between the data points, and many of them are redundant.
export const enc = (x) => [Math.cos(0.75 * Math.PI * x), Math.sin(0.75 * Math.PI * x)];

// Training inputs: one per cell of an even grid, jittered (fixed for the page). Labels: truth + noise.
export function makeData(noise, seed = 1) {
  const r = rng(seed);
  const X = Array.from({ length: N_TRAIN }, (_, i) => -1 + (2 * (i + 0.5)) / N_TRAIN + (r() - 0.5) * (1.2 / N_TRAIN));
  const z = X.map(() => gaussian(r));
  return { X, Y: X.map((x, i) => truth(x) + noise * z[i]) };
}

// P_MAX random features. Each is kept only if it is active on at least 2 and at most 39 training
// inputs, so that every feature has its kink inside the data.
export function makeFeatures(X, seed) {
  const r = rng(seed), W = [], B = [];
  const E = X.map(enc);
  while (W.length < P_MAX) {
    const w = [gaussian(r), gaussian(r)], b = gaussian(r);
    const active = E.filter((e) => w[0] * e[0] + w[1] * e[1] + b > 0).length;
    if (active >= 2 && active <= X.length - 1) { W.push(w); B.push(b); }
  }
  return { W, B };
}

export function featuresAt(F, x, p) {
  const e = enc(x), o = new Float64Array(p), s = 1 / Math.sqrt(p);
  for (let j = 0; j < p; j++) { const z = F.W[j][0] * e[0] + F.W[j][1] * e[1] + F.B[j]; o[j] = z > 0 ? z * s : 0; }
  return o;
}

export const TEST_X = Array.from({ length: 200 }, (_, i) => -1 + (2 * (i + 0.5)) / 200);

// Fit with the first p features; ridge = 0 gives the minimum-norm least-squares solution.
export function fit(data, F, p, ridge = 0) {
  const Phi = data.X.map((x) => featuresAt(F, x, p));
  const a = minNormSolve(Phi, data.Y, ridge);
  const f = (x) => { const ph = featuresAt(F, x, p); let s = 0; for (let j = 0; j < p; j++) s += ph[j] * a[j]; return s; };
  const train = data.X.reduce((s, x, i) => s + (f(x) - data.Y[i]) ** 2, 0) / data.X.length;
  // Test error: mean squared distance from the true curve at 200 evenly spaced new inputs.
  const test = TEST_X.reduce((s, x) => s + (f(x) - truth(x)) ** 2, 0) / TEST_X.length;
  return { f, train, test, norm: Math.sqrt(a.reduce((s, v) => s + v * v, 0)) };
}

// The values of p shown on the error chart: dense near the threshold, log spaced elsewhere.
export const P_GRID = [...new Set([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 50, 53, 56, 60, 65, 70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500, 650, 800, 1000])];
