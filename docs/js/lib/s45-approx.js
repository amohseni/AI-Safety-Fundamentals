// Page 5.1: sums of ReLU units on [0, 1], least-squares fits, and the sawtooth built by depth.
// Pure JavaScript, no DOM.
import { leastSquares } from './s45-linalg.js';

export const relu = (z) => (z > 0 ? z : 0);
export const GRID = Array.from({ length: 401 }, (_, i) => i / 400);

// Targets on [0, 1]. The bump is exactly three ReLU units: slopes 2.5, -5, 2.5 at 0.2, 0.4, 0.6.
export const TARGETS = {
  bump: (x) => 2.5 * relu(x - 0.2) - 5 * relu(x - 0.4) + 2.5 * relu(x - 0.6),
  smooth: (x) => 0.5 + 0.35 * Math.sin(2 * Math.PI * x),
  jagged: (x) => 0.5 + 0.22 * Math.sin(2 * Math.PI * 3 * x) + 0.12 * Math.sin(2 * Math.PI * 7 * x + 1) + 0.06 * Math.sin(2 * Math.PI * 13 * x + 2),
};

// A hand-built network: output bias b plus units { kink c, slope a }: f(x) = b + sum a ReLU(x - c).
export const handNet = (b, units) => (x) => units.reduce((s, u) => s + u.a * relu(x - u.c), b);

export const rms = (f, g, xs = GRID) => Math.sqrt(xs.reduce((s, x) => s + (f(x) - g(x)) ** 2, 0) / xs.length);

// Fit m units with kinks at 0, 1/m, ..., (m-1)/m and an output bias, by least squares on the grid.
export function fitEven(target, m, kinks = null) {
  const cs = kinks || Array.from({ length: m }, (_, k) => k / m);
  const Phi = GRID.map((x) => [1, ...cs.map((c) => relu(x - c))]);
  const coef = leastSquares(Phi, GRID.map(target), 1e-12);
  const b = coef[0], units = cs.map((c, k) => ({ c, a: coef[k + 1] }));
  return { b, units, f: handNet(b, units) };
}

// Telgarsky's tent map t(x) = 2 ReLU(x) - 4 ReLU(x - 1/2), and its k-fold composition: a network
// with k hidden layers of 2 ReLU units each (2k units in all).
export const tent = (x) => 2 * relu(x) - 4 * relu(x - 0.5);
export function deepSaw(k) {
  return (x) => { let z = x; for (let l = 0; l < k; l++) z = 2 * relu(z) - 4 * relu(z - 0.5); return z; };
}
// The same function written down directly: 2^(k-1) teeth on [0, 1].
export function sawtooth(k) {
  const n = 2 ** k;
  return (x) => { const u = (x * n) % 2; const v = u <= 1 ? u : 2 - u; return x >= 1 ? ((n % 2 === 0) ? 0 : 1) : v; };
}

// Number of linear pieces of f on [0, 1], counted from slope changes on a fine grid.
export function pieces(f, n = 1 << 14) {
  let count = 1, prev = null;
  for (let i = 0; i < n; i++) {
    const x0 = i / n, x1 = (i + 1) / n;
    const s = (f(x1) - f(x0)) * n;
    if (prev !== null && Math.abs(s - prev) > 1e-6 * Math.max(1, Math.abs(prev))) count++;
    prev = s;
  }
  return count;
}
