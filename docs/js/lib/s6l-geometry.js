// Planes through activation space for pages 6.2 and 7.2. No DOM, so the Node tests use it too.
import { pca, rng, dot, norm } from './s6l-probe.js';

export const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }
const unit = (v) => { const n = norm(v) || 1; return Float32Array.from(v, (t) => t / n); };
function orth(v, u) { const s = dot(v, u); return Float32Array.from(v, (t, j) => t - s * u[j]); }

// Two orthonormal directions spanning a plane.
//  'pca'    the two principal directions of the given rows;
//  'random' two random directions (seeded);
//  'mm'     the given direction (e.g. the difference of class means) and the principal direction
//           of the rows orthogonal to it.
export function planeBasis(X, idx, mode, { mm = null, seed = 1 } = {}) {
  const R = idx.map((i) => X[i]);
  const k = R[0].length;
  if (mode === 'random') {
    const r = rng(1000 + seed);
    const a = unit(Float32Array.from({ length: k }, () => gauss(r)));
    const b = unit(orth(Float32Array.from({ length: k }, () => gauss(r)), a));
    return { e1: a, e2: b };
  }
  if (mode === 'mm') {
    const e1 = unit(mm);
    const resid = R.map((x) => orth(x, e1));
    const { dirs } = pca(resid, 1);
    return { e1, e2: unit(orth(Float32Array.from(dirs[0]), e1)) };
  }
  const { dirs } = pca(R, 2);
  return { e1: Float32Array.from(dirs[0]), e2: Float32Array.from(dirs[1]) };
}

// The plane of the two principal directions of the seven per-day mean activations.
export function weekdayPlane(X, day) {
  const k = X[0].length;
  const means = DAYS.map((_, d) => {
    const m = new Float32Array(k); let n = 0;
    X.forEach((x, i) => { if (day[i] === d) { for (let j = 0; j < k; j++) m[j] += x[j]; n++; } });
    return m.map((v) => v / n);
  });
  const { mu, dirs } = pca(means, 2, { iters: 200 });
  // Center everything on the mean of the day means.
  const center = (x) => Float32Array.from(x, (v, j) => v - mu[j]);
  return { e1: Float32Array.from(dirs[0]), e2: Float32Array.from(dirs[1]), means: means.map(center), center };
}

// Best single threshold on a line: returns { acc, t, sign } where sign = 1 means values above t
// are predicted to be group A (true).
export function bestSplit(a, b) {
  const v = a.map((u) => [u, 1]).concat(b.map((u) => [u, 0])).sort((p, q) => p[0] - q[0]);
  const n = v.length, nA = a.length;
  let below1 = 0, best = { acc: 0, t: 0, sign: 1 };
  for (let i = 0; i <= n; i++) {
    // threshold between v[i-1] and v[i]
    const below0 = i - below1;
    const accUp = (below0 + (nA - below1)) / n; // A above
    const accDown = (below1 + (n - nA - below0)) / n; // A below
    const t = i === 0 ? v[0][0] - 1e-6 : i === n ? v[n - 1][0] + 1e-6 : (v[i - 1][0] + v[i][0]) / 2;
    if (accUp > best.acc) best = { acc: accUp, t, sign: 1 };
    if (accDown > best.acc) best = { acc: accDown, t, sign: -1 };
    if (i < n && v[i][1] === 1) below1++;
  }
  return best;
}
