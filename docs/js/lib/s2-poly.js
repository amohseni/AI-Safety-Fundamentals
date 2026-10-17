// Polynomial fits with Legendre features for page 2.8. No DOM; also used by tools/test-page-2-8.mjs.
// For degree d the features are P_0(x), ..., P_d(x) on [-1, 1]. The fit is the minimum-norm least-squares
// solution (the pseudo-inverse), which is the ordinary least-squares fit while there are no more features than
// points, and past that is the interpolating fit with the smallest sum of squared coefficients.
import { rng, gaussian } from './mlp.js';

export const truth = (x) => Math.sin(Math.PI * x) + 0.4 * x; // the smooth curve the points come from

export function legendre(x, d) {
  const P = [1, x];
  for (let k = 1; k < d; k++) P.push(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1));
  return P.slice(0, d + 1);
}

// One-sided Jacobi SVD of an m x n matrix with m >= n (rows of arrays). Returns {U (m x n), s (n), V (n x n)}.
function svdTall(A) {
  const m = A.length, n = A[0].length;
  const U = A.map((r) => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 80; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      let a = 0, b = 0, c = 0;
      for (let i = 0; i < m; i++) { a += U[i][p] * U[i][p]; b += U[i][q] * U[i][q]; c += U[i][p] * U[i][q]; }
      if (Math.abs(c) <= 1e-15 * Math.sqrt(a * b) || c === 0) continue;
      off = Math.max(off, Math.abs(c) / Math.sqrt(a * b));
      const z = (b - a) / (2 * c), t = Math.sign(z || 1) / (Math.abs(z) + Math.sqrt(1 + z * z));
      const cs = 1 / Math.sqrt(1 + t * t), sn = cs * t;
      for (let i = 0; i < m; i++) { const u = U[i][p], v = U[i][q]; U[i][p] = cs * u - sn * v; U[i][q] = sn * u + cs * v; }
      for (let i = 0; i < n; i++) { const u = V[i][p], v = V[i][q]; V[i][p] = cs * u - sn * v; V[i][q] = sn * u + cs * v; }
    }
    if (off < 1e-14) break;
  }
  const s = Array.from({ length: n }, (_, j) => Math.sqrt(U.reduce((acc, r) => acc + r[j] * r[j], 0)));
  for (let j = 0; j < n; j++) if (s[j] > 0) for (let i = 0; i < m; i++) U[i][j] /= s[j];
  return { U, s, V };
}

// Minimum-norm least-squares solution of A w = y.
export function pinvSolve(A, y) {
  const m = A.length, n = A[0].length;
  if (m >= n) {
    const { U, s, V } = svdTall(A), tol = Math.max(...s) * 1e-12 * Math.max(m, n);
    const c = s.map((sj, j) => (sj > tol ? U.reduce((acc, r, i) => acc + r[j] * y[i], 0) / sj : 0));
    return V.map((row) => row.reduce((acc, v, j) => acc + v * c[j], 0));
  }
  // Wide: A^T = U S V^T, so A = V S U^T and A^+ y = U S^+ V^T y.
  const At = Array.from({ length: n }, (_, j) => A.map((r) => r[j]));
  const { U, s, V } = svdTall(At), tol = Math.max(...s) * 1e-12 * Math.max(m, n);
  const c = s.map((sj, j) => (sj > tol ? V.reduce((acc, r, i) => acc + r[j] * y[i], 0) / sj : 0));
  return U.map((row) => row.reduce((acc, u, j) => acc + u * c[j], 0));
}

export function sample(n, noise, seed) {
  const r = rng(seed), X = [], Y = [];
  for (let i = 0; i < n; i++) { const x = 2 * r() - 1; X.push(x); Y.push(truth(x) + noise * gaussian(r)); }
  return { X, Y };
}

export function fit(data, d) {
  const A = data.X.map((x) => legendre(x, d));
  const w = pinvSolve(A, data.Y);
  return { w, f: (x) => legendre(x, d).reduce((acc, p, k) => acc + p * w[k], 0) };
}
export const mse = (f, data) => data.X.reduce((acc, x, i) => acc + (f(x) - data.Y[i]) ** 2, 0) / data.X.length;
