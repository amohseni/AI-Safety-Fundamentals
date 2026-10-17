// Small dense linear algebra for live fits (pages 5.1 and 5.3).
// Pure JavaScript, no DOM, so it runs in the browser and in Node tests.

// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
// A: array of rows (n by n). Returns { values, vectors } with vectors[k] the k-th eigenvector.
export function symEig(A, { sweeps = 60, tol = 1e-15 } = {}) {
  const n = A.length;
  const a = A.map((r) => Float64Array.from(r));
  const v = Array.from({ length: n }, (_, i) => { const r = new Float64Array(n); r[i] = 1; return r; });
  for (let s = 0; s < sweeps; s++) {
    let off = 0, diag = 0;
    for (let i = 0; i < n; i++) { diag += a[i][i] * a[i][i]; for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j]; }
    if (off <= tol * tol * diag) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - sn * akq; a[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - sn * aqk; a[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[p][k], vkq = v[q][k];
          v[p][k] = c * vkp - sn * vkq; v[q][k] = sn * vkp + c * vkq;
        }
      }
    }
  }
  return { values: Array.from({ length: n }, (_, i) => a[i][i]), vectors: v };
}

// Minimum-norm (ridge = 0) or ridge least squares for Phi a = y, where Phi is n rows by p columns.
// Uses a = Phi^T (Phi Phi^T + ridge I)^+ y, which is the minimum-norm solution when ridge = 0
// and the ridge solution otherwise, for any p. Returns the coefficient vector a (length p).
export function minNormSolve(Phi, y, ridge = 0, relTol = 1e-11) {
  const n = Phi.length, p = Phi[0].length;
  const G = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    let s = 0; const ri = Phi[i], rj = Phi[j];
    for (let k = 0; k < p; k++) s += ri[k] * rj[k];
    G[i][j] = s; G[j][i] = s;
  }
  const { values, vectors } = symEig(G);
  const vmax = Math.max(...values.map(Math.abs), 1e-300);
  const c = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const lam = values[k] + ridge;
    if (ridge === 0 && values[k] <= relTol * vmax) continue;
    let proj = 0; const u = vectors[k];
    for (let i = 0; i < n; i++) proj += u[i] * y[i];
    for (let i = 0; i < n; i++) c[i] += (u[i] * proj) / lam;
  }
  const a = new Float64Array(p);
  for (let i = 0; i < n; i++) { const ri = Phi[i], ci = c[i]; for (let k = 0; k < p; k++) a[k] += ri[k] * ci; }
  return a;
}

// Ordinary least squares with a small ridge, through the normal equations (p small).
export function leastSquares(Phi, y, ridge = 1e-10) {
  const n = Phi.length, p = Phi[0].length;
  const A = Array.from({ length: p }, () => new Float64Array(p)), b = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const r = Phi[i];
    for (let j = 0; j < p; j++) { b[j] += r[j] * y[i]; for (let k = j; k < p; k++) A[j][k] += r[j] * r[k]; }
  }
  let tr = 0;
  for (let j = 0; j < p; j++) tr += A[j][j];
  for (let j = 0; j < p; j++) { A[j][j] += ridge * (tr / p || 1); for (let k = 0; k < j; k++) A[j][k] = A[k][j]; }
  const { values, vectors } = symEig(A);
  const vmax = Math.max(...values.map(Math.abs), 1e-300);
  const a = new Float64Array(p);
  for (let k = 0; k < p; k++) {
    if (values[k] <= 1e-13 * vmax) continue;
    let proj = 0; const u = vectors[k];
    for (let j = 0; j < p; j++) proj += u[j] * b[j];
    for (let j = 0; j < p; j++) a[j] += (u[j] * proj) / values[k];
  }
  return a;
}
