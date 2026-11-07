// Probe trainer for pages 6.2, 6.3 and 7.2: logistic regression, a two-layer probe,
// the difference of class means, and contrast-consistent search, trained on stored activations.
// Pure functions, no DOM, so the same code runs in the browser and in the Node tests.
// Every routine is deterministic: the same data and options give the same probe.

// ---- Data ----
// Decodes a base64 string of little-endian float16 values into a Float32Array.
export function decodeF16(b64) {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const n = bin.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

// Rows of a flat matrix with `d` columns, as an array of Float32Array views.
export function rows(flat, d) {
  const out = [];
  for (let i = 0; i < flat.length / d; i++) out.push(flat.subarray(i * d, (i + 1) * d));
  return out;
}

// Small seeded random number generator (mulberry32).
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }

export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const norm = (a) => Math.sqrt(dot(a, a));
const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

// Per-feature mean and standard deviation over the given rows.
export function moments(X) {
  const d = X[0].length, mu = new Float64Array(d), sd = new Float64Array(d);
  for (const x of X) for (let j = 0; j < d; j++) mu[j] += x[j] / X.length;
  for (const x of X) for (let j = 0; j < d; j++) sd[j] += (x[j] - mu[j]) ** 2 / X.length;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  return { mu, sd };
}
export function standardize(X, { mu, sd }) {
  return X.map((x) => { const z = new Float32Array(x.length); for (let j = 0; j < x.length; j++) z[j] = (x[j] - mu[j]) / sd[j]; return z; });
}

// ---- Adam on a flat parameter vector ----
function adam(params, lr) {
  const m = new Float64Array(params.length), v = new Float64Array(params.length);
  let t = 0;
  return (grad) => {
    t++;
    for (let i = 0; i < params.length; i++) {
      m[i] = 0.9 * m[i] + 0.1 * grad[i];
      v[i] = 0.999 * v[i] + 0.001 * grad[i] * grad[i];
      params[i] -= (lr * m[i] / (1 - 0.9 ** t)) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8);
    }
  };
}

// ---- Logistic regression: score = w . x + b, probability = sigmoid(score) ----
// l2 is the weight of the penalty l2 * |w|^2 / 2 added to the mean log loss.
export function logistic(X, y, { l2 = 0.01, steps = 300, lr = 0.05 } = {}) {
  const d = X[0].length, n = X.length;
  const p = new Float64Array(d + 1); // w then b
  const step = adam(p, lr);
  const g = new Float64Array(d + 1);
  for (let s = 0; s < steps; s++) {
    g.fill(0);
    for (let i = 0; i < n; i++) {
      const x = X[i];
      let z = p[d];
      for (let j = 0; j < d; j++) z += p[j] * x[j];
      const e = (sigmoid(z) - y[i]) / n;
      for (let j = 0; j < d; j++) g[j] += e * x[j];
      g[d] += e;
    }
    for (let j = 0; j < d; j++) g[j] += l2 * p[j];
    step(g);
  }
  const w = Float32Array.from(p.subarray(0, d)), b = p[d];
  return { kind: 'linear', w, b, score: (x) => dot(w, x) + b, prob: (x) => sigmoid(dot(w, x) + b) };
}

// ---- Two-layer probe: ReLU hidden layer of width h, then a sigmoid output ----
export function mlp(X, y, { hidden = 64, l2 = 0.01, steps = 300, lr = 0.01, seed = 1 } = {}) {
  const d = X[0].length, n = X.length, H = hidden;
  const r = rng(seed);
  const P = new Float64Array(H * d + H + H + 1); // W1 (H x d), b1, w2, b2
  for (let i = 0; i < H * d; i++) P[i] = gauss(r) * Math.sqrt(2 / d);
  for (let i = 0; i < H; i++) P[H * d + H + i] = gauss(r) * Math.sqrt(1 / H);
  const oB1 = H * d, oW2 = H * d + H, oB2 = H * d + 2 * H;
  const step = adam(P, lr);
  const g = new Float64Array(P.length), a = new Float64Array(H);
  const forward = (x) => {
    let z = P[oB2];
    for (let k = 0; k < H; k++) {
      let s = P[oB1 + k];
      const off = k * d;
      for (let j = 0; j < d; j++) s += P[off + j] * x[j];
      a[k] = s > 0 ? s : 0;
      z += P[oW2 + k] * a[k];
    }
    return z;
  };
  for (let s = 0; s < steps; s++) {
    g.fill(0);
    for (let i = 0; i < n; i++) {
      const x = X[i];
      const e = (sigmoid(forward(x)) - y[i]) / n;
      g[oB2] += e;
      for (let k = 0; k < H; k++) {
        g[oW2 + k] += e * a[k];
        if (a[k] > 0) {
          const ek = e * P[oW2 + k], off = k * d;
          g[oB1 + k] += ek;
          for (let j = 0; j < d; j++) g[off + j] += ek * x[j];
        }
      }
    }
    for (let i = 0; i < H * d; i++) g[i] += l2 * P[i];
    for (let k = 0; k < H; k++) g[oW2 + k] += l2 * P[oW2 + k];
    step(g);
  }
  return { kind: 'mlp', score: (x) => forward(x), prob: (x) => sigmoid(forward(x)) };
}

// ---- Difference of class means: direction mean(true) - mean(false), threshold halfway ----
export function massMean(X, y) {
  const d = X[0].length, m1 = new Float64Array(d), m0 = new Float64Array(d);
  let n1 = 0, n0 = 0;
  X.forEach((x, i) => { const m = y[i] ? m1 : m0; for (let j = 0; j < d; j++) m[j] += x[j]; y[i] ? n1++ : n0++; });
  const w = new Float32Array(d);
  for (let j = 0; j < d; j++) w[j] = m1[j] / n1 - m0[j] / n0;
  const mid = (dot(w, Float32Array.from(m1, (v) => v / n1)) + dot(w, Float32Array.from(m0, (v) => v / n0))) / 2;
  // Scale so that one unit of score is one standard deviation of the projections.
  const proj = X.map((x) => dot(w, x));
  const mu = proj.reduce((s, v) => s + v, 0) / proj.length;
  const sd = Math.sqrt(proj.reduce((s, v) => s + (v - mu) ** 2, 0) / proj.length) || 1;
  return { kind: 'linear', w, b: -mid, score: (x) => (dot(w, x) - mid) / sd, prob: (x) => sigmoid(4 * (dot(w, x) - mid) / sd) };
}

// ---- Contrast-consistent search (Burns et al. 2022) ----
// Xp[i] and Xn[i] are the activations for statement i followed by "true" and by "false".
// Each set is centered and scaled separately, as Burns et al. do, so that the probe cannot
// simply read which of the two words was appended. The loss for one pair is
// (p+ - (1 - p-))^2 + min(p+, p-)^2. The best of `restarts` random starts (lowest loss) is kept.
// CCS has no labels, so its sign is arbitrary: `prob` gives the credence that the statement is true
// under the sign it found, and callers that know the labels can flip it.
export function ccs(Xp, Xn, { l2 = 0, steps = 400, lr = 0.01, restarts = 10, seed = 1 } = {}) {
  const mp = moments(Xp), mn = moments(Xn);
  const P = standardize(Xp, mp), N = standardize(Xn, mn);
  const d = P[0].length, n = P.length;
  const r = rng(seed);
  let best = null;
  for (let t = 0; t < restarts; t++) {
    const p = new Float64Array(d + 1);
    for (let j = 0; j < d; j++) p[j] = gauss(r) / Math.sqrt(d);
    const step = adam(p, lr), g = new Float64Array(d + 1);
    let loss = 0;
    for (let s = 0; s < steps; s++) {
      g.fill(0); loss = 0;
      for (let i = 0; i < n; i++) {
        let zp = p[d], zn = p[d];
        for (let j = 0; j < d; j++) { zp += p[j] * P[i][j]; zn += p[j] * N[i][j]; }
        const a = sigmoid(zp), b = sigmoid(zn);
        const c = a + b - 1; // consistency residual: p+ - (1 - p-)
        const lo = Math.min(a, b);
        loss += (c * c + lo * lo) / n;
        // d/dz of each term
        let da = 2 * c, db = 2 * c;
        if (a <= b) da += 2 * lo; else db += 2 * lo;
        da *= a * (1 - a) / n; db *= b * (1 - b) / n;
        for (let j = 0; j < d; j++) g[j] += da * P[i][j] + db * N[i][j];
        g[d] += da + db;
      }
      for (let j = 0; j < d; j++) g[j] += l2 * p[j];
      step(g);
    }
    if (!best || loss < best.loss) best = { loss, p: Float64Array.from(p) };
  }
  const w = Float32Array.from(best.p.subarray(0, d)), b = best.p[d];
  const z = (x, m) => { let s = b; for (let j = 0; j < d; j++) s += w[j] * (x[j] - m.mu[j]) / m.sd[j]; return s; };
  // Credence that statement i is true: average of p(x+) and 1 - p(x-).
  const credence = (xp, xn) => (sigmoid(z(xp, mp)) + 1 - sigmoid(z(xn, mn))) / 2;
  // The direction in the original (unscaled) coordinates of the "true" set.
  const dir = Float32Array.from(w, (v, j) => v / mp.sd[j]);
  return { kind: 'ccs', w: dir, loss: best.loss, credence };
}

// ---- Evaluation ----
export function accuracy(probe, X, y) {
  let c = 0;
  X.forEach((x, i) => { if ((probe.score(x) > 0 ? 1 : 0) === y[i]) c++; });
  return X.length ? c / X.length : NaN;
}

// Principal directions of a set of rows by power iteration with deflation (k directions).
export function pca(X, k, { iters = 60, seed = 3 } = {}) {
  const { mu } = moments(X);
  const C = X.map((x) => Float64Array.from(x, (v, j) => v - mu[j]));
  const d = X[0].length, r = rng(seed), dirs = [];
  for (let c = 0; c < k; c++) {
    let v = Float64Array.from({ length: d }, () => gauss(r));
    for (let it = 0; it < iters; it++) {
      const nv = new Float64Array(d);
      for (const x of C) { const s = dot(x, v); for (let j = 0; j < d; j++) nv[j] += s * x[j]; }
      for (const u of dirs) { const s = dot(nv, u); for (let j = 0; j < d; j++) nv[j] -= s * u[j]; }
      const nn = norm(nv) || 1;
      v = nv.map((t) => t / nn);
    }
    dirs.push(v);
  }
  return { mu, dirs };
}
