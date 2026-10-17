// Toy model of superposition (Elhage et al. 2022) and a sparse autoencoder trained on its hidden activations.
// Pure JavaScript, no DOM, so it runs in the browser and in Node tests.
//
// Toy model: x in R^n, feature i is active with probability 1 - S and then uniform on [0, 1], else 0.
//   h = W x  (W is m by n; column i is feature i's direction),  x' = ReLU(W^T h + b).
//   Loss: mean over the batch of sum_i I_i (x_i - x'_i)^2, with importance I_i = r^i (i = 0, ..., n - 1).
// Sparse autoencoder (as in Bricken et al. 2023): f = ReLU(E (h - c) + e), h' = D f + c,
//   loss: mean of ||h - h'||^2 + lambda * sum_k f_k, with each column of D kept at length 1.
// Both are trained with Adam on fresh random batches.
import { rng, gaussian } from './mlp.js';

class Adam {
  constructor(size, lr) { this.m = new Float64Array(size); this.v = new Float64Array(size); this.t = 0; this.lr = lr; }
  // Updates the parameter arrays in place; grads is a list of arrays matching params.
  step(params, grads) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    let o = 0;
    for (let p = 0; p < params.length; p++) {
      const P = params[p], G = grads[p];
      for (let i = 0; i < P.length; i++, o++) {
        const g = G[i];
        this.m[o] = b1 * this.m[o] + (1 - b1) * g;
        this.v[o] = b2 * this.v[o] + (1 - b2) * g * g;
        P[i] -= (this.lr * (this.m[o] / c1)) / (Math.sqrt(this.v[o] / c2) + eps);
      }
    }
  }
}

// Draws a batch of sparse feature vectors (rows of length n, flattened).
export function sampleFeatures(rand, n, S, batch) {
  const X = new Float64Array(batch * n);
  for (let i = 0; i < X.length; i++) if (rand() >= S) X[i] = rand();
  return X;
}

export class ToyModel {
  // The learning rate falls geometrically from lr to lrEnd over the first `total` steps, then stays at lrEnd.
  constructor({ n = 5, m = 2, S = 0.9, r = 1, seed = 1, lr = 0.01, lrEnd = 0.01, total = 4000, batch = 256 } = {}) {
    Object.assign(this, { n, m, S, r, lr, lrEnd, total, batch });
    this.rand = rng(seed);
    const scale = Math.sqrt(2 / (n + m));
    this.W = Float64Array.from({ length: m * n }, () => gaussian(this.rand) * scale); // W[k * n + i]
    this.b = new Float64Array(n);
    this.adam = new Adam(m * n + n, lr);
    this.steps = 0;
    this.lastLoss = NaN;
  }

  importance(i) { return this.r ** i; }

  // Hidden activation h = W x for one feature vector x (array of length n).
  hidden(x, off = 0) {
    const { n, m, W } = this;
    const h = new Float64Array(m);
    for (let k = 0; k < m; k++) { let s = 0; for (let i = 0; i < n; i++) s += W[k * n + i] * x[off + i]; h[k] = s; }
    return h;
  }

  // Loss and (optionally) gradients on a batch X.
  _pass(X, B, wantGrad) {
    const { n, m, W, b } = this;
    const gW = wantGrad ? new Float64Array(m * n) : null, gb = wantGrad ? new Float64Array(n) : null;
    const I = Array.from({ length: n }, (_, i) => this.importance(i));
    const h = new Float64Array(m), dz = new Float64Array(n), dh = new Float64Array(m);
    let loss = 0;
    for (let s = 0; s < B; s++) {
      const o = s * n;
      for (let k = 0; k < m; k++) { let a = 0; for (let i = 0; i < n; i++) a += W[k * n + i] * X[o + i]; h[k] = a; }
      for (let i = 0; i < n; i++) {
        let z = b[i];
        for (let k = 0; k < m; k++) z += W[k * n + i] * h[k];
        const out = z > 0 ? z : 0, err = out - X[o + i];
        loss += I[i] * err * err;
        dz[i] = z > 0 ? (2 * I[i] * err) / B : 0;
      }
      if (!wantGrad) continue;
      for (let k = 0; k < m; k++) {
        let a = 0;
        for (let i = 0; i < n; i++) { a += W[k * n + i] * dz[i]; gW[k * n + i] += dz[i] * h[k]; }
        dh[k] = a;
      }
      for (let i = 0; i < n; i++) {
        gb[i] += dz[i];
        if (X[o + i] !== 0) for (let k = 0; k < m; k++) gW[k * n + i] += dh[k] * X[o + i];
      }
    }
    return { loss: loss / B, gW, gb };
  }

  step() {
    const X = sampleFeatures(this.rand, this.n, this.S, this.batch);
    const { loss, gW, gb } = this._pass(X, this.batch, true);
    this.adam.lr = this.lr * (this.lrEnd / this.lr) ** Math.min(1, this.steps / this.total);
    this.adam.step([this.W, this.b], [gW, gb]);
    this.steps++;
    this.lastLoss = loss;
    return loss;
  }

  // Loss on a fixed evaluation sample (same sample for every call with the same seed).
  evalLoss(N = 4000, seed = 999) {
    return this._pass(sampleFeatures(rng(seed), this.n, this.S, N), N, false).loss;
  }

  // Column i of W as an array of length m.
  dir(i) { return Array.from({ length: this.m }, (_, k) => this.W[k * this.n + i]); }
  norms() { return Array.from({ length: this.n }, (_, i) => Math.hypot(...this.dir(i))); }
  // W^T W: entry (i, j) is the dot product of directions i and j.
  gram() {
    const { n, m, W } = this;
    return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
      let s = 0; for (let k = 0; k < m; k++) s += W[k * n + i] * W[k * n + j]; return s;
    }));
  }
  // Interference on feature i: sum over j != i of (unit direction i . direction j)^2.
  interference() {
    const G = this.gram(), N = this.norms();
    return G.map((row, i) => (N[i] < 1e-9 ? 0 : row.reduce((s, g, j) => (j === i ? s : s + (g / N[i]) ** 2), 0)));
  }
  // Number of features whose direction has length above a threshold.
  stored(th = 0.5) { return this.norms().filter((v) => v > th).length; }
}

// Angles (in degrees, 0 to 360) of the stored directions of a 2-dimensional model, sorted.
export function angles2d(model, th = 0.5) {
  const out = [];
  for (let i = 0; i < model.n; i++) {
    const [a, c] = model.dir(i);
    if (Math.hypot(a, c) > th) out.push(((Math.atan2(c, a) * 180) / Math.PI + 360) % 360);
  }
  return out.sort((a, b) => a - b);
}

// Gaps between consecutive stored directions in the plane (degrees); a regular pentagon has five gaps of 72.
export function gaps2d(model, th = 0.5) {
  const a = angles2d(model, th);
  return a.map((v, i) => (i + 1 < a.length ? a[i + 1] - v : 360 - v + a[0]));
}

export class SAE {
  // model: a trained ToyModel, whose hidden activations h = W x are the data.
  constructor(model, { size = 5, lambda = 0.1, seed = 1, lr = 0.005, batch = 256 } = {}) {
    Object.assign(this, { model, size, lambda, lr, batch });
    this.d = model.m;
    this.rand = rng(seed);
    this.dataRand = rng(seed + 7919);
    const d = this.d;
    // Decoder columns start as random unit vectors; the encoder starts as the decoder's transpose.
    this.D = new Float64Array(d * size);
    for (let k = 0; k < size; k++) {
      const v = Array.from({ length: d }, () => gaussian(this.rand));
      const len = Math.hypot(...v);
      v.forEach((x, j) => { this.D[j * size + k] = x / len; });
    }
    this.E = new Float64Array(size * d);
    for (let k = 0; k < size; k++) for (let j = 0; j < d; j++) this.E[k * d + j] = this.D[j * size + k];
    this.e = new Float64Array(size);
    this.c = new Float64Array(d);
    this.adam = new Adam(d * size + size * d + size + d, lr);
    this.steps = 0;
  }

  _data(rand, B) {
    const { model } = this;
    const X = sampleFeatures(rand, model.n, model.S, B);
    const H = new Float64Array(B * this.d);
    for (let s = 0; s < B; s++) { const h = model.hidden(X, s * model.n); for (let j = 0; j < this.d; j++) H[s * this.d + j] = h[j]; }
    return H;
  }

  encode(h) {
    const { size, d, E, e, c } = this;
    const f = new Float64Array(size);
    for (let k = 0; k < size; k++) { let z = e[k]; for (let j = 0; j < d; j++) z += E[k * d + j] * (h[j] - c[j]); f[k] = z > 0 ? z : 0; }
    return f;
  }

  _pass(H, B, wantGrad) {
    const { size, d, D, E, e, c, lambda } = this;
    const gD = wantGrad ? new Float64Array(d * size) : null, gE = wantGrad ? new Float64Array(size * d) : null;
    const ge = wantGrad ? new Float64Array(size) : null, gc = wantGrad ? new Float64Array(d) : null;
    const f = new Float64Array(size), hc = new Float64Array(d), r = new Float64Array(d), df = new Float64Array(size);
    let rec = 0, pen = 0, l0 = 0, varSum = 0;
    const mean = new Float64Array(d);
    for (let s = 0; s < B; s++) for (let j = 0; j < d; j++) mean[j] += H[s * d + j] / B;
    for (let s = 0; s < B; s++) {
      const o = s * d;
      for (let j = 0; j < d; j++) hc[j] = H[o + j] - c[j];
      for (let k = 0; k < size; k++) {
        let z = e[k]; for (let j = 0; j < d; j++) z += E[k * d + j] * hc[j];
        f[k] = z > 0 ? z : 0;
        if (f[k] > 0) { l0++; pen += f[k]; }
      }
      for (let j = 0; j < d; j++) {
        let y = c[j]; for (let k = 0; k < size; k++) y += D[j * size + k] * f[k];
        r[j] = y - H[o + j];
        rec += r[j] * r[j];
        varSum += (H[o + j] - mean[j]) ** 2;
      }
      if (!wantGrad) continue;
      for (let j = 0; j < d; j++) gc[j] += (2 * r[j]) / B;
      for (let k = 0; k < size; k++) {
        let a = 0;
        for (let j = 0; j < d; j++) { gD[j * size + k] += (2 * r[j] * f[k]) / B; a += D[j * size + k] * 2 * r[j]; }
        df[k] = f[k] > 0 ? (a + lambda) / B : 0;
      }
      for (let k = 0; k < size; k++) {
        if (df[k] === 0) continue;
        ge[k] += df[k];
        for (let j = 0; j < d; j++) { gE[k * d + j] += df[k] * hc[j]; gc[j] -= df[k] * E[k * d + j]; }
      }
    }
    return { rec: rec / B, pen: pen / B, l0: l0 / B, fvu: varSum > 0 ? rec / varSum : 0, gD, gE, ge, gc };
  }

  _normalize() {
    const { size, d, D } = this;
    for (let k = 0; k < size; k++) {
      let len = 0; for (let j = 0; j < d; j++) len += D[j * size + k] ** 2;
      len = Math.sqrt(len) || 1;
      for (let j = 0; j < d; j++) D[j * size + k] /= len;
    }
  }

  step() {
    const H = this._data(this.dataRand, this.batch);
    const g = this._pass(H, this.batch, true);
    this.adam.step([this.D, this.E, this.e, this.c], [g.gD, g.gE, g.ge, g.gc]);
    this._normalize();
    this.steps++;
    return g;
  }

  // Reconstruction error, penalty, average number of active units, and the fraction of variance left unexplained.
  evaluate(N = 4000, seed = 4242) {
    const H = this._data(rng(seed), N);
    const { rec, pen, l0, fvu } = this._pass(H, N, false);
    // A dictionary unit is dead if it is never active on the evaluation sample.
    const alive = new Array(this.size).fill(false);
    for (let s = 0; s < N; s++) { const f = this.encode(H.subarray(s * this.d, (s + 1) * this.d)); f.forEach((v, k) => { if (v > 0) alive[k] = true; }); }
    return { rec, pen, l0, fvu, dead: alive.filter((a) => !a).length, alive };
  }

  dir(k) { return Array.from({ length: this.d }, (_, j) => this.D[j * this.size + k]); }

  // For each stored true feature, the largest cosine similarity with any live dictionary direction.
  recovery(th = 0.5, alive = null) {
    const { model } = this;
    const out = [];
    for (let i = 0; i < model.n; i++) {
      const w = model.dir(i), len = Math.hypot(...w);
      if (len <= th) continue;
      let best = -1, arg = -1;
      for (let k = 0; k < this.size; k++) {
        if (alive && !alive[k]) continue;
        const dk = this.dir(k);
        const cos = w.reduce((s, v, j) => s + v * dk[j], 0) / len;
        if (cos > best) { best = cos; arg = k; }
      }
      out.push({ feature: i, cos: best, unit: arg });
    }
    return out;
  }
}

// Training settings used by pages 6.4 and 6.5 and by their tests.
// Each run trains several starting points and keeps the one with the lowest loss, as Elhage et al. (2022) did,
// because small toy models often settle in poor arrangements from a single start.
export const TRAIN = { steps: 4000, lr: 0.01, lrEnd: 0.0005, restarts: 4 };
export function seedsFor(group, K = TRAIN.restarts) { return Array.from({ length: K }, (_, k) => 1 + group * K + k); }
export function makeRuns({ n, m, S, r, group = 0 }) {
  return seedsFor(group).map((seed) => new ToyModel({ n, m, S, r, seed, lr: TRAIN.lr, lrEnd: TRAIN.lrEnd, total: TRAIN.steps }));
}
// Trains every run to the full number of steps and returns the one with the lowest evaluation loss.
export function trainBest(opts) {
  const runs = makeRuns(opts);
  for (const t of runs) for (let i = 0; i < TRAIN.steps; i++) t.step();
  return pickBest(runs);
}
export function pickBest(runs) {
  let best = null, bl = Infinity;
  for (const t of runs) { const l = t.evalLoss(); if (l < bl) { bl = l; best = t; } }
  return best;
}
