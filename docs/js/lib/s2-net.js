// Small multilayer network with a separate activation function (or none) for each hidden layer.
// Used by page 2.3. Pure JavaScript, no DOM, so Node tests can import it.
// Output: one unit with a sigmoid; loss: mean binary cross-entropy; optimizer: Adam, full batch.
import { rng, gaussian } from './mlp.js';

// erf by Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7).
function erf(x) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
const SQ2 = Math.SQRT2, ISQ2PI = 1 / Math.sqrt(2 * Math.PI);
// f(z) and f'(z), both from the pre-activation z.
export const ACTS = {
  relu: { name: 'ReLU', f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
  gelu: { name: 'GELU', f: (z) => 0.5 * z * (1 + erf(z / SQ2)), d: (z) => 0.5 * (1 + erf(z / SQ2)) + z * ISQ2PI * Math.exp(-0.5 * z * z) },
  tanh: { name: 'tanh', f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2 },
  sigmoid: { name: 'sigmoid', f: (z) => 1 / (1 + Math.exp(-z)), d: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); } },
  none: { name: 'none', f: (z) => z, d: () => 1 },
};

export class Net {
  // sizes: e.g. [2, 8, 8, 8, 1]; acts: one name per hidden layer
  constructor(sizes, acts, { seed = 1, lr = 0.01 } = {}) {
    this.sizes = sizes; this.acts = acts.slice(); this.lr = lr;
    const r = rng(seed);
    this.W = []; this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1], sc = Math.sqrt(2 / nIn);
      this.W.push(Array.from({ length: nOut }, () => Array.from({ length: nIn }, () => gaussian(r) * sc)));
      this.b.push(Array.from({ length: nOut }, () => gaussian(r) * 0.1));
    }
    this.t = 0;
    this.m = this._zeros(); this.v = this._zeros();
  }
  _zeros() { return { W: this.W.map((M) => M.map((row) => row.map(() => 0))), b: this.b.map((v) => v.map(() => 0)) }; }
  get L() { return this.W.length; }

  // Returns pre-activations z[l] and activations a[l] (a[0] = input). The last z is the output logit.
  forward(x) {
    const z = [], a = [x];
    for (let l = 0; l < this.L; l++) {
      const W = this.W[l], b = this.b[l], prev = a[l];
      const zl = new Array(W.length);
      for (let i = 0; i < W.length; i++) { let s = b[i]; const row = W[i]; for (let j = 0; j < row.length; j++) s += row[j] * prev[j]; zl[i] = s; }
      z.push(zl);
      if (l < this.L - 1) { const f = ACTS[this.acts[l]].f; a.push(zl.map(f)); }
    }
    return { z, a };
  }
  logit(x) { const { z } = this.forward(x); return z[z.length - 1][0]; }
  predict(x) { return 1 / (1 + Math.exp(-this.logit(x))); }

  // On/off pattern of the ReLU units in layers whose activation is ReLU, as a string.
  pattern(x) {
    const { z } = this.forward(x);
    let s = '';
    for (let l = 0; l < this.L - 1; l++) if (this.acts[l] === 'relu') for (const v of z[l]) s += v > 0 ? '1' : '0';
    return s;
  }

  accuracy(X, Y) { let c = 0; for (let n = 0; n < X.length; n++) c += (this.logit(X[n]) > 0) === (Y[n] === 1) ? 1 : 0; return c / X.length; }
  loss(X, Y) {
    let s = 0;
    for (let n = 0; n < X.length; n++) { const q = this.logit(X[n]); s += Math.max(q, 0) - q * Y[n] + Math.log1p(Math.exp(-Math.abs(q))); }
    return s / X.length;
  }

  // Gradient of the mean loss (backpropagation), as {W, b} arrays shaped like the parameters.
  grad(X, Y) {
    const g = this._zeros(), N = X.length;
    for (let n = 0; n < N; n++) {
      const { z, a } = this.forward(X[n]);
      const p = 1 / (1 + Math.exp(-z[this.L - 1][0]));
      let delta = [p - Y[n]];
      for (let l = this.L - 1; l >= 0; l--) {
        const prev = a[l];
        for (let i = 0; i < delta.length; i++) { g.b[l][i] += delta[i]; for (let j = 0; j < prev.length; j++) g.W[l][i][j] += delta[i] * prev[j]; }
        if (l > 0) {
          const d = ACTS[this.acts[l - 1]].d, zp = z[l - 1];
          const nd = new Array(prev.length).fill(0);
          for (let j = 0; j < prev.length; j++) { let s = 0; for (let i = 0; i < delta.length; i++) s += this.W[l][i][j] * delta[i]; nd[j] = s * d(zp[j]); }
          delta = nd;
        }
      }
    }
    for (let l = 0; l < this.L; l++) for (let i = 0; i < g.W[l].length; i++) { g.b[l][i] /= N; for (let j = 0; j < g.W[l][i].length; j++) g.W[l][i][j] /= N; }
    return g;
  }

  // One full-batch Adam step.
  step(X, Y) {
    const g = this.grad(X, Y), N = 1;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.t++;
    const c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    const upd = (P, G, M, V, i, j) => {
      const gr = (j === undefined ? G[i] : G[i][j]) / N;
      const mo = b1 * (j === undefined ? M[i] : M[i][j]) + (1 - b1) * gr;
      const ve = b2 * (j === undefined ? V[i] : V[i][j]) + (1 - b2) * gr * gr;
      const step = (this.lr * (mo / c1)) / (Math.sqrt(ve / c2) + eps);
      if (j === undefined) { P[i] -= step; M[i] = mo; V[i] = ve; } else { P[i][j] -= step; M[i][j] = mo; V[i][j] = ve; }
    };
    for (let l = 0; l < this.L; l++) for (let i = 0; i < this.W[l].length; i++) {
      for (let j = 0; j < this.W[l][i].length; j++) upd(this.W[l], g.W[l], this.m.W[l], this.v.W[l], i, j);
      upd(this.b[l], g.b[l], this.m.b[l], this.v.b[l], i);
    }
  }

  // When every hidden layer has no activation function, the network is x -> v.x + c followed by the sigmoid.
  collapse() {
    let M = [[1, 0], [0, 1]], c = [0, 0];
    for (let l = 0; l < this.L; l++) {
      const W = this.W[l];
      M = W.map((row) => [0, 1].map((k) => row.reduce((s, w, j) => s + w * M[j][k], 0)));
      c = W.map((row, i) => row.reduce((s, w, j) => s + w * c[j], this.b[l][i]));
    }
    return { v: M[0], c: c[0] };
  }
}

// Number of distinct ReLU on/off patterns met on an n by n grid over [lo, hi]^2.
// Each pattern's region is convex, so this counts the linear pieces that contain at least one grid point.
export function countRegions(net, n = 300, lo = -1.2, hi = 1.2) {
  const seen = new Set();
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) seen.add(net.pattern([lo + ((hi - lo) * (i + 0.5)) / n, lo + ((hi - lo) * (j + 0.5)) / n]));
  return seen.size;
}
