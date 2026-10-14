// Small multilayer perceptron for the plane classifier.
// Pure JavaScript, no DOM, so it runs in the browser and in Node tests.
// Architecture: input -> [Linear -> activation] x L -> Linear -> sigmoid.
// Loss: binary cross-entropy. Optimizer: Adam, full batch.

export function rng(seed) {
  // Mulberry32: small, fast, deterministic.
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand) {
  const u = Math.max(rand(), 1e-12), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const activations = {
  tanh: { f: Math.tanh, df: (y) => 1 - y * y }, // df takes the activation's output
  relu: { f: (z) => (z > 0 ? z : 0), df: (y) => (y > 0 ? 1 : 0) },
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), df: (y) => y * (1 - y) },
  none: { f: (z) => z, df: () => 1 },
};

export class MLP {
  // sizes: e.g. [2, 3, 3, 1]; act: name of hidden activation
  constructor(sizes, { act = 'tanh', seed = 1, lr = 0.03 } = {}) {
    this.sizes = sizes;
    this.act = act;
    this.lr = lr;
    const rand = rng(seed);
    this.W = []; this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const scale = act === 'relu' ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
      this.W.push(Array.from({ length: nOut }, () => Array.from({ length: nIn }, () => gaussian(rand) * scale)));
      this.b.push(new Array(nOut).fill(0));
    }
    this.t = 0;
    this.m = this._zeros(); this.v = this._zeros();
  }

  _zeros() {
    return {
      W: this.W.map((M) => M.map((r) => r.map(() => 0))),
      b: this.b.map((v) => v.map(() => 0)),
    };
  }

  get nLayers() { return this.W.length; }

  // Returns the stages a point passes through:
  // [input, z1, a1, z2, a2, ..., zOut, out]. z = after the linear map, a = after the activation.
  stages(x) {
    const out = [x.slice()];
    let a = x;
    for (let l = 0; l < this.nLayers; l++) {
      const W = this.W[l], b = this.b[l];
      const z = W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], b[i]));
      const last = l === this.nLayers - 1;
      const f = activations[last ? 'sigmoid' : this.act].f;
      const y = z.map(f);
      out.push(z, y);
      a = y;
    }
    return out;
  }

  predict(x) { const s = this.stages(x); return s[s.length - 1][0]; }

  loss(X, Y) {
    let L = 0;
    for (let n = 0; n < X.length; n++) {
      const p = Math.min(Math.max(this.predict(X[n]), 1e-7), 1 - 1e-7);
      L -= Y[n] * Math.log(p) + (1 - Y[n]) * Math.log(1 - p);
    }
    return L / X.length;
  }

  accuracy(X, Y) {
    let c = 0;
    for (let n = 0; n < X.length; n++) c += (this.predict(X[n]) > 0.5) === (Y[n] === 1) ? 1 : 0;
    return c / X.length;
  }

  // Full-batch gradient of the mean binary cross-entropy (backpropagation).
  grad(X, Y) {
    const g = this._zeros();
    let L = 0;
    for (let n = 0; n < X.length; n++) {
      const s = this.stages(X[n]);
      const p = s[s.length - 1][0];
      const pc = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
      L -= Y[n] * Math.log(pc) + (1 - Y[n]) * Math.log(1 - pc);
      // Sigmoid + cross-entropy: dL/dz_out = p - y.
      let delta = [p - Y[n]];
      for (let l = this.nLayers - 1; l >= 0; l--) {
        const aPrev = s[2 * l]; // activation feeding layer l
        for (let i = 0; i < delta.length; i++) {
          g.b[l][i] += delta[i];
          for (let j = 0; j < aPrev.length; j++) g.W[l][i][j] += delta[i] * aPrev[j];
        }
        if (l > 0) {
          const df = activations[this.act].df;
          const next = new Array(aPrev.length).fill(0);
          for (let j = 0; j < aPrev.length; j++) {
            let sum = 0;
            for (let i = 0; i < delta.length; i++) sum += this.W[l][i][j] * delta[i];
            next[j] = sum * df(aPrev[j]);
          }
          delta = next;
        }
      }
    }
    return { g, L: L / X.length };
  }

  // One full-batch Adam step. Returns the loss before the step.
  step(X, Y) {
    const { g, L } = this.grad(X, Y);
    const N = X.length, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.t += 1;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    const upd = (P, G, M, V) => {
      const gr = G / N;
      const m = b1 * M + (1 - b1) * gr, v = b2 * V + (1 - b2) * gr * gr;
      return [P - (this.lr * (m / c1)) / (Math.sqrt(v / c2) + eps), m, v];
    };
    for (let l = 0; l < this.nLayers; l++) {
      for (let i = 0; i < this.W[l].length; i++) {
        for (let j = 0; j < this.W[l][i].length; j++) {
          const [p, m, v] = upd(this.W[l][i][j], g.W[l][i][j], this.m.W[l][i][j], this.v.W[l][i][j]);
          this.W[l][i][j] = p; this.m.W[l][i][j] = m; this.v.W[l][i][j] = v;
        }
        const [p, m, v] = upd(this.b[l][i], g.b[l][i], this.m.b[l][i], this.v.b[l][i]);
        this.b[l][i] = p; this.m.b[l][i] = m; this.v.b[l][i] = v;
      }
    }
    return L;
  }
}

// Datasets in the square [-1, 1]^2. Labels are 0 or 1.
export const datasets = {
  xor(n = 200, seed = 3) {
    const r = rng(seed), X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const sx = r() < 0.5 ? -1 : 1, sy = r() < 0.5 ? -1 : 1;
      X.push([sx * 0.5 + gaussian(r) * 0.15, sy * 0.5 + gaussian(r) * 0.15]);
      Y.push(sx * sy > 0 ? 1 : 0);
    }
    return { X, Y };
  },
  circles(n = 240, seed = 4) {
    const r = rng(seed), X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const inner = i % 2 === 0;
      const rad = inner ? r() * 0.35 : 0.6 + r() * 0.3;
      const th = r() * 2 * Math.PI;
      X.push([rad * Math.cos(th), rad * Math.sin(th)]);
      Y.push(inner ? 1 : 0);
    }
    return { X, Y };
  },
  spirals(n = 240, seed = 5) {
    const r = rng(seed), X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const cls = i % 2;
      const t = 0.15 + 0.85 * r();
      const th = t * 3 * Math.PI + cls * Math.PI;
      const rad = t * 0.9;
      X.push([rad * Math.cos(th) + gaussian(r) * 0.02, rad * Math.sin(th) + gaussian(r) * 0.02]);
      Y.push(cls);
    }
    return { X, Y };
  },
};
