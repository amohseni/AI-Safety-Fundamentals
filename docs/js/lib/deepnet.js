// A deep multilayer perceptron for measuring gradients layer by layer (page 1.3).
// Pure JavaScript, no DOM, so it runs in the browser and in Node tests.
// Architecture: 2 inputs -> L hidden layers of `width` units with the chosen activation function
// -> 1 output with a sigmoid. Loss: mean binary cross-entropy over the whole dataset.
// Initial parameters: each weight is drawn from a normal distribution with standard deviation
// gain / sqrt(fan-in); biases start at 0.
import { rng, gaussian } from './mlp.js';

export const ACTS = {
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), df: (a) => a * (1 - a) }, // df takes the activation's output
  tanh: { f: Math.tanh, df: (a) => 1 - a * a },
  relu: { f: (z) => (z > 0 ? z : 0), df: (a) => (a > 0 ? 1 : 0) },
};

export class DeepNet {
  constructor({ depth = 10, width = 16, act = 'sigmoid', gain = 1, seed = 1, inputs = 2 } = {}) {
    this.depth = depth; this.width = width; this.act = act; this.gain = gain;
    const rand = rng(seed);
    this.sizes = [inputs, ...Array(depth).fill(width), 1];
    this.W = []; this.b = [];
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1];
      const sd = gain / Math.sqrt(nIn);
      const M = new Float64Array(nOut * nIn);
      for (let k = 0; k < M.length; k++) M[k] = gaussian(rand) * sd;
      this.W.push(M);
      this.b.push(new Float64Array(nOut));
    }
  }

  // Forward pass for one input; returns the activations of every layer (index 0 is the input).
  forward(x) {
    const A = [Float64Array.from(x)];
    const L = this.W.length;
    for (let l = 0; l < L; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], b = this.b[l], a = A[l];
      const f = l === L - 1 ? ACTS.sigmoid.f : ACTS[this.act].f;
      const out = new Float64Array(nOut);
      for (let i = 0; i < nOut; i++) {
        let z = b[i];
        for (let j = 0; j < nIn; j++) z += W[i * nIn + j] * a[j];
        out[i] = f(z);
      }
      A.push(out);
    }
    return A;
  }

  // Full-batch gradients by backpropagation. Returns the loss, the gradient arrays, and for each
  // layer with parameters the root-mean-square of the gradient of the loss with respect to that
  // layer's weights (rms[0] is hidden layer 1; the last entry is the output layer).
  gradients(X, Y) {
    const L = this.W.length;
    const gW = this.W.map((M) => new Float64Array(M.length));
    const gb = this.b.map((v) => new Float64Array(v.length));
    const df = ACTS[this.act].df;
    let loss = 0;
    for (let n = 0; n < X.length; n++) {
      const A = this.forward(X[n]);
      const p = A[L][0];
      const pc = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
      loss -= Y[n] * Math.log(pc) + (1 - Y[n]) * Math.log(1 - pc);
      let delta = new Float64Array([p - Y[n]]); // sigmoid output with cross-entropy
      for (let l = L - 1; l >= 0; l--) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], a = A[l], W = this.W[l];
        for (let i = 0; i < nOut; i++) {
          gb[l][i] += delta[i];
          for (let j = 0; j < nIn; j++) gW[l][i * nIn + j] += delta[i] * a[j];
        }
        if (l > 0) {
          const next = new Float64Array(nIn);
          for (let j = 0; j < nIn; j++) {
            let s = 0;
            for (let i = 0; i < nOut; i++) s += W[i * nIn + j] * delta[i];
            next[j] = s * df(a[j]);
          }
          delta = next;
        }
      }
    }
    const N = X.length;
    gW.forEach((M) => { for (let k = 0; k < M.length; k++) M[k] /= N; });
    gb.forEach((v) => { for (let k = 0; k < v.length; k++) v[k] /= N; });
    const rms = gW.map((M) => Math.sqrt(M.reduce((s, g) => s + g * g, 0) / M.length));
    return { loss: loss / N, gW, gb, rms };
  }

  // One step of plain gradient descent; returns the gradient report from before the step.
  step(X, Y, lr = 0.5) {
    const g = this.gradients(X, Y);
    this.W.forEach((M, l) => { for (let k = 0; k < M.length; k++) M[k] -= lr * g.gW[l][k]; });
    this.b.forEach((v, l) => { for (let k = 0; k < v.length; k++) v[k] -= lr * g.gb[l][k]; });
    return g;
  }
}
