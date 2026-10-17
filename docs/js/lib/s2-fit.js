// Data and models for page 2.1 (fit by hand). No DOM; also used by tools/test-page-2-1.mjs.
import { rng, gaussian } from './mlp.js';

function sample(f, sd, seed, lo, hi, n = 20) {
  const r = rng(seed), X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const x = lo + ((hi - lo) * (i + 0.5)) / n + (r() - 0.5) * 0.1 * ((hi - lo) / n);
    X.push(x); Y.push(f(x) + gaussian(r) * sd);
  }
  return { X, Y };
}

// Each model has its own 20 points, drawn from a member of its family plus noise.
export const MODELS = {
  line: {
    name: 'Line', formula: 'y = a x + b',
    params: [{ k: 'a', name: 'slope a', min: -3, max: 3, step: 0.01, start: 0 }, { k: 'b', name: 'intercept b', min: -3, max: 3, step: 0.01, start: 0 }],
    f: (p, x) => p.a * x + p.b,
    truth: { a: 0.8, b: 0.5 },
    data: sample((x) => 0.8 * x + 0.5, 0.6, 21, -3, 3),
    xDomain: [-3.2, 3.2], yDomain: [-4, 4],
    map: ['a', 'b'], // heat map: first key on the horizontal axis, second on the vertical
  },
  wave: {
    name: 'Wave', formula: 'y = a sin(b x) + c',
    params: [{ k: 'a', name: 'amplitude a', min: -3, max: 3, step: 0.01, start: 1 }, { k: 'b', name: 'frequency b', min: 0, max: 4, step: 0.005, start: 1 },
      { k: 'c', name: 'shift c', min: -2, max: 2, step: 0.01, start: 0 }],
    f: (p, x) => p.a * Math.sin(p.b * x) + p.c,
    truth: { a: 1.5, b: 1.5, c: 0.5 },
    data: sample((x) => 1.5 * Math.sin(1.5 * x) + 0.5, 0.3, 22, 0, 6),
    xDomain: [-0.2, 6.2], yDomain: [-2.5, 3],
    map: ['b', 'a'],
  },
};

export function loss(model, p) {
  const { X, Y } = model.data;
  let s = 0;
  for (let i = 0; i < X.length; i++) s += (model.f(p, X[i]) - Y[i]) ** 2;
  return s / X.length;
}
