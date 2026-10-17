// Loss surfaces and optimizers for page 2.6. No DOM; also used by tools/test-page-2-6.mjs.
import { rng, gaussian } from './mlp.js';

const g2 = (x, y, cx, cy, s) => Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
export const SURFACES = {
  bowl: {
    name: 'Round bowl', f: (x, y) => 0.5 * (x * x + y * y), g: (x, y) => [x, y],
    xd: [-3, 3], yd: [-3, 3], start: [-2.5, 2], lr: 0.1, min: 0, levels: 'log',
  },
  valley: {
    name: 'Narrow valley', f: (x, y) => 0.5 * (x * x + 100 * y * y), g: (x, y) => [x, 100 * y],
    xd: [-3, 3], yd: [-3, 3], start: [-2.5, 1], lr: 0.01, min: 0, levels: 'log',
  },
  rosenbrock: {
    name: 'Curved valley', f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    xd: [-2, 2], yd: [-1, 3], start: [-1.5, 2], lr: 0.001, min: 0, levels: 'log',
  },
  // A narrow, deep minimum at (-1.2, 0) and a wide, shallower one at (1.4, 0), on a gentle bowl.
  two: {
    name: 'Two minima',
    f: (x, y) => 0.03 * (x * x + y * y) - 1.0 * g2(x, y, -1.2, 0, 0.3) - 0.8 * g2(x, y, 1.4, 0, 1.0),
    g: (x, y) => {
      const a = g2(x, y, -1.2, 0, 0.3), b = g2(x, y, 1.4, 0, 1.0);
      return [0.06 * x + (1.0 * a * (x + 1.2)) / 0.09 + (0.8 * b * (x - 1.4)) / 1.0, 0.06 * y + (1.0 * a * y) / 0.09 + (0.8 * b * y) / 1.0];
    },
    xd: [-3, 3], yd: [-3, 3], start: [-0.7, 0.5], lr: 0.1, min: null, levels: 'linear',
  },
};
// Which minimum of the two-minimum surface a point drains to under noise-free gradient descent.
export function basin(p) {
  const r = makeRun(SURFACES.two, { kind: 'gd', lr: 0.05, start: p });
  for (let i = 0; i < 3000; i++) r.step();
  return r.p[0] < 0 ? 'narrow' : 'wide';
}
// Lowest value of each surface (found numerically for the two-minimum surface).
(function () {
  const s = SURFACES.two; let best = Infinity;
  for (let x = -2; x <= 2; x += 0.001) best = Math.min(best, s.f(x, 0));
  s.min = best;
})();

// Optimizer state machine. kind: 'gd' | 'momentum' | 'adam'. noise: standard deviation added to each gradient component.
export function makeRun(surface, { kind = 'gd', lr = 0.05, noise = 0, seed = 1, start = surface.start, beta = 0.9 } = {}) {
  const r = rng(seed);
  const st = { p: start.slice(), v: [0, 0], m: [0, 0], s: [0, 0], t: 0, path: [start.slice()], losses: [surface.f(...start)], diverged: false };
  st.step = () => {
    if (st.diverged) return;
    let g = surface.g(st.p[0], st.p[1]);
    if (noise > 0) g = [g[0] + noise * gaussian(r), g[1] + noise * gaussian(r)];
    st.t++;
    if (kind === 'gd') { st.p = [st.p[0] - lr * g[0], st.p[1] - lr * g[1]]; }
    else if (kind === 'momentum') { st.v = [beta * st.v[0] + g[0], beta * st.v[1] + g[1]]; st.p = [st.p[0] - lr * st.v[0], st.p[1] - lr * st.v[1]]; }
    else {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      st.m = [0, 1].map((i) => b1 * st.m[i] + (1 - b1) * g[i]);
      st.s = [0, 1].map((i) => b2 * st.s[i] + (1 - b2) * g[i] * g[i]);
      const c1 = 1 - b1 ** st.t, c2 = 1 - b2 ** st.t;
      st.p = [0, 1].map((i) => st.p[i] - (lr * (st.m[i] / c1)) / (Math.sqrt(st.s[i] / c2) + eps));
    }
    const L = surface.f(st.p[0], st.p[1]);
    if (!Number.isFinite(L) || Math.abs(st.p[0]) > 1e6 || Math.abs(st.p[1]) > 1e6) st.diverged = true;
    st.path.push(st.p.slice()); st.losses.push(L);
  };
  return st;
}
