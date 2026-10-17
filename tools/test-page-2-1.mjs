// Numbers behind page 2.1 (fit by hand). Run: node tools/test-page-2-1.mjs
import { MODELS, loss } from '../docs/js/lib/s2-fit.js';
const L = MODELS.line, W = MODELS.wave;
// Line: least squares.
const { X, Y } = L.data, n = X.length;
const mx = X.reduce((a, b) => a + b) / n, my = Y.reduce((a, b) => a + b) / n;
let sxy = 0, sxx = 0; X.forEach((x, i) => { sxy += (x - mx) * (Y[i] - my); sxx += (x - mx) ** 2; });
const a = sxy / sxx, b = my - a * mx;
console.log(`line: best a=${a.toFixed(3)} b=${b.toFixed(3)} loss=${loss(L, { a, b }).toFixed(3)}; at start (0,0) loss=${loss(L, { a: 0, b: 0 }).toFixed(3)}; at truth loss=${loss(L, L.truth).toFixed(3)}`);
// Wave: global search on a grid, then refine.
let best = { v: Infinity };
for (let A = -3; A <= 3; A += 0.02) for (let B = 0; B <= 4; B += 0.005) for (let C = -2; C <= 2; C += 0.05) { const v = loss(W, { a: A, b: B, c: C }); if (v < best.v) best = { v, a: A, b: B, c: C }; }
console.log(`wave: best on grid a=${best.a.toFixed(2)} b=${best.b.toFixed(3)} c=${best.c.toFixed(2)} loss=${best.v.toFixed(3)}; truth (1.5, 1.5, 0.5) loss=${loss(W, W.truth).toFixed(3)}; start (1,1,0) loss=${loss(W, { a: 1, b: 1, c: 0 }).toFixed(3)}`);
// Wave heat map at c = 0.5 (the page's grid: b in [0, 4], a in [-3, 3]); strict local minima on a 161 x 121 grid.
for (const c of [0.5, best.c]) {
  const na = 121, nb = 161, g = [];
  for (let i = 0; i < na; i++) { g.push([]); for (let j = 0; j < nb; j++) g[i].push(loss(W, { a: -3 + (6 * i) / (na - 1), b: (4 * j) / (nb - 1), c })); }
  const mins = [];
  for (let i = 1; i < na - 1; i++) for (let j = 1; j < nb - 1; j++) {
    let ok = true;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) if ((di || dj) && g[i + di][j + dj] <= g[i][j]) ok = false;
    if (ok) mins.push(`(a ${(-3 + (6 * i) / (na - 1)).toFixed(2)}, b ${((4 * j) / (nb - 1)).toFixed(3)}): ${g[i][j].toFixed(3)}`);
  }
  console.log(`wave heat map at c=${c.toFixed(2)}: local minima`, mins.join('; '));
}
