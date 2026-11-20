// Numbers behind page 2.1 (fit by hand). Run: node tools/test-page-2-1.mjs
import { MODELS, loss } from '../docs/js/lib/s2-fit.js';
const L = MODELS.line, W = MODELS.wave;
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const r2 = (x) => x.toFixed(2), r3 = (x) => x.toFixed(3);
// Line: least squares.
const { X, Y } = L.data, n = X.length;
const mx = X.reduce((a, b) => a + b) / n, my = Y.reduce((a, b) => a + b) / n;
let sxy = 0, sxx = 0; X.forEach((x, i) => { sxy += (x - mx) * (Y[i] - my); sxx += (x - mx) ** 2; });
const a = sxy / sxx, b = my - a * mx;
console.log(`line: best a=${a.toFixed(3)} b=${b.toFixed(3)} loss=${loss(L, { a, b }).toFixed(3)}; at start (0,0) loss=${loss(L, { a: 0, b: 0 }).toFixed(3)}; at truth loss=${loss(L, L.truth).toFixed(3)}`);
check('each model has its own 20 points', L.data.X.length === 20 && W.data.X.length === 20, `${L.data.X.length} ${W.data.X.length}`);
// Task 1 notice: "About 0.32, at slope 0.91 and intercept 0.32."
check('t1: lowest loss for the line is 0.32, at slope 0.91 and intercept 0.32', r2(loss(L, { a, b })) === '0.32' && r2(a) === '0.91' && r2(b) === '0.32', `${loss(L, { a, b })} at ${a}, ${b}`);
check('t1: the choice "About 0.3" is the closest of 0, about 0.3, about 1', Math.abs(loss(L, { a, b }) - 0.3) < 0.05, loss(L, { a, b }));
// Wave: global search on a grid, then refine.
let best = { v: Infinity };
for (let A = -3; A <= 3; A += 0.02) for (let B = 0; B <= 4; B += 0.005) for (let C = -2; C <= 2; C += 0.05) { const v = loss(W, { a: A, b: B, c: C }); if (v < best.v) best = { v, a: A, b: B, c: C }; }
console.log(`wave: best on grid a=${best.a.toFixed(2)} b=${best.b.toFixed(3)} c=${best.c.toFixed(2)} loss=${best.v.toFixed(3)}; truth (1.5, 1.5, 0.5) loss=${loss(W, W.truth).toFixed(3)}; start (1,1,0) loss=${loss(W, { a: 1, b: 1, c: 0 }).toFixed(3)}`);
// Task 2: "Its points were made from a = 1.5, b = 1.5, c = 0.5" and notice: "The setting that made the points has loss 0.120.
// Near a = 1.66, b = 1.47, c = 0.35 the loss is 0.078."
check('t2: the wave points were made from a = 1.5, b = 1.5, c = 0.5', W.truth.a === 1.5 && W.truth.b === 1.5 && W.truth.c === 0.5, JSON.stringify(W.truth));
check('t2: the setting that made the points has loss 0.120', r3(loss(W, W.truth)) === '0.120', loss(W, W.truth));
check('t2: the lowest loss on the search grid is at a = 1.66, b = 1.47, c = 0.35', r2(best.a) === '1.66' && r2(best.b) === '1.47' && r2(best.c) === '0.35', `${best.a} ${best.b} ${best.c}`);
check('t2: the lowest loss is 0.078, at the grid point and at the quoted setting (1.66, 1.47, 0.35)', r3(best.v) === '0.078' && r3(loss(W, { a: 1.66, b: 1.47, c: 0.35 })) === '0.078', `${best.v} ${loss(W, { a: 1.66, b: 1.47, c: 0.35 })}`);
const mapMins = {};
// Wave heat map at c = 0.5 (the page's grid: b in [0, 4], a in [-3, 3]); strict local minima on a 161 x 121 grid.
for (const c of [0.5, best.c]) {
  const na = 121, nb = 161, g = [];
  for (let i = 0; i < na; i++) { g.push([]); for (let j = 0; j < nb; j++) g[i].push(loss(W, { a: -3 + (6 * i) / (na - 1), b: (4 * j) / (nb - 1), c })); }
  const mins = [];
  for (let i = 1; i < na - 1; i++) for (let j = 1; j < nb - 1; j++) {
    let ok = true;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) if ((di || dj) && g[i + di][j + dj] <= g[i][j]) ok = false;
    if (ok) (mapMins[c] ||= []).push({ a: -3 + (6 * i) / (na - 1), b: (4 * j) / (nb - 1), v: g[i][j] });
    if (ok) mins.push(`(a ${(-3 + (6 * i) / (na - 1)).toFixed(2)}, b ${((4 * j) / (nb - 1)).toFixed(3)}): ${g[i][j].toFixed(3)}`);
  }
  console.log(`wave heat map at c=${c.toFixed(2)}: local minima`, mins.join('; '));
}
// Task 3 notice: "Yes, several. For example, near a = -0.55, b = 0.70 the loss is 1.32, lower than at any nearby setting,
// while the best setting has loss 0.10."
{
  const m = mapMins[0.5], lowest = m.reduce((p, q) => (q.v < p.v ? q : p)), other = m.filter((q) => q !== lowest);
  check('t3: at c = 0.5 the map has several low points other than the best one (at least 3)', other.length >= 3, other.length);
  const q = other.find((q) => r2(q.a) === '-0.55' && r2(q.b) === '0.70');
  check('t3: one of them is at a = -0.55, b = 0.70, with loss 1.32 (to within 0.005)', !!q && Math.abs(q.v - 1.32) <= 0.005, q ? q.v : 'none');
  check('t3: the best setting at c = 0.5 has loss 0.10', r2(lowest.v) === '0.10', lowest.v);
}
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
