// Numbers quoted on page 8.1, recomputed with the page's own model (docs/js/lib/8-selection.js).
// Exact curves are checked against Monte Carlo for small k. Run: node tools/test-page-8-1.mjs
import { selectionCurve, errSample, trueValue, fitProxy, optimize } from '../docs/js/lib/8-selection.js';
import { rng, gaussian } from '../docs/js/lib/mlp.js';

let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const f2 = (x) => x.toFixed(2);
const at = (err, k) => selectionCurve(err).at(k);
function peak(err, maxLog = 6, step = 0.01) {
  const c = selectionCurve(err); let best = -Infinity, bk = 1;
  for (let x = 0; x <= maxLog + 1e-9; x += step) { const k = Math.round(10 ** x); const v = c.at(k).v; if (v > best) { best = v; bk = k; } }
  return { k: bk, v: best };
}
function mc(err, k, trials, seed = 5) {
  const r = rng(seed), g = () => gaussian(r); let sv = 0, su = 0;
  for (let t = 0; t < trials; t++) { let bu = -Infinity, bv = 0; for (let i = 0; i < k; i++) { const v = g(), u = v + errSample(err, r, g); if (u > bu) { bu = u; bv = v; } } sv += bv; su += bu; }
  return [sv / trials, su / trials];
}

console.log('Exact curve against Monte Carlo (k = 10, 100):');
for (const err of [{ kind: 'normal', s: 1 }, { kind: 't', s: 1, df: 3 }, { kind: 'mix', s: 1, rate: 0.001, big: 30 }]) {
  for (const k of [10, 100]) { const e = at(err, k), m = mc(err, k, 20000); check(`the exact curve agrees with 20,000 simulated draws to within 0.05 (${err.kind}, k = ${k})`, Math.abs(e.v - m[0]) < 0.05 && Math.abs(e.u - m[1]) < 0.05, `${f2(e.v - m[0])} ${f2(e.u - m[1])}`); console.log(' ', err.kind, 'k', k, 'exact V', f2(e.v), 'U', f2(e.u), '| MC V', f2(m[0]), 'U', f2(m[1])); }
}
const N1 = { kind: 'normal', s: 1 }, N2 = { kind: 'normal', s: 2 }, T3 = { kind: 't', s: 1, df: 3 }, T10 = { kind: 't', s: 1, df: 10 }, T30 = { kind: 't', s: 1, df: 30 };
const MIX = { kind: 'mix', s: 1, rate: 0.001, big: 30 };
console.log('Task 1: normal s=1 at k=1e6', f2(at(N1, 1e6).u), f2(at(N1, 1e6).v), '(page: 6.88, 3.44); monotone:',
  [1, 10, 100, 1e3, 1e4, 1e5, 1e6].map((k) => at(N1, k).v).every((v, i, a) => i === 0 || v > a[i - 1]));
console.log('        normal s=2 at k=1e6', f2(at(N2, 1e6).u), f2(at(N2, 1e6).v), '(page: 10.87, 2.17)');
const p3 = peak(T3), p10 = peak(T10), p30 = peak(T30, 9, 0.05);
console.log('Task 2: t3 peak at k', p3.k, 'V', f2(p3.v), '(page: about 22, 0.88); at 1e6 V', f2(at(T3, 1e6).v), 'U', at(T3, 1e6).u.toFixed(1), '(page: 0.03, 140)');
console.log('        t10 peak at k', p10.k, '(page: near 900); t30 peak at k', p30.k.toExponential(1), '(page: near 20 million)');
const pm = peak(MIX); let mn = Infinity, mk = 0; const cm = selectionCurve(MIX);
for (let x = 2; x <= 6; x += 0.01) { const k = Math.round(10 ** x); const v = cm.at(k).v; if (v < mn) { mn = v; mk = k; } }
console.log('Task 3: mix k=100', f2(cm.at(100).v), '(page 1.70; normal 1.77:', f2(at(N1, 100).v) + ')', 'k=1e4', f2(cm.at(1e4).v), '(page 0.09)', 'min at k', mk, '(page: about 17,000)', 'k=1e6', f2(cm.at(1e6).v), '(page 0.11)');
const p = fitProxy(3), o = optimize(p, 12);
console.log('Task 4: fitted on [0,3], search to 12: picks', o.x, 'true', f2(o.truth), '(page: 12, -6); best in [0,3] true', f2(trueValue(3)), '(page 1.9)');
for (const R of [5, 8, 8.5, 12]) { const q = fitProxy(R); console.log('        range', R, 'slope', q.slope.toFixed(3), 'picks', optimize(q, 12).x); }
// Static figure: 10,000 options, top 1% by proxy (same seed and order as the page).
{
  const r = rng(3), g = () => gaussian(r);
  for (const err of [N1, T3]) {
    const pts = Array.from({ length: 10000 }, () => { const v = g(); return [v + errSample(err, r, g), v]; });
    const cut = pts.map((q) => q[0]).sort((a, b) => b - a)[99];
    const top = pts.filter((q) => q[0] >= cut);
    const mp = (top.reduce((s, q) => s + q[0], 0) / top.length).toFixed(1), mt = (top.reduce((s, q) => s + q[1], 0) / top.length).toFixed(1);
    if (err === N1) check('caption: with normal errors the top 1% of 10,000 have mean proxy score 3.9 and mean true value 2.0', top.length === 100 && mp === '3.9' && mt === '2.0', `${mp} ${mt}`);
    else check('caption: with Student t errors (3 degrees of freedom) their mean proxy score is 6.7 and their mean true value 0.7', top.length === 100 && mp === '6.7' && mt === '0.7', `${mp} ${mt}`);
    console.log('Figure', err.kind, 'top 1% mean proxy', (top.reduce((s, q) => s + q[0], 0) / top.length).toFixed(1), 'mean true', (top.reduce((s, q) => s + q[1], 0) / top.length).toFixed(1));
  }
}
// "Why the tails decide": E[V | U = u] for Student t errors (3 degrees of freedom) rises, then falls toward 0.
{
  const c = selectionCurve({ kind: 't', s: 1, df: 3 });
  console.log('E[V | U = u], t errors:', [0, 1, 2, 3, 4, 6, 8, 12, 20, 50, 200].map((u) => `${u}: ${c.condMean(u).toFixed(3)}`).join(', '));
}
// Task 1 notice: "It keeps rising, and it stays at exactly half the expected proxy score: at k = 1,000,000 the expected proxy score
// is 6.88 and the expected true value is 3.44. ... with errors of size 2, the true value at k = 1,000,000 is 2.17 and the proxy 10.87."
{
  const ks = [1, 2, 5, 10, 30, 100, 1e3, 1e4, 1e5, 1e6], c = ks.map((k) => at(N1, k));
  check('t1: with normal errors of size 1 the expected true value keeps rising from k = 1 to 1,000,000', c.every((e, i) => i === 0 || e.v > c[i - 1].v), c.map((e) => f2(e.v)).join(' '));
  check('t1: it stays at exactly half the expected proxy score (to 0.001)', c.every((e) => Math.abs(e.v - e.u / 2) < 1e-3), c.map((e) => (e.v - e.u / 2).toExponential(1)).join(' '));
  check('t1: at k = 1,000,000 the proxy score is 6.88 and the true value 3.44', f2(at(N1, 1e6).u) === '6.88' && f2(at(N1, 1e6).v) === '3.44', `${at(N1, 1e6).u} ${at(N1, 1e6).v}`);
  check('t1: with errors of size 2 the true value at k = 1,000,000 is 2.17 and the proxy 10.87', f2(at(N2, 1e6).v) === '2.17' && f2(at(N2, 1e6).u) === '10.87', `${at(N2, 1e6).v} ${at(N2, 1e6).u}`);
}
// Task 2 notice: "The true value peaks at 0.88 near k = 22 and then falls. At k = 1,000,000 it is 0.03 ... while the expected proxy
// score is 140. ... With 10 degrees of freedom the peak comes later, near k = 900; with 30, the true value is still rising at
// k = 1,000,000 and peaks only near k = 20 million."
check('t2: with Student t errors (3 degrees of freedom) the true value peaks at 0.88 near k = 22 (20 to 24)', f2(p3.v) === '0.88' && p3.k >= 20 && p3.k <= 24, `${p3.v} at ${p3.k}`);
check('t2: then it falls (k = 22, 100, 1e3, 1e4, 1e5, 1e6)', [22, 100, 1e3, 1e4, 1e5, 1e6].map((k) => at(T3, k).v).every((v, i, a) => i === 0 || v < a[i - 1]), [22, 100, 1e3, 1e4, 1e5, 1e6].map((k) => f2(at(T3, k).v)).join(' '));
check('t2: at k = 1,000,000 the true value is 0.03 and the proxy score 140', f2(at(T3, 1e6).v) === '0.03' && Math.round(at(T3, 1e6).u) === 140, `${at(T3, 1e6).v} ${at(T3, 1e6).u}`);
check('t2: with 10 degrees of freedom the peak is near k = 900 (850 to 950)', p10.k >= 850 && p10.k <= 950, p10.k);
check('t2: with 30 the true value is still rising at k = 1,000,000', at(T30, 1e6).v > at(T30, 5e5).v && at(T30, 2e6).v > at(T30, 1e6).v, `${at(T30, 5e5).v} ${at(T30, 1e6).v} ${at(T30, 2e6).v}`);
check('t2: and peaks near k = 20 million (15 to 25 million)', p30.k >= 1.5e7 && p30.k <= 2.5e7, p30.k);
// Task 3 prompt: "one option in 1,000 has an error 30 times the usual size". Notice: "Best of 100 gives an expected true value of
// 1.70, close to the 1.77 that normal errors give. Best of 10,000 gives 0.09. ... Past about k = 17,000 the true value rises again,
// very slowly (0.11 at 1,000,000)".
check('t3: rare huge errors: best of 100 gives 1.70; normal errors give 1.77', f2(cm.at(100).v) === '1.70' && f2(at(N1, 100).v) === '1.77', `${cm.at(100).v} ${at(N1, 100).v}`);
check('t3: best of 10,000 gives 0.09', f2(cm.at(1e4).v) === '0.09', cm.at(1e4).v);
check('t3: the true value is lowest at about k = 17,000 (16,000 to 18,000)', mk >= 16000 && mk <= 18000, mk);
check('t3: past that it rises again, very slowly, to 0.11 at 1,000,000', f2(cm.at(1e6).v) === '0.11' && cm.at(1e6).v > cm.at(1e5).v && cm.at(1e5).v > mn, `${mn} ${cm.at(1e5).v} ${cm.at(1e6).v}`);
// Panel note: "rises to a peak at x = 4 and then falls ... 24 noisy measurements". Task 4 notice: "the optimizer picks the largest
// design it may, 12, where the true value is -6. The best design between 0 and 3 has true value 1.9. Widening the data does not fix
// this: ... the optimizer always picks an end of its range, never the peak at 4."
{
  const xs = Array.from({ length: 121 }, (_, i) => i / 10), best = xs.reduce((a, x) => (trueValue(x) > trueValue(a) ? x : a));
  check('panel 2: the true value peaks at x = 4', best === 4, best);
  check('panel 2: the proxy is fitted to 24 measurements', p.xs.length === 24, p.xs.length);
  check('t4: fitted on 0 to 3 the line keeps rising, and the optimizer picks 12, where the true value is -6', p.slope > 0 && o.x === 12 && o.truth === -6, `${p.slope} ${o.x} ${o.truth}`);
  check('t4: the best design between 0 and 3 has true value 1.9', trueValue(3).toFixed(1) === '1.9' && trueValue(3) > trueValue(2.9), trueValue(3));
  const picks = [3, 5, 8.5, 12].map((R) => optimize(fitProxy(R), 12).x);
  check('t4: with data to 3, 5, 8.5, or 12 the optimizer always picks an end of its range (0 or 12), never the peak at 4', picks.every((x) => x === 0 || x === 12), picks.join(' '));
}
// "Why the tails decide": E[V | U = u] = u / (1 + s^2) for normal errors; with heavy-tailed errors it rises and then falls back
// toward zero.
{
  const cn = selectionCurve(N1), cn2 = selectionCurve(N2), ct = selectionCurve(T3), us = [0.5, 1, 2, 4, 6];
  check('with normal errors E[V | U = u] = u / (1 + s^2) (s = 1 and 2, to 0.001)', us.every((u) => Math.abs(cn.condMean(u) - u / 2) < 1e-3 && Math.abs(cn2.condMean(u) - u / 5) < 1e-3), us.map((u) => cn.condMean(u).toFixed(3)).join(' '));
  const v = [0, 1, 2, 3, 4, 6, 8, 12, 20, 50, 200].map((u) => ct.condMean(u));
  check('with Student t errors E[V | U = u] rises (to u = 3) and then falls back toward zero (0.02 at u = 200)', v.slice(1, 4).every((x, i) => x > v[i]) && v.slice(4).every((x, i) => x < v[i + 3]) && v.at(-1) < 0.03, v.map((x) => x.toFixed(3)).join(' '));
}
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
