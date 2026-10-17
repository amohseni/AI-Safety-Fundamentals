// Numbers quoted on page 8.1, recomputed with the page's own model (docs/js/lib/8-selection.js).
// Exact curves are checked against Monte Carlo for small k. Run: node tools/test-page-8-1.mjs
import { selectionCurve, errSample, trueValue, fitProxy, optimize } from '../docs/js/lib/8-selection.js';
import { rng, gaussian } from '../docs/js/lib/mlp.js';

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
  for (const k of [10, 100]) { const e = at(err, k), m = mc(err, k, 20000); console.log(' ', err.kind, 'k', k, 'exact V', f2(e.v), 'U', f2(e.u), '| MC V', f2(m[0]), 'U', f2(m[1])); }
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
    console.log('Figure', err.kind, 'top 1% mean proxy', (top.reduce((s, q) => s + q[0], 0) / top.length).toFixed(1), 'mean true', (top.reduce((s, q) => s + q[1], 0) / top.length).toFixed(1));
  }
}
// "Why the tails decide": E[V | U = u] for Student t errors (3 degrees of freedom) rises, then falls toward 0.
{
  const c = selectionCurve({ kind: 't', s: 1, df: 3 });
  console.log('E[V | U = u], t errors:', [0, 1, 2, 3, 4, 6, 8, 12, 20, 50, 200].map((u) => `${u}: ${c.condMean(u).toFixed(3)}`).join(', '));
}
