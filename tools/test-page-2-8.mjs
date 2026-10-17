// Numbers behind page 2.8 (degree slider). Run: node tools/test-page-2-8.mjs
import { sample, fit, mse, legendre, pinvSolve } from '../docs/js/lib/s2-poly.js';
// 1. The solver: least squares when overdetermined, minimum-norm interpolation when underdetermined.
{
  const d = sample(12, 0.3, 5);
  const f11 = fit(d, 11), f20 = fit(d, 20);
  console.log('train MSE at degree 11:', mse(f11.f, d).toExponential(1), ' at degree 20:', mse(f20.f, d).toExponential(1));
  // The min-norm solution has a smaller norm than another interpolating solution (add a null-space vector).
  const A = d.X.map((x) => legendre(x, 20)), w = f20.w;
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const r = Array.from({ length: 21 }, (_, i) => Math.sin(i + 1)), Ar = A.map((row) => row.reduce((s, a, j) => s + a * r[j], 0));
  const z = pinvSolve(A, Ar), nul = r.map((v, i) => v - z[i]); // r minus its row-space part
  const w2 = w.map((v, i) => v + nul[i]);
  const resid = Math.max(...A.map((row, i) => Math.abs(row.reduce((s, a, j) => s + a * w2[j], 0) - d.Y[i])));
  console.log(`degree 20: norm of min-norm fit ${norm(w).toFixed(3)}; another interpolating fit (residual ${resid.toExponential(1)}) has norm ${norm(w2).toFixed(3)}`);
}
function curve(n, noise, R = 200) {
  const test = sample(2000, noise, 999), per = Array.from({ length: 31 }, () => []), best = [];
  for (let s = 1; s <= R; s++) {
    const d = sample(n, noise, s), errs = [];
    for (let deg = 0; deg <= 30; deg++) { const e = mse(fit(d, deg).f, test); per[deg].push(e); errs.push(e); }
    best.push(errs.indexOf(Math.min(...errs)));
  }
  const med = per.map((a) => a.sort((p, q) => p - q)[Math.floor(R / 2)]);
  const hist = {}; best.forEach((b) => { hist[b] = (hist[b] || 0) + 1; });
  return { med, hist };
}
const fmt = (v) => (v < 100 ? v.toFixed(3) : v.toExponential(1));
for (const [n, noise] of [[12, 0.3], [24, 0.3]]) {
  const { med, hist } = curve(n, noise);
  const peak = med.indexOf(Math.max(...med)), low = med.indexOf(Math.min(...med));
  console.log(`n=${n}, noise ${noise}, 200 datasets: median test MSE by degree:\n  ${med.map((v, i) => `${i}:${fmt(v)}`).join(' ')}`);
  console.log(`  peak of the median at degree ${peak}; lowest at degree ${low} (${fmt(med[low])}); degree 30: ${fmt(med[30])}`);
  console.log('  best degree per dataset (degree: count):', JSON.stringify(hist));
  const after = med.slice(peak + 1); let falls = true; for (let i = 1; i < after.length; i++) if (after[i] > after[i - 1] * 1.05) falls = false;
  console.log('  median falls (within 5%) at every degree after the peak:', falls);
}
// The page's default dataset (seed 1) and the static figure (same seed).
{
  const d = sample(12, 0.3, 1), test = sample(2000, 0.3, 999);
  const errs = Array.from({ length: 31 }, (_, deg) => mse(fit(d, deg).f, test));
  console.log('default dataset: test MSE by degree', errs.map((v, i) => `${i}:${fmt(v)}`).join(' '));
  for (const deg of [1, 3, 15]) { const f = fit(d, deg); let mx = 0; for (let x = -1; x <= 1; x += 0.001) mx = Math.max(mx, Math.abs(f.f(x))); console.log(`  figure degree ${deg}: train ${fmt(mse(f.f, d))}, test ${fmt(mse(f.f, test))}, largest |f| on [-1,1] ${mx.toFixed(2)}`); }
}
// Where each single set of points has its highest test error (task 2 notice): 100 sets each of 12 and 24 points.
for (const n of [12, 24]) {
  const test = sample(2000, 0.3, 999), hist = {};
  for (let s = 1; s <= 100; s++) {
    const d = sample(n, 0.3, s), e = Array.from({ length: 31 }, (_, deg) => mse(fit(d, deg).f, test));
    const pk = e.indexOf(Math.max(...e)); hist[pk] = (hist[pk] || 0) + 1;
  }
  console.log(`n=${n}: degree of highest test error per set (degree: count of 100):`, JSON.stringify(hist));
}
