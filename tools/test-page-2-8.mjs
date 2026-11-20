// Numbers behind page 2.8 (degree slider). Run: node tools/test-page-2-8.mjs
import { sample, fit, mse, legendre, pinvSolve } from '../docs/js/lib/s2-poly.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const sci = (v) => v.toExponential(1);
// 1. The solver: least squares when overdetermined, minimum-norm interpolation when underdetermined.
{
  const d = sample(12, 0.3, 5);
  const f11 = fit(d, 11), f20 = fit(d, 20);
  console.log('train MSE at degree 11:', mse(f11.f, d).toExponential(1), ' at degree 20:', mse(f20.f, d).toExponential(1));
  // "With 12 points, degree 11 is the lowest degree that can pass through every point exactly."
  check('with 12 points, degree 11 passes through every point (training error below 1e-12) and degree 10 does not (above 1e-4)', mse(f11.f, d) < 1e-12 && mse(f20.f, d) < 1e-12 && mse(fit(d, 10).f, d) > 1e-4, `degree 10: ${mse(fit(d, 10).f, d).toExponential(1)}`);
  // The min-norm solution has a smaller norm than another interpolating solution (add a null-space vector).
  const A = d.X.map((x) => legendre(x, 20)), w = f20.w;
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const r = Array.from({ length: 21 }, (_, i) => Math.sin(i + 1)), Ar = A.map((row) => row.reduce((s, a, j) => s + a * r[j], 0));
  const z = pinvSolve(A, Ar), nul = r.map((v, i) => v - z[i]); // r minus its row-space part
  const w2 = w.map((v, i) => v + nul[i]);
  const resid = Math.max(...A.map((row, i) => Math.abs(row.reduce((s, a, j) => s + a * w2[j], 0) - d.Y[i])));
  console.log(`degree 20: norm of min-norm fit ${norm(w).toFixed(3)}; another interpolating fit (residual ${resid.toExponential(1)}) has norm ${norm(w2).toFixed(3)}`);
  check('past degree 11 the fit is the interpolating polynomial with the smallest sum of squared coefficients (smaller than another interpolating fit)', resid < 1e-9 && norm(w) < norm(w2), `${norm(w)} < ${norm(w2)}`);
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
  return { med, hist, best };
}
const fmt = (v) => (v < 100 ? v.toFixed(3) : v.toExponential(1));
const curves = {}, peaks = {};
for (const [n, noise] of [[12, 0.3], [24, 0.3]]) {
  const { med, hist } = curve(n, noise);
  curves[n] = { med, hist };
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
  // Caption: "Degree 3 follows it (test error 0.12). Degree 15 passes through every point but swings far off the chart
  // between them, as high as 48 (test error 88)."
  const f15 = fit(d, 15); let mx15 = 0; for (let x = -1; x <= 1; x += 0.001) mx15 = Math.max(mx15, Math.abs(f15.f(x)));
  check('caption: degree 3 has test error 0.12', errs[3].toFixed(2) === '0.12', errs[3]);
  check('caption: degree 15 passes through every point, swings as high as 48, and has test error 88', mse(f15.f, d) < 1e-12 && Math.round(mx15) === 48 && Math.round(errs[15]) === 88, `${mse(f15.f, d).toExponential(1)} ${mx15} ${errs[15]}`);
  check('caption: degree 1 misses the pattern (training error more than twice that of degree 3)', mse(fit(d, 1).f, d) > 2 * mse(fit(d, 3).f, d), `${mse(fit(d, 1).f, d)} ${mse(fit(d, 3).f, d)}`);
  // Task 1 notice: "For the first set of points the best is degree 3, with test error 0.12, while the training error keeps
  // falling as the degree rises."
  const tr = Array.from({ length: 31 }, (_, deg) => mse(fit(d, deg).f, d));
  check('t1: for the first set of points the best degree is 3', errs.indexOf(Math.min(...errs)) === 3, errs.indexOf(Math.min(...errs)));
  check('t1: the training error falls at every degree from 0 to 11', tr.slice(1, 12).every((v, i) => v < tr[i]), tr.slice(0, 12).map(sci).join(' '));
  // Task 3 notice: "The training error is zero all the way from degree 11." (The chart draws errors below 0.0001 at the bottom edge.)
  check('t3: the training error is zero (below 1e-12) at every degree from 11 to 30', tr.slice(11).every((v) => v < 1e-12), sci(Math.max(...tr.slice(11))));
  // Check 1 feedback: "degrees 11 and 30 both have zero training error, and for the first set of points their test errors
  // are about 300,000 and 1.0."
  check('c1: for the first set of points the test errors at degrees 11 and 30 are about 300,000 and 1.0', sci(errs[11]) === '3.0e+5' && errs[30].toFixed(1) === '1.0', `${errs[11]} ${errs[30]}`);
  for (const deg of [1, 3, 15]) { const f = fit(d, deg); let mx = 0; for (let x = -1; x <= 1; x += 0.001) mx = Math.max(mx, Math.abs(f.f(x))); console.log(`  figure degree ${deg}: train ${fmt(mse(f.f, d))}, test ${fmt(mse(f.f, test))}, largest |f| on [-1,1] ${mx.toFixed(2)}`); }
}
// Where each single set of points has its highest test error (task 2 notice): 100 sets each of 12 and 24 points.
for (const n of [12, 24]) {
  const test = sample(2000, 0.3, 999), hist = {};
  for (let s = 1; s <= 100; s++) {
    const d = sample(n, 0.3, s), e = Array.from({ length: 31 }, (_, deg) => mse(fit(d, deg).f, test));
    const pk = e.indexOf(Math.max(...e)); hist[pk] = (hist[pk] || 0) + 1;
  }
  peaks[n] = hist;
  console.log(`n=${n}: degree of highest test error per set (degree: count of 100):`, JSON.stringify(hist));
}
// Task 1 notice: "Over 200 sets of 12 points we drew, the best degree was 3 in 133 and between 3 and 5 in 178."
{
  const h = curves[12].hist, n35 = (h[3] || 0) + (h[4] || 0) + (h[5] || 0);
  check('t1: over 200 sets of 12 points the best degree is 3 in 133 and between 3 and 5 in 178', h[3] === 133 && n35 === 178, `${h[3]} ${n35}`);
}
// Task 2 notice: "Near degree 23. ... Over 100 sets of points we drew, the highest test error was at degree 11 in 97 sets of
// 12 points, and at degree 23 in 54 sets of 24 points (between 22 and 24 in 84)."
{
  const a = peaks[12], b = peaks[24], n = (b[22] || 0) + (b[23] || 0) + (b[24] || 0);
  check('t2: the highest test error is at degree 11 in 97 of 100 sets of 12 points', a[11] === 97, a[11]);
  check('t2: with 24 points it is at degree 23 in 54 of 100 sets, and between 22 and 24 in 84', b[23] === 54 && n === 84, `${b[23]} ${n}`);
  const d = sample(24, 0.3, 1), test = sample(2000, 0.3, 999), e = Array.from({ length: 31 }, (_, deg) => mse(fit(d, deg).f, test));
  check("t2: for the page's first set of 24 points the test error is highest near degree 23 (22 to 24)", Math.abs(e.indexOf(Math.max(...e)) - 23) <= 1, e.indexOf(Math.max(...e)));
  check('t2: the median test error over 200 sets of 24 points peaks at degree 23', curves[24].med.indexOf(Math.max(...curves[24].med)) === 23, curves[24].med.indexOf(Math.max(...curves[24].med)));
}
// Task 3 notice: "Over 200 sets of 12 points, the median test error peaks at degree 11 (about 20 million), falls at every degree
// after that, and reaches 0.71 at degree 30. That is still well above the 0.13 at degree 3".
{
  const m = curves[12].med;
  check('t3: the median test error peaks at degree 11, at about 20 million (between 15 and 25 million)', m.indexOf(Math.max(...m)) === 11 && m[11] > 1.5e7 && m[11] < 2.5e7, m[11]);
  check('t3: the median falls at every degree after 11', m.slice(12).every((v, i) => v < m[11 + i]), m.slice(11).map(fmt).join(' '));
  check('t3: the median reaches 0.71 at degree 30, well above the 0.13 at degree 3', m[30].toFixed(2) === '0.71' && m[3].toFixed(2) === '0.13', `${m[30]} ${m[3]}`);
}
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
