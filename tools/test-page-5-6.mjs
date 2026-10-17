// Numbers quoted on page 5.6. Run: node tools/test-page-5-6.mjs
import { SIZES, perDigit, metric, riseWidth, sizeAt } from '../docs/js/lib/s45-emergence.js';
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };
const A = 0.5;
const fmtN = (n) => n.toExponential(1);
for (const k of [1, 5, 10, 20]) {
  console.log(`     k = ${k}: exact match at the 11 sizes: ${SIZES.map((N) => metric('exact', N, A, k).toFixed(3)).join(' ')}`);
}
console.log(`     per-digit accuracy at the 11 sizes: ${SIZES.map((N) => perDigit(N, A).toFixed(3)).join(' ')}`);
// Width of the 10%-90% rise is the same for every k (closed form and numerical check).
for (const k of [1, 10, 100]) {
  const lo = sizeAt(0.1, A, k), hi = sizeAt(0.9, A, k);
  const okLo = Math.abs(metric('exact', lo, A, k) - 0.1) < 1e-9, okHi = Math.abs(metric('exact', hi, A, k) - 0.9) < 1e-9;
  check(okLo && okHi && Math.abs(Math.log10(hi / lo) - riseWidth(A)) < 1e-9, `k = ${k}: exact match rises from 10% at N = ${fmtN(lo)} to 90% at N = ${fmtN(hi)}, a width of ${Math.log10(hi / lo).toFixed(2)} powers of ten`);
}
// Largest jump between neighboring models, for k = 1 and k = 10.
for (const k of [1, 10]) {
  const v = SIZES.map((N) => metric('exact', N, A, k));
  let best = 0, at = 0; for (let i = 1; i < v.length; i++) if (v[i] - v[i - 1] > best) { best = v[i] - v[i - 1]; at = i; }
  const under = v.filter((x) => x < 0.05).length;
  console.log(`     k = ${k}: largest jump between neighboring models ${best.toFixed(2)} (from ${fmtN(SIZES[at - 1])} to ${fmtN(SIZES[at])}); models below 5%: ${under}`);
}
// Log-likelihood is a power law: a straight line on log-log axes.
{
  const k = 10, v = SIZES.map((N) => -metric('loglik', N, A, k));
  const slopes = v.slice(1).map((y, i) => Math.log10(y / v[i]) / Math.log10(SIZES[i + 1] / SIZES[i]));
  check(slopes.every((s) => Math.abs(s + A) < 1e-9), `minus log-likelihood falls as a power law, slope ${slopes[0].toFixed(3)} on log-log axes`);
}
check(Math.abs(0.9 ** 10 - 0.3487) < 1e-4, `0.9^10 = ${(0.9 ** 10).toFixed(4)}`);
process.exit(fail ? 1 : 0);
