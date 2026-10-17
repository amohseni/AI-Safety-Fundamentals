// Numbers quoted on page 5.1. Run: node tools/test-page-5-1.mjs
import { TARGETS, fitEven, rms, deepSaw, sawtooth, pieces, handNet } from '../docs/js/lib/s45-approx.js';
import { rng } from '../docs/js/lib/mlp.js';
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };

// 1. The bump is exactly three units, and no two units can make it: a bump has three slope changes,
//    and each unit adds exactly one.
check(rms(handNet(0, [{ c: 0.2, a: 2.5 }, { c: 0.4, a: -5 }, { c: 0.6, a: 2.5 }]), TARGETS.bump) < 1e-12, 'three units (slopes 2.5, -5, 2.5 at 0.2, 0.4, 0.6) give the bump exactly');
check(pieces(TARGETS.bump, 10000) === 4, `the bump has ${pieces(TARGETS.bump, 10000)} linear pieces, so 3 kinks (counted on a grid of step 0.0001, which contains the kinks)`);
{ // best fit with two freely placed units (grid search over kinks, least squares for the rest)
  let best = Infinity;
  for (let i = 0; i <= 100; i++) for (let j = i + 1; j <= 100; j++) best = Math.min(best, rms(fitEven(TARGETS.bump, 2, [i / 100, j / 100]).f, TARGETS.bump));
  console.log(`     best two-unit fit to the bump (kinks on a 0.01 grid): rms ${best.toFixed(3)}`);
}

// 2. The jagged target: error with evenly spaced kinks against number of units.
{
  const errs = []; for (let m = 1; m <= 128; m++) errs.push(rms(fitEven(TARGETS.jagged, m).f, TARGETS.jagged));
  const first = errs.findIndex((e) => e < 0.01) + 1;
  const stays = errs.slice(first - 1).every((e) => e < 0.01);
  check(first >= 30 && first <= 60, `jagged: first m with rms < 0.01 is ${first}; stays below for every larger m up to 128: ${stays}`);
  console.log(`     jagged rms at m = 4, 8, 16, 32, 64: ${[4, 8, 16, 32, 64].map((m) => errs[m - 1].toFixed(3)).join(', ')}`);
  const e2 = []; for (let m = 1; m <= 64; m++) e2.push(rms(fitEven(TARGETS.smooth, m).f, TARGETS.smooth));
  console.log(`     smooth: first m with rms < 0.01 is ${e2.findIndex((e) => e < 0.01) + 1}`);
}

// 3. Depth: k layers of 2 units compute a sawtooth with 2^k pieces; one hidden layer needs 2^k - 1.
for (let k = 1; k <= 6; k++) {
  const d = deepSaw(k);
  const exact = rms(d, sawtooth(k)) < 1e-12 && pieces(d) === 2 ** k;
  const m = 2 * k, even = rms(fitEven(d, m).f, d);
  const r = rng(k); let best = even;
  for (let it = 0; it < 2000; it++) { const cs = [0, ...Array.from({ length: m - 1 }, () => r())].sort((a, b) => a - b); best = Math.min(best, rms(fitEven(d, m, cs).f, d)); }
  const need = 2 ** k - 1;
  const exactShallow = rms(fitEven(d, need + 1).f, d);
  check(exact, `k = ${k}: deep net (${m} units) gives the sawtooth with ${2 ** k} pieces exactly; one layer of ${m} units: even kinks rms ${even.toFixed(3)}, best of 2,000 random kink sets ${best.toFixed(3)}; one layer needs ${need} units (with ${need + 1} even kinks rms ${exactShallow.toExponential(1)})`);
}
process.exit(fail ? 1 : 0);
