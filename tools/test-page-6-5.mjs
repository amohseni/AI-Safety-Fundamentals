// Numbers quoted on page 6.5, recomputed with the page's own settings: the pentagon toy model from page 6.4
// (5 features, 2 dimensions, S = 0.9, equal importance, group 0), and a sparse autoencoder trained with Adam,
// learning rate 0.005, batch 256, 3,000 steps. Starting point k of the tool is SAE seed k (k = 1, 2, ...).
// Run: node tools/test-page-6-5.mjs
import fs from 'node:fs';
import { trainBest, SAE } from '../docs/js/lib/s6-toymodel.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const REP = {};

const toy = trainBest({ n: 5, m: 2, S: 0.9, r: 1, group: 0 });
console.log('source W (row-major, 2 x 5):', JSON.stringify(Array.from(toy.W, (v) => Math.round(v * 10000) / 10000)));
console.log('source b:', JSON.stringify(Array.from(toy.b, (v) => Math.round(v * 10000) / 10000)));
console.log('true features active per input on average: 5 x 0.1 = 0.5');

const SEEDS = 10, STEPS = 3000;
function run(size, lambda, seed) {
  const s = new SAE(toy, { size, lambda, seed, lr: 0.005 });
  for (let i = 0; i < STEPS; i++) s.step();
  const ev = s.evaluate();
  const rec = s.recovery(0.5, ev.alive);
  // Live units that point (cosine > 0.95) at a true feature that another live unit also points at.
  const owner = [];
  for (let k = 0; k < size; k++) {
    if (!ev.alive[k]) continue;
    const d = s.dir(k);
    let best = -1, arg = -1;
    for (let i = 0; i < 5; i++) { const w = toy.dir(i), L = Math.hypot(...w); const c = (w[0] * d[0] + w[1] * d[1]) / L; if (c > best) { best = c; arg = i; } }
    if (best > 0.95) owner.push(arg);
  }
  const dup = owner.length - new Set(owner).size;
  const misaligned = size - ev.dead - owner.length;
  return { recovered: rec.filter((r) => r.cos > 0.95).length, mean: rec.reduce((a, r) => a + r.cos, 0) / rec.length, l0: ev.l0, fvu: ev.fvu, dead: ev.dead, dup, misaligned };
}
const range = (a, k = 2) => (Math.min(...a) === Math.max(...a) ? Math.min(...a).toFixed(k) : `${Math.min(...a).toFixed(k)} to ${Math.max(...a).toFixed(k)}`);
function report(size, lambda) {
  const R = Array.from({ length: SEEDS }, (_, i) => run(size, lambda, i + 1));
  const all5 = R.filter((r) => r.recovered === 5).length;
  REP[`${size}|${lambda}`] = R;
  console.log(`size ${size}, penalty ${lambda}: recovered (cos > 0.95) ${range(R.map((r) => r.recovered), 0)}; all five in ${all5}/${SEEDS};`,
    `mean best cos ${range(R.map((r) => r.mean))}; active units ${range(R.map((r) => r.l0))}; rebuild error ${range(R.map((r) => r.fvu), 3)};`,
    `dead ${range(R.map((r) => r.dead), 0)}; duplicates ${range(R.map((r) => r.dup), 0)}; pointing elsewhere ${range(R.map((r) => r.misaligned), 0)}`);
}
// Task 1: penalty sweep at dictionary size 5.
for (const l of [0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1, 2]) report(5, l);
// Task 2: a dictionary smaller than the number of features.
for (const l of [0.3, 0.5, 1]) report(3, l);
// Task 3: doubling the dictionary.
for (const l of [0.3, 0.5]) report(10, l);
report(20, 0.5);

// Static figure: dictionary of 5 at penalty 0.5, starting point 1; one activation with features 1 and 3 active.
{
  const s = new SAE(toy, { size: 5, lambda: 0.5, seed: 1, lr: 0.005 });
  for (let i = 0; i < STEPS; i++) s.step();
  const x = [0.9, 0, 0.6, 0, 0];
  const h = toy.hidden(x);
  const f = s.encode(h);
  console.log('figure: h =', Array.from(h, (v) => v.toFixed(3)).join(', '), '; code f =', Array.from(f, (v) => v.toFixed(3)).join(', '));
  console.log('figure D (row-major, 2 x 5):', JSON.stringify(Array.from(s.D, (v) => Math.round(v * 10000) / 10000)), 'c:', JSON.stringify(Array.from(s.c, (v) => Math.round(v * 10000) / 10000)));
  // Caption: "feature 1 at value 0.9 and feature 3 at 0.6 ... Unit C matches feature 1 (cosine 0.997) and unit B matches
  // feature 3 (cosine 0.998). The code has two nonzero entries, C = 0.90 and B = 0.68; the other three units are silent.
  // Adding C's and B's directions, scaled by those values, gives approximately the activation h."
  const rec = s.recovery(0.5), D = Array.from(s.D, (v) => Math.round(v * 10000) / 10000);
  const src = fs.readFileSync(new URL('../docs/js/page/6-5.js', import.meta.url), 'utf8');
  const pageD = JSON.parse(src.match(/const D = (\[[^\]]*\]);/)[1]);
  check('figure: the dictionary stored in docs/js/page/6-5.js is the one this script computes', JSON.stringify(pageD) === JSON.stringify(D), JSON.stringify(pageD));
  check('caption: unit C matches feature 1 (cosine 0.997) and unit B matches feature 3 (cosine 0.998)', rec[0].unit === 2 && rec[0].cos.toFixed(3) === '0.997' && rec[2].unit === 1 && rec[2].cos.toFixed(3) === '0.998', `${rec[0].unit} ${rec[0].cos} ${rec[2].unit} ${rec[2].cos}`);
  check('caption: the code has two nonzero entries, C = 0.90 and B = 0.68; the other three units are silent', f[2].toFixed(2) === '0.90' && f[1].toFixed(2) === '0.68' && [0, 3, 4].every((k) => f[k] === 0), Array.from(f, (v) => v.toFixed(3)).join(' '));
  const hh = [s.D[2] * f[2] + s.D[1] * f[1], s.D[5 + 2] * f[2] + s.D[5 + 1] * f[1]], gap = Math.hypot(hh[0] - h[0], hh[1] - h[1]) / Math.hypot(h[0], h[1]);
  check("caption: C's and B's directions scaled by those values give approximately h (within 10% of its length)", gap < 0.1, gap);
  console.log('figure: unit matched to each feature:', JSON.stringify(s.recovery(0.5).map((r) => [r.feature + 1, r.unit + 1, Number(r.cos.toFixed(3))])));
}

// Task 3: do duplicate units fire on the same inputs? For each pair of live units pointing at the same feature,
// the share of inputs (among those where either unit fires) on which only one of the two fires.
{
  const shares = [];
  for (let seed = 1; seed <= SEEDS; seed++) {
    const s = new SAE(toy, { size: 10, lambda: 0.5, seed, lr: 0.005 });
    for (let i = 0; i < STEPS; i++) s.step();
    const ev = s.evaluate();
    const owner = new Map();
    for (let k = 0; k < 10; k++) {
      if (!ev.alive[k]) continue;
      const d = s.dir(k);
      for (let i = 0; i < 5; i++) { const w = toy.dir(i), L = Math.hypot(...w); if ((w[0] * d[0] + w[1] * d[1]) / L > 0.95) owner.set(i, [...(owner.get(i) || []), k]); }
    }
    const H = s._data(rngFor(seed), 4000);
    for (const units of owner.values()) {
      for (let a = 0; a < units.length; a++) for (let b = a + 1; b < units.length; b++) {
        let either = 0, one = 0;
        for (let t = 0; t < 4000; t++) {
          const f = s.encode(H.subarray(2 * t, 2 * t + 2));
          const fa = f[units[a]] > 0.01, fb = f[units[b]] > 0.01;
          if (fa || fb) { either++; if (fa !== fb) one++; }
        }
        if (either) shares.push(one / either);
      }
    }
  }
  // Task 3 notice: "Such a copy fires on mostly the same inputs as its twin (on at least 79% of the inputs where either fires)".
  check('t3: a copy and its twin both fire on at least 79% of the inputs where either fires', shares.length > 0 && (1 - Math.max(...shares)).toFixed(2) === '0.79', 1 - Math.max(...shares));
  console.log(`task 3: duplicate pairs ${shares.length}; share of firing inputs where only one of the pair fires (activation > 0.01): ${range(shares)}`);
}
function rngFor(seed) { let a = (seed * 2654435761) >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
// ---- Claims about the sweeps (ten starting points each) ----
const col = (size, l, k) => REP[`${size}|${l}`].map((r) => r[k]), lo = (a) => Math.min(...a), hi = (a) => Math.max(...a), avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
check('on average 0.5 features are active per input', Math.abs(5 * (1 - 0.9) - 0.5) < 1e-12, 5 * (1 - 0.9));
// Task 1 notice.
check('t1: penalty 0.5 recovers all five features in every one of ten starts', col(5, 0.5, 'recovered').every((v) => v === 5), col(5, 0.5, 'recovered').join(' '));
check('t1: at penalty 0.5 about 0.4 units are active per input', col(5, 0.5, 'l0').every((v) => v.toFixed(1) === '0.4'), `${lo(col(5, 0.5, 'l0'))} to ${hi(col(5, 0.5, 'l0'))}`);
{
  const small = [0, 0.01, 0.03, 0.1], fvu = small.flatMap((l) => col(5, l, 'fvu')), rec = small.flatMap((l) => col(5, l, 'recovered')), l0 = small.flatMap((l) => col(5, l, 'l0'));
  check('t1: penalties from 0 to 0.1 rebuild almost perfectly (rebuild error at most 0.012)', hi(fvu).toFixed(3) === '0.012', hi(fvu));
  check('t1: penalties from 0 to 0.1 recover only one to four features', lo(rec) === 1 && hi(rec) === 4, `${lo(rec)} to ${hi(rec)}`);
  check('t1: with 0.7 to 3 units active per input', lo(l0).toFixed(1) === '0.7' && hi(l0).toFixed(1) === '3.0', `${lo(l0)} to ${hi(l0)}`);
}
check('t1: penalty 2 leaves the units silent on almost every input (at most 0.05 units active per input)', hi(col(5, 2, 'l0')) <= 0.05, hi(col(5, 2, 'l0')));
check('t1: penalty 2 has a rebuild error near 0.97 (0.96 to 0.98) and recovers at most two features', lo(col(5, 2, 'fvu')) >= 0.96 && hi(col(5, 2, 'fvu')) <= 0.98 && hi(col(5, 2, 'recovered')) === 2, `${lo(col(5, 2, 'fvu'))} to ${hi(col(5, 2, 'fvu'))}; ${hi(col(5, 2, 'recovered'))}`);
// The ten starts range from 0.167 to 0.179 (starting point 1, the tool's first run, is 0.178); the page quotes one number.
check('t1, check 2: at penalty 0.5 the rebuild error is 0.17 (average over ten starts; each start within 0.01)', avg(col(5, 0.5, 'fvu')).toFixed(2) === '0.17' && lo(col(5, 0.5, 'fvu')) >= 0.16 && hi(col(5, 0.5, 'fvu')) <= 0.18, `${avg(col(5, 0.5, 'fvu'))}; ${lo(col(5, 0.5, 'fvu'))} to ${hi(col(5, 0.5, 'fvu'))}; start 1 ${col(5, 0.5, 'fvu')[0]}`);
// Task 2 notice: "In our tests (ten starts at each of the three penalties) the SAE recovered one to three."
{
  const rec = [0.3, 0.5, 1].flatMap((l) => col(3, l, 'recovered'));
  check('t2: a dictionary of 3 units recovers one to three features (30 runs)', rec.length === 30 && lo(rec) === 1 && hi(rec) === 3, `${lo(rec)} to ${hi(rec)}`);
}
// Task 3 notice: "In ten starts at 10 units, all five features were recovered every time; one to four units never fired, and one to
// four more pointed at a feature that another unit already matched, and in one start of the ten one unit pointed at no feature."
check('t3: 10 units, penalty 0.5: all five features recovered in every one of ten starts', col(10, 0.5, 'recovered').every((v) => v === 5), col(10, 0.5, 'recovered').join(' '));
check('t3: one to four units never fire', lo(col(10, 0.5, 'dead')) === 1 && hi(col(10, 0.5, 'dead')) === 4, col(10, 0.5, 'dead').join(' '));
check('t3: one to four more point at a feature another unit already matches', lo(col(10, 0.5, 'dup')) === 1 && hi(col(10, 0.5, 'dup')) === 4, col(10, 0.5, 'dup').join(' '));
check('t3: in one start of the ten one unit points at no feature', hi(col(10, 0.5, 'misaligned')) === 1 && col(10, 0.5, 'misaligned').filter((v) => v === 1).length === 1 && lo(col(10, 0.5, 'misaligned')) === 0, col(10, 0.5, 'misaligned').join(' '));
check('t3: at 20 units all five features are still recovered in every start, with more silent units and copies', col(20, 0.5, 'recovered').every((v) => v === 5) && lo(col(20, 0.5, 'dead')) > hi(col(10, 0.5, 'dead')), `dead ${col(20, 0.5, 'dead').join(' ')}`);
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
