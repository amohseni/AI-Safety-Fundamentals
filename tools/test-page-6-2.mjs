// Numerical tests behind the claims on page 6.2. Run: node tools/test-page-6-2.mjs
import fs from 'node:fs';
import { decodeTruth } from '../docs/js/lib/s6l-truth.js';
import { decodeF16, rows, massMean, logistic, accuracy, dot, norm } from '../docs/js/lib/s6l-probe.js';
import { planeBasis, weekdayPlane, DAYS, bestSplit } from '../docs/js/lib/s6l-geometry.js';

const T = decodeTruth(JSON.parse(fs.readFileSync(new URL('../docs/data/truth/statements.json', import.meta.url))));
const W = JSON.parse(fs.readFileSync(new URL('../docs/data/6-2/weekdays.json', import.meta.url)));
let fails = 0;
const check = (name, ok, detail) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: ${detail}`); if (!ok) fails++; };
const pct = (v) => `${(v * 100).toFixed(1)}%`;

const plain = T.S.filter((s) => s.form === 'plain');
const by = (topic, f = () => true) => plain.filter((s) => s.topic === topic && f(s));
const train = by('cities', T.isTrain);
const mm = massMean(train.map((s) => T.X[s.i]), train.map((s) => s.label));
const mmUnit = Float32Array.from(mm.w, (v) => v / norm(mm.w));
const splitAlong = (set, u) => bestSplit(set.filter((s) => s.label).map((s) => dot(T.X[s.i], u)), set.filter((s) => !s.label).map((s) => dot(T.X[s.i], u)));
const fixedThr = (set, probe) => accuracy(probe, set.map((s) => T.X[s.i]), set.map((s) => s.label));

console.log('variance kept by 32 directions:', T.json.variance);
// Best direction within each plane (sweep the angle in 0.5 degree steps).
function bestInPlane(basis, set) {
  let best = { acc: 0 };
  for (let a = 0; a < 360; a += 0.5) {
    const r = (a * Math.PI) / 180;
    const u = basis.e1.map((v, j) => Math.cos(r) * v + Math.sin(r) * basis.e2[j]);
    const s = splitAlong(set, u);
    if (s.acc > best.acc) best = { ...s, angle: a };
  }
  return best;
}
const allPlain = plain.map((s) => s.i);
const cities = by('cities');
const pcaB = planeBasis(T.X, allPlain, 'pca', {});
const bp = bestInPlane(pcaB, cities);
check('task 1: best arrow in the principal-directions plane sorts about 80% of cities', Math.abs(bp.acc - 0.795) < 0.006, `${pct(bp.acc)} at ${bp.angle} degrees`);
for (const t of ['spanish', 'elements']) console.log('  best in PCA plane for', t, pct(bestInPlane(pcaB, by(t)).acc));
const randBest = [1, 2, 3, 4, 5].map((seed) => bestInPlane(planeBasis(T.X, allPlain, 'random', { seed }), cities).acc);
check('task 1: first five random planes give between 78% and 95%', Math.min(...randBest) >= 0.775 && Math.max(...randBest) < 0.955 && Math.min(...randBest) < bp.acc && Math.max(...randBest) > bp.acc, randBest.map(pct).join(', '));
const mmB = planeBasis(T.X, allPlain, 'mm', { mm: mmUnit });
check('task 2: difference of means sorts 97.5% of cities (best threshold)', Math.abs(splitAlong(cities, mmUnit).acc - 0.975) < 0.003, pct(splitAlong(cities, mmUnit).acc));
check('task 2: 95.5% of held-out cities at the midpoint threshold', Math.abs(fixedThr(by('cities', (s) => !T.isTrain(s)), mm) - 0.955) < 0.003, pct(fixedThr(by('cities', (s) => !T.isTrain(s)), mm)));
console.log('difference of means, fixed midpoint threshold: held-out cities', pct(fixedThr(by('cities', (s) => !T.isTrain(s)), mm)),
  'spanish', pct(fixedThr(by('spanish'), mm)), 'elements', pct(fixedThr(by('elements'), mm)));
console.log('difference of means, best threshold: spanish', pct(splitAlong(by('spanish'), mmUnit).acc), 'elements', pct(splitAlong(by('elements'), mmUnit).acc));
check('task 3: city direction sorts 99% of Spanish words and of elements (best threshold)', splitAlong(by('spanish'), mmUnit).acc >= 0.985 && splitAlong(by('elements'), mmUnit).acc >= 0.985, '');
const lr = logistic(train.map((s) => T.X[s.i]), train.map((s) => s.label), { l2: 0.01 });
console.log('logistic probe on training cities: held-out', pct(fixedThr(by('cities', (s) => !T.isTrain(s)), lr)), 'spanish', pct(fixedThr(by('spanish'), lr)), 'elements', pct(fixedThr(by('elements'), lr)));
const lrU = Float32Array.from(lr.w, (v) => v / norm(lr.w));
console.log('angle between logistic and mean-difference directions', (Math.acos(dot(lrU, mmUnit)) * 180 / Math.PI).toFixed(1));
// Directions found on other topics
for (const t of ['spanish', 'elements']) {
  const tr = by(t, T.isTrain);
  const m2 = massMean(tr.map((s) => T.X[s.i]), tr.map((s) => s.label));
  const u2 = Float32Array.from(m2.w, (v) => v / norm(m2.w));
  const best = splitAlong(cities, u2).acc, mid = fixedThr(cities, m2);
  console.log(`mean difference from ${t}: angle to cities direction`, (Math.acos(dot(u2, mmUnit)) * 180 / Math.PI).toFixed(1), 'accuracy on cities: midpoint threshold', pct(mid), 'best threshold', pct(best));
  // Task 3: the direction transfers back to cities with a threshold chosen for cities; only the threshold fails.
  check(`task 3: direction from ${t} sorts at least 95% of cities with the best threshold, about 62% at its own midpoint`, best >= 0.95 && mid < 0.64, `${pct(best)} vs ${pct(mid)}`);
}

// ---- Weekdays ----
const Xw = rows(decodeF16(W.X), W.k);
const { e1, e2, means, center } = weekdayPlane(Xw, W.day);
const ang = means.map((m) => (Math.atan2(dot(m, e2), dot(m, e1)) * 180) / Math.PI);
console.log('weekday angles in the plane:', DAYS.map((d, i) => `${d.slice(0, 3)} ${ang[i].toFixed(0)}`).join(', '));
// Circular order: sort by angle and check that consecutive days are neighbors in the week.
const order = ang.map((a, d) => [a, d]).sort((p, q) => p[0] - q[0]).map(([, d]) => d);
const neighbors = order.every((d, i) => { const n = order[(i + 1) % 7]; return (n - d + 7) % 7 === 1 || (d - n + 7) % 7 === 1; });
check('weekday means lie in weekly order around the origin', neighbors, `order by angle: ${order.map((d) => DAYS[d].slice(0, 3)).join(' ')}`);
const radii = means.map((m) => Math.hypot(dot(m, e1), dot(m, e2)));
console.log('radii', radii.map((r) => r.toFixed(2)).join(' '));
// Share of each context's points that are nearer their own day's mean than any other day's mean, in the plane.
let near = 0;
Xw.forEach((x, i) => {
  const c = center(x), p = [dot(c, e1), dot(c, e2)];
  const dists = means.map((m) => Math.hypot(p[0] - dot(m, e1), p[1] - dot(m, e2)));
  if (dists.indexOf(Math.min(...dists)) === W.day[i]) near++;
});
console.log('points nearest their own day mean in the plane:', pct(near / Xw.length));
// For every direction in the plane, two days that are not neighbors land close together.
let worst = Infinity;
for (let a = 0; a < 180; a += 1) {
  const r = (a * Math.PI) / 180;
  const proj = means.map((m) => Math.cos(r) * dot(m, e1) + Math.sin(r) * dot(m, e2));
  let minGap = Infinity;
  for (let i = 0; i < 7; i++) for (let j = i + 1; j < 7; j++) {
    const g = Math.min((j - i) % 7, (i - j + 7) % 7);
    if (g >= 2) minGap = Math.min(minGap, Math.abs(proj[i] - proj[j]));
  }
  worst = Math.min(worst, minGap);
  if (a % 30 === 0) console.log(`  angle ${a}: closest non-neighbor pair gap ${minGap.toFixed(3)} (radius ~${(radii.reduce((s, v) => s + v, 0) / 7).toFixed(2)})`);
}
// Task 4: no direction in the plane puts the days in weekly order, and along every direction some two
// days that are at least two apart in the week are adjacent in the order along the arrow.
let ordered = 0, allFold = true;
for (let a = 0; a < 360; a += 0.25) {
  const r = (a * Math.PI) / 180;
  const o = means.map((m, d) => [Math.cos(r) * dot(m, e1) + Math.sin(r) * dot(m, e2), d]).sort((p, q) => p[0] - q[0]).map(([, d]) => d);
  const weekly = [1, -1].some((dir) => o.every((d, i) => i === 0 || (d - o[i - 1] - dir + 14) % 7 === 0));
  if (weekly) ordered++;
  const fold = o.some((d, i) => i > 0 && Math.min((d - o[i - 1] + 7) % 7, (o[i - 1] - d + 7) % 7) >= 2);
  if (!fold) allFold = false;
}
check('task 4: no direction in the plane gives the weekly order', ordered === 0, `${ordered} of 1440 directions`);
check('task 4: along every direction two days far apart in the week are next to each other', allFold, '');
process.exit(fails ? 1 : 0);
