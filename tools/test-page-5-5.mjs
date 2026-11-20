// Numbers quoted on page 5.5, panel 1 (spirals, width 3), and the choice of the five seeds.
// Run: node tools/test-page-5-5.mjs   (about a minute and a half)
import { SEEDS, STEPS, DATA, FRESH, makeNet, agreement, responses, bestMatches, median } from '../docs/js/page/5-5-measure.js';

let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };

// 1. Which of 40 starting points reach 100% on the training points within STEPS steps,
// unperturbed and with all initial parameters perturbed by +1e-9 and -1e-7 (times a fixed pattern).
const reach = [];
for (let seed = 1; seed <= 40; seed++) {
  const ok = [0, 1e-9, -1e-7].every((eps) => {
    const n = makeNet(seed);
    let k = 1;
    n.W.forEach((M) => M.forEach((r) => r.forEach((_, j) => { r[j] += eps * Math.sin(1000 * k++); })));
    for (let i = 0; i < STEPS; i++) n.step(DATA.X, DATA.Y);
    return n.accuracy(DATA.X, DATA.Y) === 1;
  });
  if (ok) reach.push(seed);
}
console.log(`seeds reaching 100% in ${STEPS} steps in all three versions (of 1..40):`, reach.join(' '), '| page uses', SEEDS.join(' '));

// Panel note: "trained on the spirals from five seeds for 4,000 steps each. Of 40 seeds we tried, these five reached 100% on
// the training points within 4,000 steps in every test we ran."
check('panel 1: five seeds, 4,000 steps each', SEEDS.length === 5 && STEPS === 4000, `${SEEDS.length} ${STEPS}`);
check('panel 1: each of the five seeds reaches 100% on the training points within 4,000 steps in all three versions', SEEDS.every((s) => reach.includes(s)), reach.join(' '));

// 2. The five runs.
const nets = SEEDS.map((s) => { const n = makeNet(s); for (let i = 0; i < STEPS; i++) n.step(DATA.X, DATA.Y); return n; });
console.log('training accuracy', nets.map((n) => n.accuracy(DATA.X, DATA.Y)).join(' '));
console.log('accuracy on 2,000 new spiral points', nets.map((n) => (n.accuracy(FRESH.X, FRESH.Y) * 100).toFixed(1)).join(' '));

// Task 2 prompt: "All five runs classify every training point correctly."
check('t2: all five runs classify every training point correctly', nets.every((n) => n.accuracy(DATA.X, DATA.Y) === 1), nets.map((n) => n.accuracy(DATA.X, DATA.Y)).join(' '));
check('t1: 2,000 new points', FRESH.X.length === 2000, FRESH.X.length);
const pairs = [];
for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) pairs.push([a, b]);
const ag = pairs.map(([a, b]) => agreement(nets[a], nets[b]));
console.log('pairwise agreement on new spiral points: min', (Math.min(...ag) * 100).toFixed(1), 'mean', (ag.reduce((s, v) => s + v) / ag.length * 100).toFixed(1), 'max', (Math.max(...ag) * 100).toFixed(1));

// Task 1 notice: "any two of the five runs give the same class to between 92% and 99.8% of the points (97% on average over
// the ten pairs)". 92% is the lowest pair rounded to a whole percent, checked to within half a point.
const mean = (a) => a.reduce((s, v) => s + v) / a.length;
check('t1: ten pairs; the lowest agreement on new spiral points is 92% (to within half a point)', ag.length === 10 && Math.abs(Math.min(...ag) * 100 - 92) <= 0.5, Math.min(...ag));
check('t1: the highest is 99.8%', (Math.max(...ag) * 100).toFixed(1) === '99.8', Math.max(...ag));
check('t1: 97% on average over the ten pairs', Math.round(mean(ag) * 100) === 97, mean(ag));
const square = [];
for (let i = 0; i < 80; i++) for (let j = 0; j < 80; j++) square.push([-1 + (2 * i + 1) / 80, -1 + (2 * j + 1) / 80]);
const agS = pairs.map(([a, b]) => agreement(nets[a], nets[b], square));
console.log('pairwise agreement over the whole square [-1,1]^2: min', (Math.min(...agS) * 100).toFixed(1), 'mean', (agS.reduce((s, v) => s + v) / agS.length * 100).toFixed(1));

// "Over the whole square ... two runs agree on as little as 77% of it (90% on average)."
check('t1: over the whole square two runs agree on as little as 77% of it', Math.round(Math.min(...agS) * 100) === 77, Math.min(...agS));
// FAILS as of 2026-11-19: the average is 89.475%, which is 89% to the nearest percent (the line above prints it as 89.5).
// See AUDIT-TESTS-FINDINGS.md.
check('t1: over the whole square the average agreement is 89%', Math.round(mean(agS) * 100) === 89, mean(agS));
const L = {};
for (const layer of [1, 2]) {
  const R = nets.map((n) => responses(n, layer));
  const m = [];
  for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) if (a !== b) m.push(...bestMatches(R[a], R[b]));
  L[layer] = m;
  console.log(`layer ${layer}: best-match |correlation| over ${m.length} unit comparisons: min ${Math.min(...m).toFixed(2)} median ${median(m).toFixed(2)} max ${Math.max(...m).toFixed(2)}; below 0.9: ${m.filter((v) => v < 0.9).length}`);
}
// Task 2 notice: "Over all 60 such comparisons the median correlation is 0.78, and 42 of the 60 units have no match at 0.9 or
// above. The units of layer 1 are more alike: median 0.94, with 36 of the 60 matched at 0.9 or above."
check('t2: layer 2: 60 comparisons, median correlation 0.78', L[2].length === 60 && median(L[2]).toFixed(2) === '0.78', median(L[2]));
check('t2: layer 2: 42 of the 60 units have no match at 0.9 or above', L[2].filter((v) => v < 0.9).length === 42, L[2].filter((v) => v < 0.9).length);
check('t2: layer 1: median 0.94, with 36 of the 60 matched at 0.9 or above', L[1].length === 60 && median(L[1]).toFixed(2) === '0.94' && L[1].filter((v) => v >= 0.9).length === 36, `${median(L[1])} ${L[1].filter((v) => v >= 0.9).length}`);
// Caption: "30 + 40 comes out as 11 when p = 59".
check('caption: (30 + 40) mod 59 = 11', (30 + 40) % 59 === 11, (30 + 40) % 59);
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
