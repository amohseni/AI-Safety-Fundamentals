// Numbers quoted on page 5.5, panel 1 (spirals, width 3), and the choice of the five seeds.
// Run: node tools/test-page-5-5.mjs   (about three minutes)
import { SEEDS, STEPS, DATA, FRESH, makeNet, agreement, responses, bestMatches, median } from '../docs/js/page/5-5-measure.js';

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

// 2. The five runs.
const nets = SEEDS.map((s) => { const n = makeNet(s); for (let i = 0; i < STEPS; i++) n.step(DATA.X, DATA.Y); return n; });
console.log('training accuracy', nets.map((n) => n.accuracy(DATA.X, DATA.Y)).join(' '));
console.log('accuracy on 2,000 new spiral points', nets.map((n) => (n.accuracy(FRESH.X, FRESH.Y) * 100).toFixed(1)).join(' '));

const pairs = [];
for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) pairs.push([a, b]);
const ag = pairs.map(([a, b]) => agreement(nets[a], nets[b]));
console.log('pairwise agreement on new spiral points: min', (Math.min(...ag) * 100).toFixed(1), 'mean', (ag.reduce((s, v) => s + v) / ag.length * 100).toFixed(1), 'max', (Math.max(...ag) * 100).toFixed(1));

const square = [];
for (let i = 0; i < 80; i++) for (let j = 0; j < 80; j++) square.push([-1 + (2 * i + 1) / 80, -1 + (2 * j + 1) / 80]);
const agS = pairs.map(([a, b]) => agreement(nets[a], nets[b], square));
console.log('pairwise agreement over the whole square [-1,1]^2: min', (Math.min(...agS) * 100).toFixed(1), 'mean', (agS.reduce((s, v) => s + v) / agS.length * 100).toFixed(1));

for (const layer of [1, 2]) {
  const R = nets.map((n) => responses(n, layer));
  const m = [];
  for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) if (a !== b) m.push(...bestMatches(R[a], R[b]));
  console.log(`layer ${layer}: best-match |correlation| over ${m.length} unit comparisons: min ${Math.min(...m).toFixed(2)} median ${median(m).toFixed(2)} max ${Math.max(...m).toFixed(2)}; below 0.9: ${m.filter((v) => v < 0.9).length}`);
}
