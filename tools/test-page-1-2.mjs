// Numbers behind pages 1.2 and 2.2 (perceptron playground). Run: node tools/test-page-1-2.mjs
import { datasets, perceptronPass, errors, hiddenOut, step, sum } from '../docs/js/lib/perceptron.js';
const START = { w: [-0.5, 1], b: 0.3 }; // the page's starting line
function run(name, lr, maxPass = 100) {
  const { X, Y } = datasets[name]();
  const s = { w: START.w.slice(), b: START.b };
  const hist = [], states = [];
  for (let p = 0; p < maxPass; p++) {
    const e = perceptronPass(s, X, Y, lr);
    hist.push(e);
    states.push(`${s.w.map((v) => v.toFixed(9))},${s.b.toFixed(9)}`);
    if (e === 0) break;
  }
  const last = states[states.length - 1];
  const repeatsAt = states.indexOf(last);
  return { hist, converged: hist[hist.length - 1] === 0, passes: hist.length, updates: hist.reduce((a, b) => a + b, 0), repeats: repeatsAt < states.length - 1 };
}
console.log('Rule from the starting line; hist = mistakes per pass');
for (const lr of [0.1, 0.25, 0.5, 1]) for (const d of ['and', 'or', 'xor', 'separable', 'nonseparable']) {
  const r = run(d, lr);
  console.log(`lr ${lr} ${d.padEnd(12)} converged=${r.converged} passes=${r.passes} updates=${r.updates} min-mistakes-after-pass5=${Math.min(...r.hist.slice(5))} weights-repeat=${r.repeats} first: ${r.hist.slice(0, 10).join(' ')}`);
}
// Minimum number of errors any line makes on XOR (search over lines).
let best = 9;
for (let a = 0; a < 360; a += 1) for (let c = -2; c <= 2; c += 0.01) { const w = [Math.cos(a * Math.PI / 180), Math.sin(a * Math.PI / 180)]; const { X, Y } = datasets.xor(); best = Math.min(best, errors(w, c, X, Y)); }
console.log('fewest errors any line makes on XOR:', best);
// Hidden-layer XOR with slider values on a 0.5 grid: h1 = OR, h2 = AND, out = h1 AND NOT h2.
const H = { W: [[1, 1], [1, 1]], b: [-0.5, -1.5], v: [1, -1], c: -0.5 };
const { X, Y } = datasets.xor();
console.log('hidden layer XOR outputs', X.map((x) => hiddenOut(H, x)).join(','), 'targets', Y.join(','));
// x1 AND NOT x2 is computed by one perceptron:
const w = [1, -1], b = -0.5;
console.log('x1 AND NOT x2:', X.map((x) => step(sum(w, b, x))).join(','), '(inputs', X.map((x) => x.join('')).join(' '), ')');
// Neuron mode: lengthening w with b fixed moves the line toward the origin: distance = |b|/|w|.
for (const L of [1, 2, 4]) console.log(`|w|=${L}, b=1: distance of line from origin ${(1 / L).toFixed(3)}`);
// XOR: does the rule cycle? Find the first pass after which the weights repeat an earlier state.
for (const lr of [0.1, 0.25, 0.5, 1]) {
  const { X, Y } = datasets.xor();
  const s = { w: START.w.slice(), b: START.b }, seen = new Map();
  for (let p = 1; p <= 1000; p++) {
    perceptronPass(s, X, Y, lr);
    const k = `${s.w.map((v) => v.toFixed(9))},${s.b.toFixed(9)}`;
    if (seen.has(k)) { console.log(`XOR lr ${lr}: weights after pass ${p} equal those after pass ${seen.get(k)} (cycle of ${p - seen.get(k)} passes); w=${s.w.map((v) => v.toFixed(2))} b=${s.b.toFixed(2)}`); break; }
    seen.set(k, p);
  }
}
