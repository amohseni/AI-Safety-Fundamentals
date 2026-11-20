// Numbers behind page 2.3 (collapse). Run: node tools/test-page-2-3.mjs   (takes about a minute and a half)
import { Net, countRegions } from '../docs/js/lib/s2-net.js';
import { datasets } from '../docs/js/lib/mlp.js';
const { X, Y } = datasets.spirals();
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const STEPS = 2000, LR = 0.01, SEEDS = [1, 2, 3, 4, 5];

// 1. Backpropagation agrees with finite differences.
{
  const net = new Net([2, 3, 3, 3, 1], ['relu', 'gelu', 'sigmoid'], { seed: 2 });
  const Xs = X.slice(0, 20), Ys = Y.slice(0, 20), g = net.grad(Xs, Ys), h = 1e-6;
  let worst = 0;
  for (let l = 0; l < net.L; l++) for (let i = 0; i < net.W[l].length; i++) for (let j = 0; j < net.W[l][i].length; j++) {
    net.W[l][i][j] += h; const lp = net.loss(Xs, Ys); net.W[l][i][j] -= 2 * h; const lm = net.loss(Xs, Ys); net.W[l][i][j] += h;
    worst = Math.max(worst, Math.abs((lp - lm) / (2 * h) - g.W[l][i][j]));
  }
  console.log('largest gap between backprop and finite-difference gradient:', worst.toExponential(2));
  check('backpropagation agrees with finite differences to 1e-6', worst < 1e-6, worst.toExponential(2));
}
// 2. With no activation functions the network equals one linear map followed by the sigmoid.
{
  const net = new Net([2, 8, 8, 8, 1], ['none', 'none', 'none'], { seed: 3 });
  for (let s = 0; s < 300; s++) net.step(X, Y);
  const { v, c } = net.collapse();
  const gap = Math.max(...X.map((x) => Math.abs(net.logit(x) - (v[0] * x[0] + v[1] * x[1] + c))));
  console.log('collapse: largest gap between network logit and v.x + c:', gap.toExponential(2));
  // Theorem: any stack of linear layers collapses into a single linear layer.
  check('theorem: with no activation functions the trained network equals one linear map plus a constant (gap below 1e-9)', gap < 1e-9, gap.toExponential(2));
}
function train(width, acts, seed) {
  const net = new Net([2, width, width, width, 1], acts, { seed, lr: LR });
  const r0 = countRegions(net);
  for (let s = 0; s < STEPS; s++) net.step(X, Y);
  return { acc: net.accuracy(X, Y), r0, r1: countRegions(net) };
}
const fmt = (rs) => `acc ${rs.map((r) => Math.round(r.acc * 100)).join(' ')} | pieces at start ${rs.map((r) => r.r0).join(' ')} | after ${rs.map((r) => r.r1).join(' ')}`;
const off = SEEDS.map((s) => train(8, ['none', 'none', 'none'], s)), one = SEEDS.map((s) => train(8, ['relu', 'none', 'none'], s)), relu = {};
console.log('width 8, all off      ', fmt(off));
console.log('width 8, layer 1 ReLU ', fmt(one));
for (const w of [2, 4, 8, 16]) { relu[w] = SEEDS.map((s) => train(w, ['relu', 'relu', 'relu'], s)); console.log(`width ${String(w).padEnd(2)}, all ReLU    `, fmt(relu[w])); }
for (const a of ['gelu', 'tanh', 'sigmoid']) console.log(`width 8, all ${a.padEnd(8)}`, `acc ${SEEDS.map((s) => Math.round(train(8, [a, a, a], s).acc * 100)).join(' ')}`);
const pc = (r) => Math.round(r.acc * 100), lo = (a) => Math.min(...a), hi = (a) => Math.max(...a);
// Task 1 notice: "A straight line, and the accuracy ends at 60% in each of the five starts the tool offers."
check('t1: all activation functions off: the accuracy ends at 60% in each of the five starts', off.every((r) => pc(r) === 60), off.map(pc).join(' '));
check('t1: all off: the boundary is a straight line (one linear piece before and after training)', off.every((r) => r.r0 === 1 && r.r1 === 1), off.map((r) => r.r1).join(' '));
// Task 2 notice: "In the five starts the tool offers, the accuracy ended between about 85% and 100% (100% in three)."
{
  const a = one.map(pc);
  check('t2: ReLU in layer 1 only: accuracy between about 85% (83 to 87) and 100%, with 100% in three starts', lo(a) >= 83 && lo(a) <= 87 && hi(a) === 100 && a.filter((v) => v === 100).length === 3, a.join(' '));
}
// Task 3 notice: "before training: about 40 to 80 pieces at width 4, 185 to 255 at width 8, and 690 to 950 at width 16.
// After training: about 45 to 80, 245 to 360, and 1,000 to 1,400." Each end of each range is allowed to be off by 5.
const near = (a, l, h) => Math.abs(lo(a) - l) <= 5 && Math.abs(hi(a) - h) <= 5, rng = (a) => `${lo(a)} to ${hi(a)}`;
const before = (w) => relu[w].map((r) => r.r0), after = (w) => relu[w].map((r) => r.r1);
check('t3: before training, about 40 to 80 pieces at width 4', near(before(4), 40, 80), rng(before(4)));
check('t3: before training, 185 to 255 at width 8', near(before(8), 185, 255), rng(before(8)));
check('t3: before training, 690 to 950 at width 16', near(before(16), 690, 950), rng(before(16)));
check('t3: after training, about 45 to 80 at width 4', near(after(4), 45, 80), rng(after(4)));
check('t3: after training, 245 to 360 at width 8', near(after(8), 245, 360), rng(after(8)));
check('t3: after training, 1,000 to 1,400 at width 16', near(after(16), 1000, 1400), rng(after(16)));
// "The count grows several times over each time the width doubles, much faster than the width itself."
{
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const ratios = [[4, 8], [8, 16]].flatMap(([a, b]) => [mean(before(b)) / mean(before(a)), mean(after(b)) / mean(after(a))]);
  const each = [[4, 8], [8, 16]].every(([a, b]) => lo(before(b)) > 2 * hi(before(a)) && lo(after(b)) > 2 * hi(after(a)));
  check('t3: the count grows several times over each time the width doubles (mean count at least 3 times larger; every start more than 2 times every start at half the width)', ratios.every((r) => r >= 3) && each, ratios.map((r) => r.toFixed(1)).join(' '));
}
// Caption: "Softmax ... turns the scores 2, 1, 0 into the probabilities 0.67, 0.24, 0.09."
{
  const e = [2, 1, 0].map(Math.exp), z = e[0] + e[1] + e[2], p = e.map((v) => (v / z).toFixed(2));
  check('caption: softmax of 2, 1, 0 is 0.67, 0.24, 0.09', p.join(' ') === '0.67 0.24 0.09', p.join(' '));
}
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
