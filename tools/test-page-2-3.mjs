// Numbers behind page 2.3 (collapse). Run: node tools/test-page-2-3.mjs   (takes about a minute)
import { Net, countRegions } from '../docs/js/lib/s2-net.js';
import { datasets } from '../docs/js/lib/mlp.js';
const { X, Y } = datasets.spirals();
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
}
// 2. With no activation functions the network equals one linear map followed by the sigmoid.
{
  const net = new Net([2, 8, 8, 8, 1], ['none', 'none', 'none'], { seed: 3 });
  for (let s = 0; s < 300; s++) net.step(X, Y);
  const { v, c } = net.collapse();
  const gap = Math.max(...X.map((x) => Math.abs(net.logit(x) - (v[0] * x[0] + v[1] * x[1] + c))));
  console.log('collapse: largest gap between network logit and v.x + c:', gap.toExponential(2));
}
function train(width, acts, seed) {
  const net = new Net([2, width, width, width, 1], acts, { seed, lr: LR });
  const r0 = countRegions(net);
  for (let s = 0; s < STEPS; s++) net.step(X, Y);
  return { acc: net.accuracy(X, Y), r0, r1: countRegions(net) };
}
const fmt = (rs) => `acc ${rs.map((r) => Math.round(r.acc * 100)).join(' ')} | pieces at start ${rs.map((r) => r.r0).join(' ')} | after ${rs.map((r) => r.r1).join(' ')}`;
console.log('width 8, all off      ', fmt(SEEDS.map((s) => train(8, ['none', 'none', 'none'], s))));
console.log('width 8, layer 1 ReLU ', fmt(SEEDS.map((s) => train(8, ['relu', 'none', 'none'], s))));
for (const w of [2, 4, 8, 16]) console.log(`width ${String(w).padEnd(2)}, all ReLU    `, fmt(SEEDS.map((s) => train(w, ['relu', 'relu', 'relu'], s))));
for (const a of ['gelu', 'tanh', 'sigmoid']) console.log(`width 8, all ${a.padEnd(8)}`, `acc ${SEEDS.map((s) => Math.round(train(8, [a, a, a], s).acc * 100)).join(' ')}`);
