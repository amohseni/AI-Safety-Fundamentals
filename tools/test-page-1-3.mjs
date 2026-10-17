// Numbers quoted on page 1.3, recomputed with the tool's own network (16 units per hidden layer,
// circles dataset, gradients at the random starting parameters, starts 1 to 20).
// Run: node tools/test-page-1-3.mjs
import { DeepNet } from '../docs/js/lib/deepnet.js';
import { datasets } from '../docs/js/lib/mlp.js';

const { X, Y } = datasets.circles();
const SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);
const f = (v) => v.toExponential(2);
function ratios(act, depth, gain) {
  return SEEDS.map((seed) => { const r = new DeepNet({ depth, act, gain, seed }).gradients(X, Y).rms; return r[0] / r[depth - 1]; });
}
function report(label, rs, claim) {
  const lo = Math.min(...rs), hi = Math.max(...rs);
  const ok = claim(lo, hi);
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: layer 1 / last hidden layer, min ${f(lo)}, max ${f(hi)}`);
  if (!ok) process.exitCode = 1;
}

// Task 1: sigmoid, 10 layers, scale 1: layer 1 between 400,000 and 17 million times smaller.
report('sigmoid 10 layers g=1', ratios('sigmoid', 10, 1), (lo, hi) => 1 / hi >= 4e5 && 1 / lo <= 1.75e7);
// Task 1: with 20 layers, typically a few trillion times smaller (geometric mean between 10^12 and 10^13).
{
  const rs = ratios('sigmoid', 20, 1);
  const gm = Math.exp(rs.reduce((s, r) => s + Math.log(r), 0) / rs.length);
  const ok = 1 / gm > 1e12 && 1 / gm < 1e13;
  console.log(`${ok ? 'ok  ' : 'FAIL'} sigmoid 20 layers g=1: geometric mean ratio ${f(gm)}`);
  if (!ok) process.exitCode = 1;
}
// Task 2: ReLU, 10 layers, scale 1: layer 1 between 1/14 of and 1.7 times layer 10.
report('relu 10 layers g=1', ratios('relu', 10, 1), (lo, hi) => lo >= 1 / 14 && hi <= 1.7);
// Task 2: tanh at scale 1: between 0.3 and 6 times.
report('tanh 10 layers g=1', ratios('tanh', 10, 1), (lo, hi) => lo >= 0.28 && hi <= 6);
// Task 3: tanh, 20 layers, scale 4: layer 1 between 130 and 3,600 times larger than layer 20.
report('tanh 20 layers g=4', ratios('tanh', 20, 4), (lo, hi) => lo >= 130 && hi <= 3600);
// Task 3: tanh, 20 layers, scale 1: roughly level (for comparison).
report('tanh 20 layers g=1', ratios('tanh', 20, 1), () => true);
// Task 3: ReLU, 20 layers, scale 4: every layer's gradient above 100,000.
{
  const mins = SEEDS.map((seed) => Math.min(...new DeepNet({ depth: 20, act: 'relu', gain: 4, seed }).gradients(X, Y).rms));
  const ok = Math.min(...mins) > 1e5;
  console.log(`${ok ? 'ok  ' : 'FAIL'} relu 20 layers g=4: smallest layer gradient over all starts ${f(Math.min(...mins))}`);
  if (!ok) process.exitCode = 1;
}
// Task 3: ReLU at scale 2 or more grows with depth (mean gradient at 20 layers > at 5 layers).
for (const gain of [1, 1.2, 1.41, 1.6, 2]) {
  const mean = (depth) => SEEDS.reduce((s, seed) => s + Math.log10(new DeepNet({ depth, act: 'relu', gain, seed }).gradients(X, Y).rms[0]), 0) / SEEDS.length;
  console.log(`     relu g=${gain}: mean log10 layer-1 gradient, 5 layers ${mean(5).toFixed(2)}, 20 layers ${mean(20).toFixed(2)}`);
}
// Check 2: sigmoid, 10 layers, scale 4: layer 1 still more than 100 times smaller.
report('sigmoid 10 layers g=4', ratios('sigmoid', 10, 4), (lo, hi) => 1 / hi >= 100);
// Gradient check: backpropagation agrees with finite differences.
{
  const net = new DeepNet({ depth: 4, act: 'tanh', gain: 1.5, seed: 3 });
  const sub = { X: X.slice(0, 20), Y: Y.slice(0, 20) };
  const g = net.gradients(sub.X, sub.Y);
  let worst = 0;
  for (const [l, k] of [[0, 3], [2, 17], [4, 5]]) {
    const h = 1e-6, w0 = net.W[l][k];
    net.W[l][k] = w0 + h; const lp = net.gradients(sub.X, sub.Y).loss;
    net.W[l][k] = w0 - h; const lm = net.gradients(sub.X, sub.Y).loss;
    net.W[l][k] = w0;
    worst = Math.max(worst, Math.abs((lp - lm) / (2 * h) - g.gW[l][k]));
  }
  const ok = worst < 1e-7;
  console.log(`${ok ? 'ok  ' : 'FAIL'} finite-difference check, worst error ${f(worst)}`);
  if (!ok) process.exitCode = 1;
}
