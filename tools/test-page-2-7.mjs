// Numbers behind page 2.7 (gradients by hand). Run: node tools/test-page-2-7.mjs
import { GRAPHS, forward, backward, chainPaths, nudge } from '../docs/js/lib/s2-graph.js';
import { rng } from '../docs/js/lib/mlp.js';
const defaults = (g) => Object.fromEntries(g.nodes.filter((n) => n.op === 'input').map((n) => [n.id, n.value]));
for (const [k, g] of Object.entries(GRAPHS)) {
  const inp = defaults(g), val = forward(g, inp), { grad, parts } = backward(g, val);
  console.log(`graph ${k}: values`, Object.entries(val).map(([i, v]) => `${i}=${v.toFixed(4)}`).join(' '));
  console.log('  grads ', Object.entries(grad).map(([i, v]) => `${i}=${v.toFixed(4)}`).join(' '));
  console.log('  nodes receiving more than one derivative:', Object.entries(parts).filter(([, p]) => p.length > 1).map(([i, p]) => `${i} (${p.map((q) => q.value.toFixed(4)).join(' + ')})`).join(', ') || 'none');
  for (const n of g.nodes.filter((n) => n.param)) {
    const paths = chainPaths(g, val, n.id), sum = paths.reduce((s, p) => s + p.product, 0), t = nudge(g, inp, n.id);
    console.log(`  ${n.id}: backprop ${grad[n.id].toFixed(6)}, sum over ${paths.length} path(s) ${sum.toFixed(6)}, nudge slope ${t.slope.toFixed(6)}`);
  }
}
// Over random inputs in the sliders' range [-3, 3]: does the nudge slope agree with backprop to 4 decimals?
let total = 0, agree = 0, worst = 0;
const r = rng(7);
for (const g of Object.values(GRAPHS)) for (let trial = 0; trial < 20000; trial++) {
  const inp = Object.fromEntries(g.nodes.filter((n) => n.op === 'input').map((n) => [n.id, Math.round((r() * 6 - 3) * 100) / 100]));
  const { grad } = backward(g, forward(g, inp));
  for (const n of g.nodes.filter((n) => n.param)) {
    const s = nudge(g, inp, n.id).slope; total++;
    worst = Math.max(worst, Math.abs(s - grad[n.id]));
    if (s.toFixed(4) === grad[n.id].toFixed(4) || Math.abs(s - grad[n.id]) < 1e-12) agree++;
  }
}
console.log(`random inputs: ${agree} of ${total} nudge slopes match backprop to 4 decimals; largest absolute gap ${worst.toExponential(2)}`);
