// Numbers behind page 2.7 (gradients by hand). Run: node tools/test-page-2-7.mjs
import { GRAPHS, forward, backward, chainPaths, nudge } from '../docs/js/lib/s2-graph.js';
import { rng } from '../docs/js/lib/mlp.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const out = {};
const defaults = (g) => Object.fromEntries(g.nodes.filter((n) => n.op === 'input').map((n) => [n.id, n.value]));
for (const [k, g] of Object.entries(GRAPHS)) {
  const inp = defaults(g), val = forward(g, inp), { grad, parts } = backward(g, val);
  out[k] = { inp, val, grad, parts, params: [] };
  console.log(`graph ${k}: values`, Object.entries(val).map(([i, v]) => `${i}=${v.toFixed(4)}`).join(' '));
  console.log('  grads ', Object.entries(grad).map(([i, v]) => `${i}=${v.toFixed(4)}`).join(' '));
  console.log('  nodes receiving more than one derivative:', Object.entries(parts).filter(([, p]) => p.length > 1).map(([i, p]) => `${i} (${p.map((q) => q.value.toFixed(4)).join(' + ')})`).join(', ') || 'none');
  for (const n of g.nodes.filter((n) => n.param)) {
    const paths = chainPaths(g, val, n.id), sum = paths.reduce((s, p) => s + p.product, 0), t = nudge(g, inp, n.id);
    out[k].params.push({ id: n.id, grad: grad[n.id], sum, slope: t.slope, npaths: paths.length });
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
const f3 = (x) => x.toFixed(3), f4 = (x) => x.toFixed(4);
{
  const { inp, val, grad, params } = out.one;
  // Caption: "x = 1.5, w1 = 0.8, b = -0.5, w2 = 2, y = 0.5".
  check('caption: the default values are x = 1.5, w1 = 0.8, b = -0.5, w2 = 2, y = 0.5', inp.x === 1.5 && inp.w1 === 0.8 && inp.b === -0.5 && inp.w2 === 2 && inp.y === 0.5, JSON.stringify(inp));
  check('caption: the graph computes L = (w2 sigma(w1 x + b) - y)^2', Math.abs(val.L - (2 / (1 + Math.exp(-(0.8 * 1.5 - 0.5))) - 0.5) ** 2) < 1e-12, val.L);
  // Task 1: "the forward pass gives p = 1.336 while the target is y = 0.5"; notice: "dL/dw2 = 1.118 ... d = p - y = 0.836 ... a = 0.668".
  check('t1: the forward pass gives p = 1.336', f3(val.p) === '1.336', val.p);
  check('t1: dL/dw2 = 1.118, positive', f3(grad.w2) === '1.118', grad.w2);
  check('t1: d = p - y = 0.836 and a = 0.668', f3(val.d) === '0.836' && f3(val.a) === '0.668', `${val.d} ${val.a}`);
  // Task 2 notice: "Yes: both are 1.1126 at the default values."
  const w1 = params.find((p) => p.id === 'w1');
  check('t2: for w1 the nudge slope and the backpropagated derivative are both 1.1126', f4(w1.slope) === '1.1126' && f4(w1.grad) === '1.1126', `${w1.slope} ${w1.grad}`);
}
{
  // Task 3 notice: "At h. ... With the default values the two derivatives are -2.2370 and 0.3412, and dL/dh is their sum, -1.8958."
  const { grad, parts } = out.two, many = Object.entries(parts).filter(([, p]) => p.length > 1);
  check('t3: h is the only node where two derivatives arrive', many.length === 1 && many[0][0] === 'h' && many[0][1].length === 2, many.map(([i]) => i).join(' '));
  const vals = parts.h.map((q) => f4(q.value)).sort();
  check('t3: the two derivatives are -2.2370 and 0.3412', vals.join(' ') === '-2.2370 0.3412', vals.join(' '));
  check('t3: dL/dh is their sum, -1.8958', f4(grad.h) === '-1.8958' && Math.abs(parts.h[0].value + parts.h[1].value - grad.h) < 1e-12, grad.h);
}
check('for every parameter of both graphs, the sum of chain-rule products over paths equals the backpropagated derivative', Object.values(out).every((o) => o.params.every((p) => Math.abs(p.sum - p.grad) < 1e-12)), Object.values(out).flatMap((o) => o.params.map((p) => (p.sum - p.grad).toExponential(1))).join(' '));
// Task 2 notice: "In 120,000 comparisons at random settings of the sliders, the two numbers never differed by more than 0.00003,
// and 99.9% agreed to all four decimals."
check('t2: 120,000 comparisons', total === 120000, total);
check('t2: the two numbers never differ by more than 0.00003', worst <= 0.00003, worst);
check('t2: 99.9% agree to all four decimals', ((100 * agree) / total).toFixed(1) === '99.9', (100 * agree) / total);
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
