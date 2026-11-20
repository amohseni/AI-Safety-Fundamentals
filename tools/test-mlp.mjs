// Checks the claims page 2.4 and its spec rely on: which datasets each width can separate.
// (The numbers quoted on page 2.4 itself are asserted in test-page-2-4.mjs, with the page's 6,000 steps.)
// Run: node tools/test-mlp.mjs
import { MLP, datasets } from '../docs/js/lib/mlp.js';

let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };

// Numerical gradient check on a tiny net.
{
  const net = new MLP([2, 3, 3, 1], { seed: 7 });
  const { X, Y } = datasets.xor(20);
  const lossAt = () => net.loss(X, Y);
  // analytic gradient for W[0][1][0] via finite difference of step direction is awkward with Adam;
  // instead compare the loss decrease of a tiny SGD-like nudge.
  const h = 1e-5, base = lossAt();
  net.W[0][1][0] += h; const up = lossAt(); net.W[0][1][0] -= 2 * h; const dn = lossAt(); net.W[0][1][0] += h;
  const numeric = (up - dn) / (2 * h);
  const analytic = net.grad(X, Y).g.W[0][1][0] / X.length;
  console.log('gradient check: numeric', numeric.toFixed(6), 'analytic', analytic.toFixed(6), 'base loss', base.toFixed(4));
  check('backpropagation agrees with a finite-difference gradient to 1e-6', Math.abs(numeric - analytic) < 1e-6, Math.abs(numeric - analytic).toExponential(2));
}

function run(name, sizes, steps, seeds = [1, 2, 3, 4, 5], lr = 0.03, act = 'tanh') {
  const { X, Y } = datasets[name]();
  const accs = seeds.map((seed) => {
    const net = new MLP(sizes, { seed, lr, act });
    for (let i = 0; i < steps; i++) net.step(X, Y);
    return net.accuracy(X, Y);
  });
  console.log(`${name.padEnd(8)} sizes=${JSON.stringify(sizes).padEnd(14)} ${act} steps=${steps} acc=`, accs.map((a) => a.toFixed(2)).join(' '));
  return accs;
}

const pct = (a) => Math.round(a * 100), all = (a, f) => a.every(f), list = (a) => a.map(pct).join(' ');
const xor2 = run('xor', [2, 2, 2, 1], 1500);
check('XOR, width 2: the network usually learns it (at least 4 of 5 starting points reach 100%)', xor2.filter((a) => a === 1).length >= 4, list(xor2));
const circ2 = run('circles', [2, 2, 2, 1], 3000);
check('spec 2.4: a width-2 network classifies circles at 94 to 99% accuracy', all(circ2, (a) => pct(a) >= 94 && pct(a) <= 99), list(circ2));
const circ3 = run('circles', [2, 3, 3, 1], 1500);
check('circles, width 3: every starting point reaches 100%', all(circ3, (a) => a === 1), list(circ3));
const sp2 = run('spirals', [2, 2, 2, 1], 3000);
check('spirals, width 2: every starting point stalls well short of 100% (at most 85%)', all(sp2, (a) => pct(a) <= 85), list(sp2));
const sp3 = run('spirals', [2, 3, 3, 1], 3000);
check('spirals, width 3: accuracy goes higher than at width 2 (every run at least 84%, best run 100%)', all(sp3, (a) => pct(a) >= 84) && Math.max(...sp3) === 1 && Math.min(...sp3) > Math.max(...sp2), list(sp3));
run('spirals', [2, 3, 3, 3, 1], 3000);
run('spirals', [2, 3, 3, 3, 3, 1], 3000);
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
