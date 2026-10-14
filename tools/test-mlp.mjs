// Checks the claims page 2.4 relies on: which datasets each width can separate.
// Run: node tools/test-mlp.mjs
import { MLP, datasets } from '../docs/js/lib/mlp.js';

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
}

function run(name, sizes, steps, seeds = [1, 2, 3, 4, 5], lr = 0.03, act = 'tanh') {
  const { X, Y } = datasets[name]();
  const accs = seeds.map((seed) => {
    const net = new MLP(sizes, { seed, lr, act });
    for (let i = 0; i < steps; i++) net.step(X, Y);
    return net.accuracy(X, Y);
  });
  console.log(`${name.padEnd(8)} sizes=${JSON.stringify(sizes).padEnd(14)} ${act} steps=${steps} acc=`, accs.map((a) => a.toFixed(2)).join(' '));
}

run('xor', [2, 2, 2, 1], 1500);
run('circles', [2, 2, 2, 1], 3000);
run('circles', [2, 3, 3, 1], 1500);
run('spirals', [2, 2, 2, 1], 3000);
run('spirals', [2, 3, 3, 1], 3000);
run('spirals', [2, 3, 3, 3, 1], 3000);
run('spirals', [2, 3, 3, 3, 3, 1], 3000);
