// Numbers quoted on page 2.4, recomputed with the page's own settings (6,000 steps, lr 0.03, tanh).
// Run: node tools/test-page-2-4.mjs
import { MLP, datasets } from '../docs/js/lib/mlp.js';
const SEEDS = { xor: [1, 2, 4, 5, 6, 7], spirals: [1, 3, 2, 4, 5, 6, 7, 8] };
function run(name, width, seeds, steps = 6000) {
  const { X, Y } = datasets[name]();
  return seeds.map((seed) => { const n = new MLP([2, width, width, 1], { seed, lr: 0.03 }); for (let i = 0; i < steps; i++) n.step(X, Y); return Math.round(n.accuracy(X, Y) * 1000) / 10; });
}
console.log('xor width 2  ', run('xor', 2, SEEDS.xor).join(' '));
console.log('spirals w2   ', run('spirals', 2, SEEDS.spirals).join(' '));
console.log('spirals w3   ', run('spirals', 3, SEEDS.spirals).join(' '));
