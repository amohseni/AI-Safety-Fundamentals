// Numbers quoted on page 2.4, recomputed with the page's own settings (6,000 steps, lr 0.03, tanh).
// Run: node tools/test-page-2-4.mjs
import { MLP, datasets } from '../docs/js/lib/mlp.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const SEEDS = { xor: [1, 2, 4, 5, 6, 7], spirals: [1, 3, 2, 4, 5, 6, 7, 8] };
function run(name, width, seeds, steps = 6000) {
  const { X, Y } = datasets[name]();
  return seeds.map((seed) => { const n = new MLP([2, width, width, 1], { seed, lr: 0.03 }); for (let i = 0; i < steps; i++) n.step(X, Y); return Math.round(n.accuracy(X, Y) * 1000) / 10; });
}
const xor2 = run('xor', 2, SEEDS.xor), sp2 = run('spirals', 2, SEEDS.spirals), sp3 = run('spirals', 3, SEEDS.spirals);
console.log('xor width 2  ', xor2.join(' '));
console.log('spirals w2   ', sp2.join(' '));
console.log('spirals w3   ', sp3.join(' '));
const lo = (a) => Math.round(Math.min(...a)), hi = (a) => Math.round(Math.max(...a));
// Task 1 asks the reader to train XOR at width 2 until the accuracy reaches 100%.
check("t1: XOR, width 2 reaches 100% from every one of the page's starting points", xor2.every((a) => a === 100), xor2.join(' '));
// Task 2 notice: "At width 2, training stalls well short of 100%: in our runs, between 65% and 84%."
check('t2: spirals, width 2 end between 65% and 84%', lo(sp2) === 65 && hi(sp2) === 84, `${lo(sp2)} to ${hi(sp2)}`);
// "At width 3 ... the first two starting points reach 100%, and the others end between 85% and 99%."
check('t2: spirals, width 3: the first two starting points reach 100%', sp3[0] === 100 && sp3[1] === 100, sp3.slice(0, 2).join(' '));
check('t2: spirals, width 3: the others end between 85% and 99%', lo(sp3.slice(2)) === 85 && hi(sp3.slice(2)) === 99, `${lo(sp3.slice(2))} to ${hi(sp3.slice(2))}`);
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
