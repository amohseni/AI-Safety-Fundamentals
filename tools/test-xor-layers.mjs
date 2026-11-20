// Is XOR linearly separable after layer 1, for width-2 tanh nets? Checked with a perceptron run.
import { MLP, datasets } from '../docs/js/lib/mlp.js';
const { X, Y } = datasets.xor();
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const rows = [];
function separable(P, Y) {
  let w = [0, 0], b = 0;
  for (let epoch = 0; epoch < 2000; epoch++) {
    let errs = 0;
    P.forEach((p, i) => { const s = (Y[i] ? 1 : -1); if (s * (w[0]*p[0] + w[1]*p[1] + b) <= 0) { w[0] += s*p[0]; w[1] += s*p[1]; b += s; errs++; } });
    if (!errs) return true;
  }
  return false;
}
for (const seed of [1,2,3,4,5,6,7,8]) {
  const net = new MLP([2,2,2,1], { seed });
  for (let i = 0; i < 1500; i++) net.step(X, Y);
  const S = X.map((x) => net.stages(x));
  const row = { seed, acc: net.accuracy(X,Y), input: separable(X,Y), z1: separable(S.map(s=>s[1]),Y), a1: separable(S.map(s=>s[2]),Y), a2: separable(S.map(s=>s[4]),Y) };
  rows.push(row);
  console.log('seed', seed, 'acc', row.acc.toFixed(2), 'input', row.input, 'z1', row.z1, 'a1', row.a1, 'a2', row.a2);
}
// Page 2.4, task 1: "In every successful run we tested, the classes first become separable after layer 1's activation."
const won = rows.filter((r) => r.acc === 1);
check('a width-2 network usually learns XOR (at least 6 of 8 starting points reach 100%)', won.length >= 6, `${won.length} of ${rows.length}`);
check('no line separates the classes in the input', rows.every((r) => !r.input), rows.map((r) => r.input).join(' '));
check("layer 1's linear map never makes the classes separable", rows.every((r) => !r.z1), rows.map((r) => r.z1).join(' '));
check("in every successful run the classes are separable after layer 1's activation, and stay separable after layer 2's", won.every((r) => r.a1 && r.a2), won.map((r) => `${r.a1}/${r.a2}`).join(' '));
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
