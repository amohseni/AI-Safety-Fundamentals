// Is XOR linearly separable after layer 1, for width-2 tanh nets? Checked with a perceptron run.
import { MLP, datasets } from '../docs/js/lib/mlp.js';
const { X, Y } = datasets.xor();
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
  console.log('seed', seed, 'acc', net.accuracy(X,Y).toFixed(2), 'input', separable(X,Y), 'z1', separable(S.map(s=>s[1]),Y), 'a1', separable(S.map(s=>s[2]),Y), 'a2', separable(S.map(s=>s[4]),Y));
}
