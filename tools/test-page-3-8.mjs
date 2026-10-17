// Numbers quoted on page 3.8, from the Chinchilla fit in docs/js/lib/3-models.js. Run: node tools/test-page-3-8.mjs
import { FIT, loss, tokensFor, optimum, optimumNumeric } from '../docs/js/lib/3-models.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const B = (x) => (x / 1e9).toFixed(1) + 'B', T = (x) => (x / 1e12).toFixed(2) + 'T';
const C0 = 5.76e23;
const PRINTED = { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 };

const o = optimum(C0), on = optimumNumeric(C0);
check('closed form matches numerical optimum', Math.abs(o.N / on.N - 1) < 0.003, `${B(o.N)} vs ${B(on.N)}`);
check('exponent a = beta/(alpha+beta) is about 0.46 (paper Table 2: 0.46)', Math.abs(o.a - 0.4565) < 0.001, o.a.toFixed(4));
const op = optimum(C0, PRINTED);
console.log(`     with the printed rounded constants: a = ${op.a.toFixed(4)}, optimum ${B(op.N)}, ${T(op.D)}, ${(op.D / op.N).toFixed(0)} tokens per parameter`);
check('optimum at 5.76e23 is about 40B (paper, Figure 4: 40B)', Math.abs(o.N / 1e9 - 40) < 1, `${B(o.N)} parameters, ${T(o.D)} tokens, ${(o.D / o.N).toFixed(0)} tokens per parameter, loss ${loss(o.N, o.D).toFixed(3)}`);
const L70 = loss(70e9, tokensFor(C0, 70e9)), L280 = loss(280e9, tokensFor(C0, 280e9));
check('task 1: 70B beats 280B at 5.76e23', L70 < L280, `70B: D ${T(tokensFor(C0, 70e9))}, loss ${L70.toFixed(3)}; 280B: D ${T(tokensFor(C0, 280e9))}, loss ${L280.toFixed(3)}`);
check('task 1: 70B ratio near 20 tokens per parameter', Math.abs(tokensFor(C0, 70e9) / 70e9 - 19.6) < 0.1, (tokensFor(C0, 70e9) / 70e9).toFixed(1));
const o10 = optimum(10 * C0);
check('task 2: 10x compute multiplies optimal N by about 2.9', Math.abs(o10.N / o.N - 2.86) < 0.02, `N x${(o10.N / o.N).toFixed(2)}, D x${(o10.D / o.D).toFixed(2)}; square root would be x${Math.sqrt(10).toFixed(2)}`);
const o100 = optimum(100 * C0);
console.log(`     100x compute under the fit: N x${(o100.N / o.N).toFixed(1)}, D x${(o100.D / o.D).toFixed(1)}`);
const chin = 6 * 70e9 * 1.4e12, n100 = Math.sqrt(100 * chin / 120);
check('check: 100x Chinchilla compute at 20 tokens/parameter gives 700B and 14T', Math.abs(n100 / 700e9 - 1) < 0.001, `C = ${chin.toExponential(2)}; N = ${B(n100)}, D = ${T(20 * n100)}`);
for (const S of [0, 1e12, 1e13]) {
  const r = optimumNumeric(C0, S);
  console.log(`     serving ${S.toExponential(0)} tokens: best N ${B(r.N)}, D ${T(r.D)}, loss ${r.L.toFixed(3)}`);
}
const s13 = optimumNumeric(C0, 1e13);
check('task 3: serving 1e13 tokens moves the optimum to a smaller model', s13.N < o.N * 0.5, `${B(s13.N)} vs ${B(o.N)}`);
// Monotone: more serving, smaller optimum, at every budget on the slider.
let mono = true;
for (let lc = 19; lc <= 26; lc += 0.5) { let prev = Infinity; for (let ls = 9; ls <= 14; ls += 0.25) { const n = optimumNumeric(10 ** lc, 10 ** ls).N; if (Number.isFinite(n) && n > prev * 1.002) mono = false; if (Number.isFinite(n)) prev = n; } }
check('more serving never makes the optimal model larger (budgets 1e19 to 1e26)', mono, mono);
const ratios = [1e18, 1e20, 1e22, 5.76e23, 1e24, 1e26].map((C) => { const r = optimum(C); return `${C.toExponential(0)}: ${(r.D / r.N).toFixed(0)}`; });
console.log(`     tokens per parameter at the optimum, by budget: ${ratios.join(', ')}`);
check('loss floor E', FIT.E === 1.6934, FIT.E);
process.exit(fails ? 1 : 0);
