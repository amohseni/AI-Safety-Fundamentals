// Claims on page 3.5, checked against the precomputed GPT-2 small data. Run: node tools/test-page-3-5.mjs
// The data come from tools/3-5/heads.py; the removal (ablation) numbers and the loss on the first
// token of the second copy come from tools/3-5/ablate.py (see tools/3-5/README.md).
import fs from 'node:fs';
const D = JSON.parse(fs.readFileSync(new URL('../docs/data/3-5/heads.json', import.meta.url)));
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const S = D.scores;
const heads = (M, f) => { const o = []; for (let l = 0; l < 12; l++) for (let h = 0; h < 12; h++) if (f(M[l][h])) o.push(`${l}.${h}`); return o; };
const val = (M, id) => { const [l, h] = id.split('.').map(Number); return M[l][h]; };
const same = (a, b) => a.length === b.length && a.every((x) => b.includes(x));

const ind = heads(S.ind, (v) => v > 0.7);
check('induction heads (score > 0.7) are 5.1, 5.5, 6.9, 7.2, 7.10', same(ind, ['5.1', '5.5', '6.9', '7.2', '7.10']), ind.join(' '));
const iv = ind.map((id) => val(S.ind, id));
check('their scores lie in 0.76 to 0.93', Math.min(...iv) >= 0.755 && Math.max(...iv) < 0.935, iv.join(' '));
const rest = heads(S.ind, (v) => v <= 0.7).map((id) => val(S.ind, id));
check('next highest induction score is 0.60', Math.abs(Math.max(...rest) - 0.60) < 0.005, Math.max(...rest));
const faint = heads(S.ind, (v) => v >= 0.45 && v <= 0.7);
check('six heads score 0.45 to 0.60: five in layers 9 to 11 and 5.0', same(faint, ['10.7', '10.1', '9.6', '11.10', '9.9', '5.0']), faint.join(' '));
const dup = heads(S.dup, (v) => v > 0.4);
check('duplicate-token heads (> 0.4) are 0.1, 0.5, 3.0', same(dup, ['0.1', '0.5', '3.0']), dup.join(' '));
const prev = heads(S.prev, (v) => v > 0.45);
check('previous-token heads (> 0.45) are 4.11, 3.7, 6.8, 2.2', same(prev, ['4.11', '3.7', '6.8', '2.2']), prev.map((id) => `${id}=${val(S.prev, id)}`).join(' '));
check('4.11 scores 0.99; 3.7 0.53, 6.8 0.50, 2.2 0.49', Math.round(val(S.prev, '4.11') * 100) === 99 && Math.round(val(S.prev, '3.7') * 100) === 53
  && Math.round(val(S.prev, '6.8') * 100) === 50 && Math.round(val(S.prev, '2.2') * 100) === 49, 'as printed above');
const nextPrev = Math.max(...heads(S.prev, (v) => v <= 0.45).map((id) => val(S.prev, id)));
check('no other head is close to the 0.45 threshold (next below 0.43)', nextPrev < 0.43, nextPrev);
check('over 100 sequences, loss averages 13.9 (first copy) and 0.53 (second copy)', D.loss_copies.first_mean.toFixed(1) === '13.9' && D.loss_copies.second_mean.toFixed(2) === '0.53', JSON.stringify(D.loss_copies));
check('13.9 is more than ln 50257', D.loss_copies.first_mean > Math.log(50257), Math.log(50257).toFixed(2));

function unpack(inp) {
  const b = Buffer.from(inp.att, 'base64'), n = inp.n, tri = n * (n + 1) / 2;
  return (l, h, i, j) => (j > i ? 0 : b[(l * 12 + h) * tri + i * (i + 1) / 2 + j] / 255);
}
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const R = D.inputs[0], T = D.meta.T;
check('displayed sequence: average loss 13.8 on copy 1 and 0.40 on copy 2 (first tokens not counted)',
  mean(R.loss.slice(2, T + 1)).toFixed(1) === '13.8' && mean(R.loss.slice(T + 2)).toFixed(2) === '0.40', `${mean(R.loss.slice(2, T + 1))} ${mean(R.loss.slice(T + 2))}`);
{
  const A = unpack(R);
  let on = 0, k = 0;
  for (let i = T + 1; i < R.n; i++) { on += A(5, 5, i, i - T + 1); k++; }
  check('figure: head 5.5 on the displayed sequence attends along the diagonal offset by 19', on / k > 0.8, (on / k).toFixed(3));
  let p = 0; for (let i = 2; i < R.n; i++) p += A(4, 11, i, i - 1);
  check('figure: head 4.11 attends to the previous token', p / (R.n - 2) > 0.9, (p / (R.n - 2)).toFixed(3));
  // Head 1.10: the most spread-out pattern (normalized entropy of rows from position 10 on).
  const ent = (l, h) => { let s = 0, c = 0; for (let i = 10; i < R.n; i++) { let z = 0; for (let j = 0; j <= i; j++) z += A(l, h, i, j); let e = 0; for (let j = 0; j <= i; j++) { const q = A(l, h, i, j) / z; if (q > 0) e -= q * Math.log(q); } s += e / Math.log(i + 1); c++; } return s / c; };
  let best = null; for (let l = 0; l < 12; l++) for (let h = 0; h < 12; h++) { const e = ent(l, h); if (!best || e > best[0]) best = [e, `${l}.${h}`]; }
  check('figure: head 1.10 is among the most spread-out (entropy within 0.02 of the maximum)', ent(1, 10) > best[0] - 0.02, `1.10 ${ent(1, 10).toFixed(3)}, max ${best[1]} ${best[0].toFixed(3)}`);
}
{
  const N = D.inputs.find((d) => d.key === 'names'), A = unpack(N);
  const t = N.tokens;
  const harry = t.map((x, i) => (x === ' Harry' ? i : -1)).filter((i) => i >= 0), potter = t.map((x, i) => (x === ' Potter' ? i : -1)).filter((i) => i >= 0);
  const row = (i) => t.map((_, j) => A(5, 5, i, j));
  const r2 = row(harry[1]), arg = r2.indexOf(Math.max(...r2));
  check('task 4: at the second Harry, 5.5 puts its largest weight, 0.58, on the first Potter', arg === potter[0] && r2[arg].toFixed(2) === '0.58', `${arg} ${r2[arg]}`);
  check('task 4: most of the rest on the first token', r2[0] > 0.25 && r2[0] + r2[arg] > 0.85, r2[0]);
  const r1 = row(harry[0]);
  check('task 4: at the first Harry, 0.94 on the first token', r1[0].toFixed(2) === '0.94', r1[0]);
  check('task 4: loss 2.42 on the second Potter, 2.59 on the first', N.loss[potter[1]].toFixed(2) === '2.42' && N.loss[potter[0]].toFixed(2) === '2.59', `${N.loss[potter[1]]} ${N.loss[potter[0]]}`);
}
console.log(fails ? `${fails} FAILED` : 'all ok');
process.exit(fails ? 1 : 0);
