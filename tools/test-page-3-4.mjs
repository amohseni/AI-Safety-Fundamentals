// Claims on page 3.4, checked numerically. Run: node tools/test-page-3-4.mjs
import { attention, mix } from '../docs/js/lib/3-models.js';
import { START, LIMIT, FIGURE } from '../docs/js/page/3-4-data.js';
import { rng } from '../docs/js/lib/mlp.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const f2 = (a) => a.map((v) => (v === null ? '-' : v.toFixed(3))).join(' ');
const r = rng(7);
const rv = () => [(r() * 2 - 1) * LIMIT, (r() * 2 - 1) * LIMIT];

// The formula: softmax(Q K^T / sqrt(d) + M) V, computed directly with matrices, matches the library.
{
  let maxErr = 0;
  for (let t = 0; t < 200; t++) {
    const n = 4, Q = [...Array(n)].map(rv), K = [...Array(n)].map(rv), V = [...Array(n)].map(rv);
    for (const causal of [true, false]) for (const scale of [true, false]) {
      const { weights } = attention(Q, K, { causal, scale });
      for (let i = 0; i < n; i++) {
        const s = K.map((k, j) => (causal && j > i ? -Infinity : (Q[i][0] * k[0] + Q[i][1] * k[1]) / (scale ? Math.SQRT2 : 1)));
        const e = s.map(Math.exp), z = e.reduce((a, b) => a + b, 0);
        const w = e.map((x) => x / z);
        const o1 = mix(weights[i], V), o2 = [0, 1].map((c) => w.reduce((a, wj, j) => a + wj * V[j][c], 0));
        maxErr = Math.max(maxErr, ...w.map((x, j) => Math.abs(x - weights[i][j])), ...o1.map((x, c) => Math.abs(x - o2[c])));
        if (Math.abs(weights[i].reduce((a, b) => a + b, 0) - 1) > 1e-12) maxErr = Infinity;
      }
    }
  }
  check('library matches softmax(QK^T/sqrt(d) + M)V, rows sum to 1', maxErr < 1e-12, `max error ${maxErr.toExponential(1)}`);
}

const W0 = attention(START.q, START.k).weights;
console.log('     starting weights (mask on, scaled):\n' + W0.map((w, i) => `       token ${i + 1}: ${f2(w)}`).join('\n'));

// Task 1: q4 along k2 at its starting length is not enough; length about 2.75 is.
{
  const k2 = START.k[1], u = Math.hypot(...k2), L0 = Math.hypot(...START.q[3]);
  const w = (L) => { const q = START.q.slice(); q[3] = [k2[0] / u * L, k2[1] / u * L]; return attention(q, START.k).weights[3][1]; };
  check('task 1: q4 pointed along k2 at its starting length gives about 0.64', Math.abs(w(L0) - 0.642) < 0.005, `|q4| = ${L0.toFixed(2)}, weight ${w(L0).toFixed(3)}`);
  let need = null; for (let L = L0; L < 6; L += 0.001) if (w(L) > 0.9) { need = L; break; }
  const tip = [k2[0] / u * need, k2[1] / u * need];
  check('task 1: weight above 0.9 reachable inside the plane by lengthening q4', need !== null && Math.max(...tip.map(Math.abs)) <= LIMIT, `needs length ${need.toFixed(2)} (tip ${tip.map((v) => v.toFixed(2)).join(', ')}); at length 3: ${w(3).toFixed(3)}`);
}
// Task 2: two identical keys get identical weights from every query.
{
  let maxGap = 0;
  for (let t = 0; t < 1000; t++) { const k = START.k.slice(); k[2] = k[1].slice(); const q = START.q.slice(); q[3] = rv(); const w = attention(q, k).weights[3]; maxGap = Math.max(maxGap, Math.abs(w[1] - w[2])); }
  check('task 2: k3 = k2 gives equal weights on tokens 2 and 3 for any q4', maxGap < 1e-12, `max gap ${maxGap.toExponential(1)} over 1000 random q4`);
}
// Task 3: with the mask on, token 1 always puts weight 1 on itself; with the mask off it does not.
{
  let minW = 1, offMin = 1;
  for (let t = 0; t < 1000; t++) { const q = START.q.slice(); q[0] = rv(); minW = Math.min(minW, attention(q, START.k).weights[0][0]); offMin = Math.min(offMin, attention(q, START.k, { causal: false }).weights[0][0]); }
  check('task 3: mask on, token 1 weight on itself is always 1', minW === 1, `min ${minW}`);
  check('task 3: mask off, token 1 spreads its weight', offMin < 0.5, `starting row with mask off: ${f2(attention(START.q, START.k, { causal: false }).weights[0])}`);
}
// Task 4: removing the scaling raises the largest weight (unless all allowed scores are equal).
{
  const a = Math.max(...W0[3]), b = Math.max(...attention(START.q, START.k, { scale: false }).weights[3]);
  check('task 4: token 4 largest weight rises without scaling', b > a, `${a.toFixed(3)} with scaling, ${b.toFixed(3)} without`);
  let bad = 0;
  for (let t = 0; t < 2000; t++) { const q = [...Array(4)].map(rv), k = [...Array(4)].map(rv); const s = attention(q, k).weights, u = attention(q, k, { scale: false }).weights; for (let i = 0; i < 4; i++) if (Math.max(...u[i]) < Math.max(...s[i]) - 1e-12) bad++; }
  check('task 4: holds for random queries and keys', bad === 0, `${bad} violations in 8000 rows`);
}
// Check question: all keys identical, causal mask: position 3 outputs (v1 + v2 + v3) / 3.
{
  const k = [...Array(4)].map(() => [0.7, -1.2]);
  let maxErr = 0;
  for (let t = 0; t < 500; t++) {
    const q = [...Array(4)].map(rv), V = [...Array(4)].map(rv);
    const w = attention(q, k).weights;
    for (let i = 0; i < 4; i++) { const avg = [0, 1].map((c) => V.slice(0, i + 1).reduce((a, v) => a + v[c], 0) / (i + 1)); const o = mix(w[i], V); maxErr = Math.max(maxErr, ...o.map((x, c) => Math.abs(x - avg[c]))); }
  }
  check('check: identical keys give the plain average of the allowed values', maxErr < 1e-12, `max error ${maxErr.toExponential(1)}`);
}
// Static figure numbers.
{
  const { weights, scores } = attention([FIGURE.q, FIGURE.q, FIGURE.q], FIGURE.k);
  const o = mix(weights[2], FIGURE.v);
  console.log(`     figure: scores ${f2(scores[2])}, weights ${f2(weights[2])}, output (${o.map((v) => v.toFixed(2)).join(', ')})`);
}
process.exit(fails ? 1 : 0);
