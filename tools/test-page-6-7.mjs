// Numbers quoted on page 6.7, computed from the exported parameters (docs/data/6-7/model.json)
// with the same code the page runs (docs/js/lib/s67-modadd.js).
// Run: node tools/test-page-6-7.mjs   (about a minute)
import fs from 'fs';
import { ModAdd, fourierBasis } from '../docs/js/lib/s67-modadd.js';

const json = JSON.parse(fs.readFileSync(new URL('../docs/data/6-7/model.json', import.meta.url)));
const M = new ModAdd(json);
const { p, d, m, H } = M;
const pct = (v) => (v * 100).toFixed(1) + '%';
const quant = (xs, q) => { const s = Float64Array.from(xs).sort(); return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]; };
const min = (xs) => xs.reduce((s, v) => Math.min(s, v), Infinity);
const max = (xs) => xs.reduce((s, v) => Math.max(s, v), -Infinity);
const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;
let failed = 0;
const check = (name, ok) => { console.log((ok ? 'ok   ' : 'FAIL ') + name); if (!ok) failed++; };

// ---- 1. The model and the frequency panel ----
const full = M.accuracy();
check(`test accuracy of the exported model is 100% (${pct(full.testAcc)})`, full.testAcc === 1);
const allRight = full.pred.every((c, i) => c === (Math.floor(i / p) + (i % p)) % p);
check('every one of the 3,481 pairs is answered correctly', allRight);

const spec = M.spectrum();
const order = spec.map((v, i) => [v, i + 1]).sort((x, y) => y[0] - x[0]);
const tot = spec.reduce((s, v) => s + v * v, 0);
let cum = 0; const keys = [];
for (const [v, k] of order) { cum += v * v; keys.push(k); if (cum / tot >= 0.9) break; }
keys.sort((x, y) => x - y);
console.log('spectrum, five largest:', order.slice(0, 5).map(([v, k]) => `${k}: ${v.toFixed(2)}`).join(', '));
check(`key frequencies are 1, 12, 27 (found ${keys.join(', ')}); they hold ${pct(cum / tot)} of the embedding`, keys.join() === '1,12,27');
const next = order[3][0];
console.log(`largest key ${order[0][0].toFixed(2)}, smallest key ${order[2][0].toFixed(2)}, largest other (frequency ${order[3][1]}) ${next.toFixed(2)}`);

const keepSet = (ks) => { const k = {}; for (let j = 1; j <= 29; j++) k[j] = ks.includes(j); return k; };
const accWith = (ks) => { M.setEmbedding(M.ablated(keepSet(ks))); const a = M.accuracy().testAcc; M.setEmbedding(null); return a; };
const all = Array.from({ length: 29 }, (_, i) => i + 1);
const onlyKeys = accWith(keys);
check(`keeping only the key frequencies: test accuracy ${pct(onlyKeys)} (page: 100%)`, onlyKeys === 1);
const noKeys = accWith(all.filter((k) => !keys.includes(k)));
check(`removing the three key frequencies: ${pct(noKeys)} (page: 1.8%, chance 1.7%)`, Math.abs(noKeys - 0.0176) < 0.001);
const removeOne = keys.map((k) => accWith(all.filter((j) => j !== k)));
console.log('removing one key frequency:', keys.map((k, i) => `${k}: ${pct(removeOne[i])}`).join(', '));
check('removing one key frequency leaves 30% to 38% (page: 31% to 37%)', removeOne.every((a) => a > 0.30 && a < 0.38));
const removeOther = all.filter((k) => !keys.includes(k)).map((k) => accWith(all.filter((j) => j !== k)));
check(`removing any one other frequency leaves 100% (min ${pct(Math.min(...removeOther))})`, Math.min(...removeOther) === 1);
const oneKey = keys.map((k) => accWith([k]));
console.log('keeping one key frequency only:', keys.map((k, i) => `${k}: ${pct(oneKey[i])}`).join(', '));
const seeds = JSON.parse(fs.readFileSync(new URL('../docs/data/modadd/seeds.json', import.meta.url))).seeds[0];
check('remove-one accuracies agree with docs/data/modadd/seeds.json within 0.1 points',
  keys.every((k, i) => Math.abs(removeOne[i] - seeds.acc_remove_one[k]) < 0.001));

for (const k of [1, 12, 27, 5]) {
  const r = M.circle(k).map(([x, y]) => Math.hypot(x, y));
  console.log(`circle at frequency ${k}: distance from the center ${Math.min(...r).toFixed(2)} to ${Math.max(...r).toFixed(2)}`);
}
const ringOk = keys.every((k) => { const r = M.circle(k).map(([x, y]) => Math.hypot(x, y)); return Math.min(...r) / Math.max(...r) > 0.75; });
check('at each key frequency the numbers lie near a circle (min/max distance > 0.75)', ringOk);
// Neighboring numbers on the circle at frequency k are a step of k apart in angle: check the angle order.
{
  const pts = M.circle(12), ang = pts.map(([x, y]) => Math.atan2(y, x));
  const steps = [];
  for (let a = 0; a < p; a++) { let dlt = ang[(a + 1) % p] - ang[a]; dlt = Math.atan2(Math.sin(dlt), Math.cos(dlt)); steps.push(Math.abs(dlt)); }
  const want = (2 * Math.PI * 12) / p, want2 = Math.min(want, 2 * Math.PI - want);
  console.log(`frequency 12: angle from a to a + 1 is ${(Math.min(...steps) * 180 / Math.PI).toFixed(0)} to ${(Math.max(...steps) * 180 / Math.PI).toFixed(0)} degrees (12/59 of a turn is ${(want2 * 180 / Math.PI).toFixed(0)})`);
}

// ---- 2. Neurons and heads by frequency ----
const F = fourierBasis(p);
const acts = []; // acts[i] = activations for pair i
for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) acts.push(M.run(a, b).act);
let match = 0, strong = 0;
const counts = {};
for (let n = 0; n < m; n++) {
  const X = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (_, b) => acts[a * p + b][n]));
  // C = F X F^T
  const T = F.map((row) => Array.from({ length: p }, (_, b) => row.reduce((s, f, a) => s + f * X[a][b], 0)));
  const C = T.map((row) => F.map((col) => row.reduce((s, v, b) => s + v * col[b], 0)));
  let total = 0; C.forEach((r) => r.forEach((v) => { total += v * v; })); total -= C[0][0] ** 2;
  let best = 0, bk = 0;
  for (let k = 1; k <= 29; k++) {
    const ix = [0, 2 * k - 1, 2 * k]; let e = -(C[0][0] ** 2);
    for (const i of ix) for (const j of ix) e += C[i][j] ** 2;
    if (e / total > best) { best = e / total; bk = k; }
  }
  if (bk === M.neuronFreq[n]) match++;
  if (best > 0.85) strong++;
  counts[bk] = (counts[bk] || 0) + 1;
}
check(`dominant frequency of each neuron agrees with the exported list (${match} of ${m})`, match === m);
console.log('neurons by dominant frequency:', JSON.stringify(counts), `; more than 85% of the variance at that frequency: ${strong} of ${m}`);
check('every neuron has a key frequency as its dominant frequency', Object.keys(counts).every((k) => keys.includes(Number(k))));

for (let h = 0; h < H; h++) {
  // What head h writes from position 0, as a function of a: its frequency content.
  const out = Array.from({ length: p }, (_, a) => M.pos[0][a].vo[h]);
  const sq = F.map((row) => { let s = 0; for (let j = 0; j < d; j++) { let c = 0; for (let a = 0; a < p; a++) c += row[a] * out[a][j]; s += c * c; } return s; });
  const per = Array.from({ length: 29 }, (_, i) => sq[2 * i + 1] + sq[2 * i + 2]);
  const t = per.reduce((s, v) => s + v, 0);
  console.log(`head ${h} output from a: share by frequency 1, 12, 27: ${[1, 12, 27].map((k) => pct(per[k - 1] / t)).join(', ')}`);
}
{
  const att = []; for (let i = 0; i < p * p; i++) att.push(M.run(Math.floor(i / p), i % p).att);
  for (let h = 0; h < H; h++) {
    const toA = att.map((x) => x[h * 3]), toB = att.map((x) => x[h * 3 + 1]), toEq = att.map((x) => x[h * 3 + 2]);
    console.log(`head ${h} attention: to a ${pct(mean(toA))} (${pct(Math.min(...toA))} to ${pct(Math.max(...toA))}), to b ${pct(mean(toB))}, to "=" at most ${pct(Math.max(...toEq))}`);
  }
}

// ---- 3. Activation patching over every clean pair and every corrupted value of a (and of b) ----
// Fast exact computation using linearity downstream of the patched activation; checked against M.patch.
const runs = []; for (let i = 0; i < p * p; i++) runs.push(M.run(Math.floor(i / p), i % p));
const U = M.P.W_U, Win = M.P.W_in;
const hoIn = runs.map((r) => Array.from({ length: H }, (_, h) => {
  const o = new Float64Array(m); for (let j = 0; j < d; j++) { const v = r.ho[h * d + j]; for (let n = 0; n < m; n++) o[n] += v * Win[j * m + n]; } return o;
}));
const preOf = runs.map((r, i) => { // MLP pre-activation, recomputed from head outputs
  const o = new Float64Array(m); for (let n = 0; n < m; n++) o[n] = M.P.b_in[n];
  const x = Float64Array.from(M.eq.x); for (let h = 0; h < H; h++) for (let j = 0; j < d; j++) x[j] += r.ho[h * d + j];
  for (let j = 0; j < d; j++) for (let n = 0; n < m; n++) o[n] += x[j] * Win[j * m + n];
  return { pre: o, rmid: x };
});
const uDiff = (x, c, c2) => { let s = 0; for (let j = 0; j < d; j++) s += x[j] * (U[j * p + c] - U[j * p + c2]); return s; };
const COMPS = ['head 0', 'head 1', 'head 2', 'head 3', 'all heads', 'MLP', 'freq 1', 'freq 12', 'freq 27'];
function restoredAll(i, i2) {
  const a = Math.floor(i / p), b = i % p, c = (a + b) % p, c2 = (Math.floor(i2 / p) + (i2 % p)) % p;
  const cl = runs[i], co = runs[i2];
  const ldC = cl.logits[c] - cl.logits[c2], ldX = co.logits[c] - co.logits[c2];
  const out = [];
  const wd = new Float64Array(m); for (let n = 0; n < m; n++) wd[n] = M.WoU[n * p + c] - M.WoU[n * p + c2];
  for (let h = 0; h < H; h++) {
    const pre = preOf[i2].pre, a1 = hoIn[i][h], a0 = hoIn[i2][h];
    let s = uDiff(M.P.b_out, c, c2);
    const rm = Float64Array.from(preOf[i2].rmid); for (let j = 0; j < d; j++) rm[j] += cl.ho[h * d + j] - co.ho[h * d + j];
    s += uDiff(rm, c, c2);
    for (let n = 0; n < m; n++) { const v = pre[n] + a1[n] - a0[n]; if (v > 0) s += v * wd[n]; }
    out.push(s);
  }
  out.push(ldC); // all four heads: the "=" position carries nothing else, so this is the clean run
  { const x = Float64Array.from(preOf[i2].rmid); for (let j = 0; j < d; j++) x[j] += cl.mo[j]; out.push(uDiff(x, c, c2)); }
  for (const k of [1, 12, 27]) { let s = ldX; for (let n = 0; n < m; n++) if (M.neuronFreq[n] === k) s += (cl.act[n] - co.act[n]) * wd[n]; out.push(s); }
  return out.map((v) => (v - ldX) / (ldC - ldX));
}
// Spot check against the page's code path.
{
  const PATCH = [{ heads: [0] }, { heads: [1] }, { heads: [2] }, { heads: [3] }, { heads: [0, 1, 2, 3] }, { mlp: true },
    ...[1, 12, 27].map((k) => ({ neurons: M.neuronFreq.map((f) => (f === k ? 1 : 0)) }))];
  let worst = 0;
  for (let t = 0; t < 200; t++) {
    const a = (t * 7) % p, b = (t * 13 + 5) % p, a2 = (a + 1 + (t % 58)) % p;
    const fast = restoredAll(a * p + b, a2 * p + b);
    PATCH.forEach((P, j) => { worst = Math.max(worst, Math.abs(M.patch([a, b], [a2, b], P).restored - fast[j])); });
  }
  check(`fast patching computation matches M.patch (largest difference ${worst.toExponential(1)})`, worst < 1e-9);
}
function sweep(which) {
  const R = COMPS.map(() => []);
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) for (let off = 1; off < p; off++) {
    const i = a * p + b, i2 = which === 'a' ? ((a + off) % p) * p + b : a * p + ((b + off) % p);
    restoredAll(i, i2).forEach((v, j) => R[j].push(v));
  }
  return R;
}
const R = sweep('a');
const N = R[0].length;
console.log(`\nPatching, all ${N.toLocaleString()} (clean pair, corrupted a) combinations: mean, 5th and 95th percentiles, min, max`);
COMPS.forEach((c, j) => console.log(`  ${c.padEnd(10)} ${pct(mean(R[j]))}  ${pct(quant(R[j], 0.05))} to ${pct(quant(R[j], 0.95))}  [${pct(min(R[j]))}, ${pct(max(R[j]))}]`));
const best = [0, 0, 0, 0]; let anyHalf = 0, headBeatsMlp = 0;
for (let t = 0; t < N; t++) {
  const hs = [0, 1, 2, 3].map((h) => R[h][t]); const k = hs.indexOf(Math.max(...hs)); best[k]++;
  if (Math.max(...hs) > 0.5) anyHalf++;
  if (Math.max(...hs) > R[5][t]) headBeatsMlp++;
}
console.log('  head that restores the most:', best.map((v, h) => `head ${h} ${pct(v / N)}`).join(', '));
console.log(`  some single head restores more than 50%: ${pct(anyHalf / N)}; some single head restores more than the MLP output: ${pct(headBeatsMlp / N)}`);
check('patching all four heads restores exactly 100% every time', R[4].every((v) => Math.abs(v - 1) < 1e-9));
check('patching the MLP output restores 86% to 105%', min(R[5]) > 0.86 && max(R[5]) < 1.05);
check('the three neuron groups add up to the MLP output', R[5].every((v, t) => Math.abs(v - (R[6][t] + R[7][t] + R[8][t])) < 1e-9));
check('mean restoration of each single head is between 15% and 35%', [0, 1, 2, 3].every((h) => mean(R[h]) > 0.15 && mean(R[h]) < 0.35));
check('every single head restores 80% or more for some (clean, corrupted) pair (check question 2)', [0, 1, 2, 3].every((h) => max(R[h]) >= 0.8));
const Rb = sweep('b');
console.log('Corrupting b instead: means', COMPS.map((c, j) => `${c} ${pct(mean(Rb[j]))}`).join(', '));

// ---- 4. The default pair on the page: clean 12 + 30, corrupted 20 + 30 ----
{
  const i = 12 * p + 30, i2 = 20 * p + 30, r = restoredAll(i, i2);
  console.log('\nDefault pair (12, 30) -> (20, 30):', COMPS.map((c, j) => `${c} ${pct(r[j])}`).join(', '));
  const cl = runs[i], co = runs[i2];
  console.log(`  logit difference: clean ${(cl.logits[42] - cl.logits[50]).toFixed(1)}, corrupted ${(co.logits[42] - co.logits[50]).toFixed(1)}`);
  // Head 2 over the 58 corrupted values of a, for this clean pair.
  const h2 = []; for (let off = 1; off < p; off++) h2.push(restoredAll(i, ((12 + off) % p) * p + 30)[2]);
  console.log(`  head 2 over the 58 corrupted values of a: ${pct(Math.min(...h2))} to ${pct(Math.max(...h2))}`);
  const h3 = []; for (let off = 1; off < p; off++) h3.push(restoredAll(i, ((12 + off) % p) * p + 30)[3]);
  console.log(`  head 3 over the 58 corrupted values of a: ${pct(Math.min(...h3))} to ${pct(Math.max(...h3))}`);
}
// ---- 5. The static figure's pair: 12 + 30, logits ----
{
  const L = runs[12 * p + 30].logits; const s = Array.from(L).map((v, c) => [v, c]).sort((x, y) => y[0] - x[0]);
  console.log(`figure: logits for 12 + 30: largest at c = ${s[0][1]} (${s[0][0].toFixed(1)}), next c = ${s[1][1]} (${s[1][0].toFixed(1)})`);
}
// Fit of the logits for 12 + 30 by a constant plus cos and sin of w_k (42 - c), k = 1, 12, 27 (least squares).
{
  const L = Array.from(runs[12 * p + 30].logits);
  const cols = [Array(p).fill(1)];
  for (const k of [1, 12, 27]) { cols.push(L.map((_, c) => Math.cos(2 * Math.PI * k * (42 - c) / p))); cols.push(L.map((_, c) => Math.sin(2 * Math.PI * k * (42 - c) / p))); }
  // The columns are orthogonal over c = 0..p-1, so each coefficient is a projection.
  const coef = cols.map((col) => col.reduce((s, v, c) => s + v * L[c], 0) / col.reduce((s, v) => s + v * v, 0));
  const fit = L.map((_, c) => cols.reduce((s, col, j) => s + coef[j] * col[c], 0));
  const mu = mean(L), ssT = L.reduce((s, v) => s + (v - mu) ** 2, 0), ssR = L.reduce((s, v, c) => s + (v - fit[c]) ** 2, 0);
  console.log(`figure: cos coefficients ${[1, 12, 27].map((k, i) => `k=${k}: ${coef[1 + 2 * i].toFixed(1)}`).join(', ')}; sin coefficients ${[1, 12, 27].map((k, i) => coef[2 + 2 * i].toFixed(2)).join(', ')}; share of variance explained ${pct(1 - ssR / ssT)}`);
  check('the logits for 12 + 30 are fit by the three cosines with R^2 above 95%', 1 - ssR / ssT > 0.95);
}
{
  const small = all.filter((k) => !keys.includes(k)).map((k) => max(M.circle(k).map(([x, y]) => Math.hypot(x, y))));
  check(`at the other frequencies the points stay within ${max(small).toFixed(2)} of the center (below 0.1)`, max(small) < 0.1);
}
console.log(failed ? `\n${failed} check(s) FAILED` : '\nall checks passed');
process.exit(failed ? 1 : 0);
