// Numbers quoted on page 5.4 (and in panel 2 of page 5.5), read from the exported runs in docs/data/modadd.
// Run: node tools/test-page-5-4.mjs
import fs from 'node:fs';
import path from 'node:path';

const dir = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../docs/data/modadd');
const load = (f) => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
const bitsOf = (b64, n) => { const B = Buffer.from(b64, 'base64'); return Array.from({ length: n }, (_, i) => (B[i >> 3] >> (7 - (i & 7))) & 1); };
const first = (c, key, thr) => { const i = c[key].findIndex((v) => v >= thr); return i < 0 ? null : c.step[i]; };
const atStep = (c, key, s) => { let j = 0; while (j + 1 < c.step.length && c.step[j + 1] <= s) j++; return c[key][j]; };
const share = (spec, n = 5) => { const sq = spec.map((v) => v * v).sort((a, b) => b - a); return sq.slice(0, n).reduce((s, v) => s + v, 0) / sq.reduce((s, v) => s + v, 0); };
const pct = (v) => (v * 100).toFixed(1) + '%';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const R = {}, M = {}, r0 = (v) => Math.round(v * 100);
const revSplit = (r, j) => { // of the test pairs answered correctly at checkpoint j, the share whose reverse is a training pair
  const p = r.p, mask = bitsOf(r.train_mask, p * p), ok = bitsOf(r.ckpt.correct[j], p * p); let rev = 0, oth = 0;
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) { const i = a * p + b; if (!mask[i] && ok[i]) { if (mask[b * p + a]) rev++; else oth++; } }
  return rev / (rev + oth);
};

for (const wd of [1.0, 0.3, 0.0]) for (const frac of [0.3, 0.5, 0.7]) {
  const r = load(`run-wd${wd.toFixed(1)}-f${frac.toFixed(1)}.json`), c = r.curve, p = r.p;
  const mask = bitsOf(r.train_mask, p * p);
  const last = bitsOf(r.ckpt.correct.at(-1), p * p);
  // Test pairs whose reverse (b, a) is a training pair, and how the correct test answers split.
  let rev = 0, revOk = 0, oth = 0, othOk = 0;
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
    const i = a * p + b;
    if (mask[i]) continue;
    if (mask[b * p + a]) { rev++; revOk += last[i]; } else { oth++; othOk += last[i]; }
  }
  R[`${wd}|${frac}`] = { r, c, train99: first(c, 'train_acc', 0.99), test50: first(c, 'test_acc', 0.5), test99: first(c, 'test_acc', 0.99), final: c.test_acc.at(-1), at119: atStep(c, 'test_acc', 119), revAcc: revOk / rev, othOk, revShare: revOk / (revOk + othOk) };
  console.log(`wd ${wd} frac ${frac}: train>=99% at ${first(c, 'train_acc', 0.99)}, test>=50% at ${first(c, 'test_acc', 0.5)}, test>=99% at ${first(c, 'test_acc', 0.99)}; final test ${pct(c.test_acc.at(-1))}; ` +
    `test at step 119: ${pct(atStep(c, 'test_acc', 119))}; reversed-pair test answers right ${pct(revOk / rev)}, others ${pct(othOk / oth)}, share of correct test answers that are reversed pairs ${pct(revOk / (revOk + othOk))}`);
}

const main = load('run-wd1.0-f0.3.json');
for (const s of [0, 1000, 3600, 20000]) {
  const j = main.ckpt.step.indexOf(s), spec = main.ckpt.spectrum[j];
  const top = spec.map((v, k) => [k + 1, v]).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, v]) => `${k}:${v.toFixed(2)}`);
  M[s] = { test: atStep(main.curve, 'test_acc', s), restricted: main.ckpt.restricted_test_acc[j], share: share(spec), order: spec.map((v, k) => [k + 1, v]).sort((a, b) => b[1] - a[1]) };
  console.log(`main run step ${s}: test ${pct(atStep(main.curve, 'test_acc', s))}, restricted test ${pct(main.ckpt.restricted_test_acc[j])}, top-5 share ${pct(share(spec))}, largest ${top.join(' ')}`);
}
console.log('main run key frequencies', main.key_freqs_final.join(' '));

const seeds = load('seeds.json').seeds;
for (const s of seeds) {
  console.log(`seed ${s.seed}: test ${pct(s.test_acc)}, keys ${s.key_freqs.join(' ')}, keep only keys ${pct(s.acc_keep_only_keys)}, remove all keys ${pct(s.acc_remove_all_keys)}, remove each key ${s.key_freqs.map((k) => pct(s.acc_remove_one[k])).join(' ')}`);
}
const sets = seeds.map((s) => s.key_freqs.join(','));
console.log('all key-frequency sets distinct:', new Set(sets).size === sets.length);
const count = {};
seeds.forEach((s) => s.key_freqs.forEach((k) => { count[k] = (count[k] || 0) + 1; }));
console.log('frequencies used by more than one seed:', Object.entries(count).filter(([, n]) => n > 1).map(([k, n]) => `${k} (${n})`).join(', '));
console.log('chance level 1/59 =', pct(1 / 59));
// ---- Page 5.4 ----
const run = (wd, frac) => R[`${wd}|${frac}`];
check('the runs use p = 59, embeddings of 128 numbers, 4 heads, an MLP of 512 units, learning rate 0.001, 20,000 steps', Object.values(R).every(({ r }) => r.p === 59 && r.d_model === 128 && r.heads === 4 && r.d_mlp === 512 && r.lr === 0.001 && r.steps_total === 20000 && r.curve.step.at(-1) === 20000), 'nine runs');
// Task 1 notice.
{
  const z = run(0, 0.3), i = z.c.step.indexOf(100);
  check('t1: with weight decay 0 (30%) the model answers every training pair correctly from step 100 on', i > 0 && z.c.train_acc.slice(i).every((v) => v === 1) && z.c.train_acc[i - 1] < 1, z.c.train_acc[i - 1]);
  check('t1: at step 20,000 it answers only 4% of the test pairs', r0(z.final) === 4, z.final);
  check('t1: it gets 15% of the reversed training pairs right and none of the others', r0(z.revAcc) === 15 && z.othOk === 0, `${z.revAcc} ${z.othOk}`);
  check('t1: with 50% and 70% and no weight decay, final test accuracy is 42% and 70%', r0(run(0, 0.5).final) === 42 && r0(run(0, 0.7).final) === 70, `${run(0, 0.5).final} ${run(0, 0.7).final}`);
  check('t1: more than 98% of the correctly answered test pairs are reversed training pairs (30%, 50%, 70%)', [0.3, 0.5, 0.7].every((f) => run(0, f).revShare > 0.98), [0.3, 0.5, 0.7].map((f) => pct(run(0, f).revShare)).join(' '));
  check('t1: with weight decay 1, test accuracy passes 99% by step 5,800', run(1, 0.3).test99 === 5800, run(1, 0.3).test99);
}
// Task 2 notice.
check('t2: with weight decay 1, test accuracy passes 99% at step 1,200 (70%), 1,850 (50%), 5,800 (30%)', run(1, 0.7).test99 === 1200 && run(1, 0.5).test99 === 1850 && run(1, 0.3).test99 === 5800, `${run(1, 0.7).test99} ${run(1, 0.5).test99} ${run(1, 0.3).test99}`);
{
  const z = run(1, 0.7), i = z.c.step.findIndex((s) => s > 120), jumped = z.c.test_acc.slice(0, i).some((v) => r0(v) === 69);
  check('t2: with 70%, test accuracy jumps to 69% within the first 120 steps', jumped && r0(z.at119) === 69, z.at119);
  let j = 0; while (j + 1 < z.r.ckpt.step.length && z.r.ckpt.step[j + 1] <= 120) j++;
  check(`t2: nearly all of those correct test pairs are reversed training pairs (more than 98%, at the stored step ${z.r.ckpt.step[j]})`, z.r.ckpt.step[j] >= 100 && revSplit(z.r, j) > 0.98, pct(revSplit(z.r, j)));
}
check('t2: with weight decay 0.3 the rise comes later in every case (99% later than with weight decay 1, or never)', [0.3, 0.5, 0.7].every((f) => run(0.3, f).test99 === null || run(0.3, f).test99 > run(1, f).test99), [0.3, 0.5, 0.7].map((f) => run(0.3, f).test99).join(' '));
check('t2: with weight decay 0.3 and 30%, test accuracy has reached only 88% by step 20,000', r0(run(0.3, 0.3).final) === 88, run(0.3, 0.3).final);
// Task 3 prompt and notice.
check('t3: at step 3,600 test accuracy is 19%, still below 20%', r0(M[3600].test) === 19 && M[3600].test < 0.2, M[3600].test);
check('t3: the five largest frequencies hold 47% of the embedding at step 3,600, against 20% at step 0', r0(M[3600].share) === 47 && r0(M[0].share) === 20, `${M[3600].share} ${M[0].share}`);
check('t3: at step 3,600 four frequencies, 12, 27, 2, and 1, stand above the rest (the fifth is at most 3/4 of the fourth)', M[3600].order.slice(0, 4).map((x) => x[0]).join(' ') === '12 27 2 1' && M[3600].order[4][1] < 0.75 * M[3600].order[3][1], M[3600].order.slice(0, 5).map(([k, v]) => `${k}:${v.toFixed(2)}`).join(' '));
check('t3: by step 20,000 the five largest hold 99%, nearly all in frequencies 1, 12, 27; frequency 2 has faded', r0(M[20000].share) === 99 && M[20000].order.slice(0, 3).map((x) => x[0]).sort((a, b) => a - b).join(' ') === '1 12 27' && share(main.ckpt.spectrum.at(-1), 3) > 0.95 && M[20000].order.find((x) => x[0] === 2)[1] < 0.2 * M[3600].order.find((x) => x[0] === 2)[1], `${M[20000].share}; top-3 share ${share(main.ckpt.spectrum.at(-1), 3)}`);
// After the tasks: the restricted part, and check 1 feedback.
check('the final key frequencies of the main run are 1, 12, and 27', main.key_freqs_final.join(' ') === '1 12 27', main.key_freqs_final.join(' '));
check('at step 3,600 the key-frequency part alone would answer 78% of the test pairs (full model 19%)', r0(M[3600].restricted) === 78, M[3600].restricted);
check('at step 1,000 it already answers 27%, against the full model 7%', r0(M[1000].restricted) === 27 && r0(M[1000].test) === 7, `${M[1000].restricted} ${M[1000].test}`);
// Caption.
{
  const z = run(1, 0.3), i = z.c.step.indexOf(100);
  check('caption: main run accuracy on the training pairs reaches 100% at step 100', z.c.train_acc[i] === 1 && z.c.train_acc.slice(0, i).every((v) => v < 1), z.c.train_acc[i]);
  check('caption: accuracy on the test pairs passes 50% at step 4,650 and 99% at step 5,800', z.test50 === 4650 && z.test99 === 5800, `${z.test50} ${z.test99}`);
  const rj = main.ckpt.restricted_test_acc.findIndex((v) => v >= 0.5);
  check('caption: the key-frequency part of the output rises thousands of steps earlier (passes 50% at least 1,000 steps before the full model)', main.ckpt.step[rj] <= z.test50 - 1000, main.ckpt.step[rj]);
}
// ---- Page 5.5, panel 2 and task 3 ----
{
  const S = load('seeds.json');
  check('5.5: five seeds, (a + b) mod 59, 30% of the pairs, weight decay 1, 20,000 steps; key frequencies hold at least 90%', seeds.length === 5 && S.p === 59 && S.frac === 0.3 && S.wd === 1 && S.key_share === 0.9 && seeds.every((s) => s.steps_total === 20000), seeds.length);
  check('5.5: all five runs answer the test pairs correctly', seeds.every((s) => s.test_acc > 0.999), seeds.map((s) => s.test_acc).join(' '));
  check('5.5: the five runs use frequencies 1, 12, 27; 4, 5, 15; 2, 5, 21; 15, 20, 21; and 2, 21, 22, 27', sets.join('; ') === '1,12,27; 4,5,15; 2,5,21; 15,20,21; 2,21,22,27', sets.join('; '));
  check('5.5: no two runs use the same set', new Set(sets).size === sets.length, new Set(sets).size);
  const rec = Object.entries(count).filter(([, n]) => n > 1).map(([k, n]) => `${k}:${n}`).join(' ');
  check('5.5: 21 recurs in three runs, and 2, 5, 15, and 27 in two each', rec === '2:2 5:2 15:2 21:3 27:2', rec);
  // Compared at the one decimal the page quotes: the lowest stored value is 0.9988, which the page's readout shows as 99.9%.
  check('5.5: keeping only the key frequencies leaves test accuracy at 99.9% or more in every run (to one decimal)', seeds.every((s) => Number((s.acc_keep_only_keys * 100).toFixed(1)) >= 99.9), seeds.map((s) => s.acc_keep_only_keys).join(' '));
  check('5.5: removing them drops it to about 2% (1.5% to 2.5%), the rate of guessing one answer in 59', seeds.every((s) => s.acc_remove_all_keys >= 0.015 && s.acc_remove_all_keys <= 0.025) && r0(1 / 59) === 2, seeds.map((s) => s.acc_remove_all_keys).join(' '));
}
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
