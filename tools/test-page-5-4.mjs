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
  console.log(`wd ${wd} frac ${frac}: train>=99% at ${first(c, 'train_acc', 0.99)}, test>=50% at ${first(c, 'test_acc', 0.5)}, test>=99% at ${first(c, 'test_acc', 0.99)}; final test ${pct(c.test_acc.at(-1))}; ` +
    `test at step 119: ${pct(atStep(c, 'test_acc', 119))}; reversed-pair test answers right ${pct(revOk / rev)}, others ${pct(othOk / oth)}, share of correct test answers that are reversed pairs ${pct(revOk / (revOk + othOk))}`);
}

const main = load('run-wd1.0-f0.3.json');
for (const s of [0, 1000, 3600, 20000]) {
  const j = main.ckpt.step.indexOf(s), spec = main.ckpt.spectrum[j];
  const top = spec.map((v, k) => [k + 1, v]).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, v]) => `${k}:${v.toFixed(2)}`);
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
