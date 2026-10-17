// Numbers quoted on page 4.2. Run: node tools/test-page-4-2.mjs
// The reader's labels are stood in for by a "careful" labeler who always picks the correct answer
// when the two differ in correctness, and otherwise picks at random (20 tie-break seeds).
import { readerPairs, biasedLabels, fitBT, score, prefProb, compose } from '../docs/js/lib/s45-prefs.js';
import { rng } from '../docs/js/lib/mlp.js';
const pairs = readerPairs();
console.log(`pairs that differ in correctness: ${pairs.filter((p) => p.a.correct !== p.b.correct).length} of 20; in length: ${pairs.filter((p) => p.a.long !== p.b.long).length}`);
const lcw = { long: 1, correct: 0, confident: 1, flattering: 0 }, shc = { long: 0, correct: 1, confident: 0, flattering: 0 };
const rows = [];
for (let s = 1; s <= 20; s++) {
  const r = rng(s);
  const careful = pairs.map((p) => ({ ...p, aWins: p.a.correct !== p.b.correct ? p.a.correct === 1 : r() < 0.5 }));
  const w0 = fitBT(careful);
  const w1 = fitBT(careful.concat(biasedLabels(20, 1001)));
  const w2 = fitBT(careful.concat(biasedLabels(20, 1001), biasedLabels(20, 1002)));
  rows.push({ w0, w1, w2, d0: score(w0, lcw) - score(w0, shc), d1: score(w1, lcw) - score(w1, shc), d2: score(w2, lcw) - score(w2, shc) });
}
const rng2 = (a) => `${Math.min(...a).toFixed(2)} to ${Math.max(...a).toFixed(2)}`;
console.log(`reader only: correctness weight ${rng2(rows.map((r) => r.w0[1]))}; largest |other weight| ${Math.max(...rows.map((r) => Math.max(Math.abs(r.w0[0]), Math.abs(r.w0[2]), Math.abs(r.w0[3])))).toFixed(2)}`);
console.log(`+20 biased: length ${rng2(rows.map((r) => r.w1[0]))}; correctness ${rng2(rows.map((r) => r.w1[1]))}`);
console.log(`+40 biased: length ${rng2(rows.map((r) => r.w2[0]))}; correctness ${rng2(rows.map((r) => r.w2[1]))}`);
console.log(`score(long, confident, wrong) - score(short, hedged, correct): reader only ${rng2(rows.map((r) => r.d0))}; +20 ${rng2(rows.map((r) => r.d1))}; +40 ${rng2(rows.map((r) => r.d2))}`);
console.log(`long-confident-wrong preferred after +40 in ${rows.filter((r) => r.d2 > 0).length} of 20 cases`);
// The biased labeler's choices: fraction picking the longer answer.
const b = biasedLabels(2000, 5);
console.log(`biased labeler picks the longer answer in ${(b.filter((l) => (l.aWins ? l.a.long : l.b.long) === 1).length / 20).toFixed(1)}% of 2,000 simulated pairs`);
// Shifting all scores leaves predictions unchanged (Bradley-Terry depends only on differences).
console.log(`sigma(1) = ${(1 / (1 + Math.exp(-1))).toFixed(3)}, sigma(3) = ${(1 / (1 + Math.exp(-3))).toFixed(3)}`);
console.log('example:', compose(0, lcw), '|', compose(0, shc));
