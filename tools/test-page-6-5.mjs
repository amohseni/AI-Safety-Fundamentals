// Numbers quoted on page 6.5, recomputed with the page's own settings: the pentagon toy model from page 6.4
// (5 features, 2 dimensions, S = 0.9, equal importance, group 0), and a sparse autoencoder trained with Adam,
// learning rate 0.005, batch 256, 3,000 steps. Starting point k of the tool is SAE seed k (k = 1, 2, ...).
// Run: node tools/test-page-6-5.mjs
import { trainBest, SAE } from '../docs/js/lib/s6-toymodel.js';

const toy = trainBest({ n: 5, m: 2, S: 0.9, r: 1, group: 0 });
console.log('source W (row-major, 2 x 5):', JSON.stringify(Array.from(toy.W, (v) => Math.round(v * 10000) / 10000)));
console.log('source b:', JSON.stringify(Array.from(toy.b, (v) => Math.round(v * 10000) / 10000)));
console.log('true features active per input on average: 5 x 0.1 = 0.5');

const SEEDS = 10, STEPS = 3000;
function run(size, lambda, seed) {
  const s = new SAE(toy, { size, lambda, seed, lr: 0.005 });
  for (let i = 0; i < STEPS; i++) s.step();
  const ev = s.evaluate();
  const rec = s.recovery(0.5, ev.alive);
  // Live units that point (cosine > 0.95) at a true feature that another live unit also points at.
  const owner = [];
  for (let k = 0; k < size; k++) {
    if (!ev.alive[k]) continue;
    const d = s.dir(k);
    let best = -1, arg = -1;
    for (let i = 0; i < 5; i++) { const w = toy.dir(i), L = Math.hypot(...w); const c = (w[0] * d[0] + w[1] * d[1]) / L; if (c > best) { best = c; arg = i; } }
    if (best > 0.95) owner.push(arg);
  }
  const dup = owner.length - new Set(owner).size;
  const misaligned = size - ev.dead - owner.length;
  return { recovered: rec.filter((r) => r.cos > 0.95).length, mean: rec.reduce((a, r) => a + r.cos, 0) / rec.length, l0: ev.l0, fvu: ev.fvu, dead: ev.dead, dup, misaligned };
}
const range = (a, k = 2) => (Math.min(...a) === Math.max(...a) ? Math.min(...a).toFixed(k) : `${Math.min(...a).toFixed(k)} to ${Math.max(...a).toFixed(k)}`);
function report(size, lambda) {
  const R = Array.from({ length: SEEDS }, (_, i) => run(size, lambda, i + 1));
  const all5 = R.filter((r) => r.recovered === 5).length;
  console.log(`size ${size}, penalty ${lambda}: recovered (cos > 0.95) ${range(R.map((r) => r.recovered), 0)}; all five in ${all5}/${SEEDS};`,
    `mean best cos ${range(R.map((r) => r.mean))}; active units ${range(R.map((r) => r.l0))}; rebuild error ${range(R.map((r) => r.fvu), 3)};`,
    `dead ${range(R.map((r) => r.dead), 0)}; duplicates ${range(R.map((r) => r.dup), 0)}; pointing elsewhere ${range(R.map((r) => r.misaligned), 0)}`);
}
// Task 1: penalty sweep at dictionary size 5.
for (const l of [0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1, 2]) report(5, l);
// Task 2: a dictionary smaller than the number of features.
for (const l of [0.3, 0.5, 1]) report(3, l);
// Task 3: doubling the dictionary.
for (const l of [0.3, 0.5]) report(10, l);
report(20, 0.5);

// Static figure: dictionary of 5 at penalty 0.5, starting point 1; one activation with features 1 and 3 active.
{
  const s = new SAE(toy, { size: 5, lambda: 0.5, seed: 1, lr: 0.005 });
  for (let i = 0; i < STEPS; i++) s.step();
  const x = [0.9, 0, 0.6, 0, 0];
  const h = toy.hidden(x);
  const f = s.encode(h);
  console.log('figure: h =', Array.from(h, (v) => v.toFixed(3)).join(', '), '; code f =', Array.from(f, (v) => v.toFixed(3)).join(', '));
  console.log('figure D (row-major, 2 x 5):', JSON.stringify(Array.from(s.D, (v) => Math.round(v * 10000) / 10000)), 'c:', JSON.stringify(Array.from(s.c, (v) => Math.round(v * 10000) / 10000)));
  console.log('figure: unit matched to each feature:', JSON.stringify(s.recovery(0.5).map((r) => [r.feature + 1, r.unit + 1, Number(r.cos.toFixed(3))])));
}

// Task 3: do duplicate units fire on the same inputs? For each pair of live units pointing at the same feature,
// the share of inputs (among those where either unit fires) on which only one of the two fires.
{
  const shares = [];
  for (let seed = 1; seed <= SEEDS; seed++) {
    const s = new SAE(toy, { size: 10, lambda: 0.5, seed, lr: 0.005 });
    for (let i = 0; i < STEPS; i++) s.step();
    const ev = s.evaluate();
    const owner = new Map();
    for (let k = 0; k < 10; k++) {
      if (!ev.alive[k]) continue;
      const d = s.dir(k);
      for (let i = 0; i < 5; i++) { const w = toy.dir(i), L = Math.hypot(...w); if ((w[0] * d[0] + w[1] * d[1]) / L > 0.95) owner.set(i, [...(owner.get(i) || []), k]); }
    }
    const H = s._data(rngFor(seed), 4000);
    for (const units of owner.values()) {
      for (let a = 0; a < units.length; a++) for (let b = a + 1; b < units.length; b++) {
        let either = 0, one = 0;
        for (let t = 0; t < 4000; t++) {
          const f = s.encode(H.subarray(2 * t, 2 * t + 2));
          const fa = f[units[a]] > 0.01, fb = f[units[b]] > 0.01;
          if (fa || fb) { either++; if (fa !== fb) one++; }
        }
        if (either) shares.push(one / either);
      }
    }
  }
  console.log(`task 3: duplicate pairs ${shares.length}; share of firing inputs where only one of the pair fires (activation > 0.01): ${range(shares)}`);
}
function rngFor(seed) { let a = (seed * 2654435761) >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
