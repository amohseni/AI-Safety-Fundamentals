// Numbers quoted on page 6.4, recomputed with the page's own settings (docs/js/lib/s6-toymodel.js:
// Adam, batch 256, 4,000 steps, learning rate falling from 0.01 to 0.0005, best of 4 starting points).
// The tool's first run uses group 0; each press of "New start" moves to the next group.
// Run: node tools/test-page-6-4.mjs
import { trainBest, gaps2d } from '../docs/js/lib/s6-toymodel.js';

const GROUPS = 8;
const pattern = (t) => t.norms().map((v) => (v > 0.5 ? '#' : v > 0.2 ? '+' : '.')).join('');
const gaps = (t) => gaps2d(t).map(Math.round).join(',');

console.log('Stored features ("#": length above 0.5) and angles between neighbouring directions, groups 0 to 7');
function sweep(label, r, Ss) {
  for (const S of Ss) {
    const res = [];
    for (let g = 0; g < GROUPS; g++) { const t = trainBest({ n: 5, m: 2, S, r, group: g }); res.push(`${pattern(t)}[${gaps(t)}]`); }
    console.log(`${label} r=${r} S=${S}:`, res.join(' '));
  }
}
// Task 1: dense features, importance falling by 0.8 per feature.
sweep('task 1', 0.8, [0]);
// Task 2: equal importance, rising sparsity.
sweep('task 2', 1, [0.5, 0.7, 0.8, 0.9, 0.95]);
// Task 3: importance falling by 0.8 per feature, rising sparsity.
sweep('task 3', 0.8, [0.5, 0.7, 0.8, 0.9]);

// Task 4: in the pentagon, how many features have a weight of size at least 0.3 on each hidden coordinate?
// (The bars in the tool show these weights; a weight is the coordinate's value when that feature alone is 1.)
{
  const counts = [];
  let minTop4 = Infinity, maxSmallest = 0;
  for (let g = 0; g < GROUPS; g++) {
    const t = trainBest({ n: 5, m: 2, S: 0.9, r: 1, group: g });
    for (let k = 0; k < 2; k++) {
      const w = Array.from({ length: 5 }, (_, i) => Math.abs(t.W[k * 5 + i])).sort((a, b) => b - a);
      counts.push(w.filter((v) => v >= 0.3).length);
      minTop4 = Math.min(minTop4, w[3]);
      maxSmallest = Math.max(maxSmallest, w[4]);
    }
  }
  console.log('task 4: features with |weight| >= 0.3 on a coordinate, 8 groups x 2 coordinates:', counts.join(' '),
    `; fourth-largest |weight| is at least ${minTop4.toFixed(3)}; smallest |weight| is at most ${maxSmallest.toFixed(3)}`);
}

// Static figure: importance falling by 0.8, sparsity 0, 0.5, 0.8, 0.9, group 0. Printed as JSON for docs/js/page/6-4.js.
{
  const fig = {};
  for (const S of [0, 0.5, 0.8, 0.9]) { const t = trainBest({ n: 5, m: 2, S, r: 0.8, group: 0 }); fig[S] = Array.from(t.W, (v) => Math.round(v * 1000) / 1000); }
  console.log('figure W (row-major, 2 x 5):', JSON.stringify(fig));
}

// Biases: in the dense case a left-out feature's output is the constant ReLU(b_i); in the pentagon the biases are negative.
{
  const d = trainBest({ n: 5, m: 2, S: 0, r: 0.8, group: 0 });
  const p = trainBest({ n: 5, m: 2, S: 0.9, r: 1, group: 0 });
  console.log('biases, dense r=0.8:', Array.from(d.b, (v) => v.toFixed(2)).join(' '), '; pentagon:', Array.from(p.b, (v) => v.toFixed(2)).join(' '));
  console.log('W^T W pentagon (row 1):', p.gram()[0].map((v) => v.toFixed(2)).join(' '), '; lengths', p.norms().map((v) => v.toFixed(2)).join(' '));
}
