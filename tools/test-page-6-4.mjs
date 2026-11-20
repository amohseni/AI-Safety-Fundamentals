// Numbers quoted on page 6.4, recomputed with the page's own settings (docs/js/lib/s6-toymodel.js:
// Adam, batch 256, 4,000 steps, learning rate falling from 0.01 to 0.0005, best of 4 starting points).
// The tool's first run uses group 0; each press of "New start" moves to the next group.
// Run: node tools/test-page-6-4.mjs   (about a minute and a half)
import fs from 'node:fs';
import { trainBest, gaps2d } from '../docs/js/lib/s6-toymodel.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const SW = {}; // SW[`${r}|${S}`] = per group { pat, gaps, norms }

const GROUPS = 8;
const pattern = (t) => t.norms().map((v) => (v > 0.5 ? '#' : v > 0.2 ? '+' : '.')).join('');
const gaps = (t) => gaps2d(t).map(Math.round).join(',');

console.log('Stored features ("#": length above 0.5) and angles between neighbouring directions, groups 0 to 7');
function sweep(label, r, Ss) {
  for (const S of Ss) {
    const res = [];
    for (let g = 0; g < GROUPS; g++) { const t = trainBest({ n: 5, m: 2, S, r, group: g }); res.push(`${pattern(t)}[${gaps(t)}]`); (SW[`${r}|${S}`] ||= []).push({ pat: pattern(t), gaps: gaps2d(t), norms: t.norms() }); }
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
  // Task 4 notice: "Four or five. In all eight groups we tested, every coordinate had at least four features with components
  // of size at least 0.35, and one coordinate had a fifth above 0.3."
  check('t4: in all eight groups every coordinate has at least four features with components of size at least 0.35', minTop4 >= 0.35, minTop4);
  check('t4: one coordinate (of 16) has a fifth above 0.3', counts.filter((c) => c === 5).length === 1 && counts.every((c) => c >= 4), counts.join(' '));
  check('t4 theorem: cos 72 degrees is about 0.31', Math.cos((72 * Math.PI) / 180).toFixed(2) === '0.31', Math.cos((72 * Math.PI) / 180));
  console.log('task 4: features with |weight| >= 0.3 on a coordinate, 8 groups x 2 coordinates:', counts.join(' '),
    `; fourth-largest |weight| is at least ${minTop4.toFixed(3)}; smallest |weight| is at most ${maxSmallest.toFixed(3)}`);
}

// Static figure: importance falling by 0.8, sparsity 0, 0.5, 0.8, 0.9, group 0. Printed as JSON for docs/js/page/6-4.js.
{
  const fig = {};
  for (const S of [0, 0.5, 0.8, 0.9]) { const t = trainBest({ n: 5, m: 2, S, r: 0.8, group: 0 }); fig[S] = Array.from(t.W, (v) => Math.round(v * 1000) / 1000); }
  console.log('figure W (row-major, 2 x 5):', JSON.stringify(fig));
  // The page draws the figure from these numbers, copied into docs/js/page/6-4.js as FIG.
  const src = fs.readFileSync(new URL('../docs/js/page/6-4.js', import.meta.url), 'utf8');
  const FIG = Function(`return (${src.match(/const FIG = (\{[\s\S]*?\});/)[1]})`)();
  check('figure: the parameters stored in docs/js/page/6-4.js are the ones this script computes', [0, 0.5, 0.8, 0.9].every((S) => JSON.stringify(FIG[S]) === JSON.stringify(fig[S])), JSON.stringify(FIG[0.9]));
  // Caption: "At S = 0 the model stores features 1 and 2 at right angles. At S = 0.5 it adds feature 3, opposite feature 2.
  // At S = 0.8 it stores features 1 to 4 as two opposite pairs. At S = 0.9 it stores all five, spaced unevenly".
  const col = (S, i) => [fig[S][i], fig[S][5 + i]], len = (v) => Math.hypot(v[0], v[1]), cos = (u, v) => (u[0] * v[0] + u[1] * v[1]) / (len(u) * len(v));
  const stored = (S) => [0, 1, 2, 3, 4].filter((i) => len(col(S, i)) > 0.5).map((i) => i + 1).join('');
  check('caption: at S = 0 features 1 and 2 are stored, at right angles (within 2 degrees)', stored(0) === '12' && Math.abs(cos(col(0, 0), col(0, 1))) < 0.035, `${stored(0)} cos ${cos(col(0, 0), col(0, 1)).toFixed(3)}`);
  check('caption: at S = 0.5 the model adds feature 3, opposite feature 2', stored(0.5) === '123' && cos(col(0.5, 1), col(0.5, 2)) < -0.999, `${stored(0.5)} cos ${cos(col(0.5, 1), col(0.5, 2)).toFixed(4)}`);
  {
    const c = (i, j) => cos(col(0.8, i), col(0.8, j)), opp = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]].filter(([i, j]) => c(i, j) < -0.999);
    check('caption: at S = 0.8 features 1 to 4 are stored as two opposite pairs', stored(0.8) === '1234' && opp.length === 2 && new Set(opp.flat()).size === 4, `${stored(0.8)} pairs ${JSON.stringify(opp)}`);
  }
  {
    const ang = [0, 1, 2, 3, 4].map((i) => Math.atan2(col(0.9, i)[1], col(0.9, i)[0]) * 180 / Math.PI).sort((a, b) => a - b);
    const g = ang.map((a, i) => (i < 4 ? ang[i + 1] - a : ang[0] + 360 - a));
    check('caption: at S = 0.9 all five are stored, spaced unevenly (largest gap at least 5 degrees wider than the smallest)', stored(0.9) === '12345' && Math.max(...g) - Math.min(...g) >= 5, g.map((v) => v.toFixed(0)).join(' '));
  }
}

// Biases: in the dense case a left-out feature's output is the constant ReLU(b_i); in the pentagon the biases are negative.
{
  const d = trainBest({ n: 5, m: 2, S: 0, r: 0.8, group: 0 });
  const p = trainBest({ n: 5, m: 2, S: 0.9, r: 1, group: 0 });
  console.log('biases, dense r=0.8:', Array.from(d.b, (v) => v.toFixed(2)).join(' '), '; pentagon:', Array.from(p.b, (v) => v.toFixed(2)).join(' '));
  // Task 1 notice: "For a left-out feature the model outputs a constant, ReLU(b_i), and training sets b_i near 0.5".
  check('t1: the biases of the three left-out features are near 0.5 (within 0.03)', [2, 3, 4].every((i) => Math.abs(d.b[i] - 0.5) <= 0.03), Array.from(d.b, (v) => v.toFixed(3)).join(' '));
  // Argument: "In the pentagon we trained, neighboring directions have dot product 0.37, the others have dot product -0.98,
  // and every bias is -0.21. ... the negative bias removes the positive interference whenever x_j is below 0.56."
  const G = p.gram(), pos = [], neg = [];
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) (G[i][j] > 0 ? pos : neg).push(G[i][j]);
  const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  console.log('pentagon dot products: positive', pos.map((v) => v.toFixed(3)).join(' '), '; negative', neg.map((v) => v.toFixed(3)).join(' '));
  // The five pairs of each kind differ a little (0.369 to 0.381 and -0.988 to -0.978); the page quotes one number for each kind,
  // so the check is on their average, with every pair within 0.015 of the quoted number.
  check('pentagon: neighboring directions have dot product 0.37 (average of the five pairs; each within 0.015)', pos.length === 5 && avg(pos).toFixed(2) === '0.37' && pos.every((v) => Math.abs(v - 0.37) <= 0.015), `${avg(pos)}: ${pos.map((v) => v.toFixed(3)).join(' ')}`);
  check('pentagon: the others have dot product -0.98 (average of the five pairs; each within 0.015)', neg.length === 5 && avg(neg).toFixed(2) === '-0.98' && neg.every((v) => Math.abs(v + 0.98) <= 0.015), `${avg(neg)}: ${neg.map((v) => v.toFixed(3)).join(' ')}`);
  check('pentagon: every bias is -0.21', Array.from(p.b).every((v) => v.toFixed(2) === '-0.21'), Array.from(p.b, (v) => v.toFixed(3)).join(' '));
  // FAILS as of 2026-11-19: the trained parameters give 0.56 (0.2114 / 0.3748 = 0.564). See AUDIT-TESTS-FINDINGS.md.
  check('pentagon: the bias removes the positive interference whenever x_j is below 0.56 (average bias / average neighboring dot product, from the trained parameters)', (-avg(Array.from(p.b)) / avg(pos)).toFixed(2) === '0.56', -avg(Array.from(p.b)) / avg(pos));
  check('at S = 0.9 two given features are both active 1% of the time', Math.abs((1 - 0.9) ** 2 - 0.01) < 1e-12, (1 - 0.9) ** 2);
  console.log('W^T W pentagon (row 1):', p.gram()[0].map((v) => v.toFixed(2)).join(' '), '; lengths', p.norms().map((v) => v.toFixed(2)).join(' '));
}
// ---- Claims about the sweeps ----
const sw = (r, S) => SW[`${r}|${S}`], n = (r, S, f) => sw(r, S).filter(f).length;
const right = (g) => g.gaps.every((v) => Math.abs(v - 90) <= 1.5 || Math.abs(v - 270) <= 1.5 || Math.abs(v - 180) <= 1.5);
const pairs = (g) => g.pat.replace(/[^#]/g, '').length === 4 && g.gaps.length === 4 && g.gaps.every((v) => Math.abs(v - 90) <= 1.5);
const pent = (g) => g.pat === '#####' && g.gaps.every((v) => Math.abs(v - 72) <= 1.5);
// "In practice a stored feature's direction has length near 1, and a feature that is left out has length near 0."
{
  const all = Object.values(SW).flat().flatMap((g) => g.norms), st = all.filter((v) => v > 0.5), out = all.filter((v) => v <= 0.5);
  check('stored directions have length near 1 (0.7 to 1.3), left-out ones near 0 (below 0.1), in every run of the sweeps', st.every((v) => v >= 0.7 && v <= 1.3) && out.every((v) => v < 0.1), `stored ${Math.min(...st).toFixed(2)} to ${Math.max(...st).toFixed(2)}; left out at most ${Math.max(...out).toFixed(3)}`);
}
// Task 1 notice: "Two: features 1 and 2, the two most important, at right angles to each other. In all eight groups of starting
// points we tested, the other three directions end with length near zero."
check('t1: S = 0, falloff 0.8: features 1 and 2 stored, at right angles, the other three near zero, in all eight groups', n(0.8, 0, (g) => g.pat === '##...' && right(g)) === 8, sw(0.8, 0).map((g) => g.pat).join(' '));
// Task 2 notice.
check('t2: falloff 1: a regular pentagon, 72 degrees between neighbors, in all eight groups at S = 0.9 and at 0.95', n(1, 0.9, pent) === 8 && n(1, 0.95, pent) === 8, `${n(1, 0.9, pent)} ${n(1, 0.95, pent)}`);
check('t2: at S = 0.5 and 0.7 four features as two opposite pairs, 90 degrees between neighbors (eight of eight groups)', n(1, 0.5, pairs) === 8 && n(1, 0.7, pairs) === 8, `${n(1, 0.5, pairs)} ${n(1, 0.7, pairs)}`);
check('t2: at S = 0.8 the pentagon in three groups and the two pairs in five', n(1, 0.8, pent) === 3 && n(1, 0.8, pairs) === 5, `${n(1, 0.8, pent)} ${n(1, 0.8, pairs)}`);
// Task 3 notice: "features 1 and 2 at S = 0 (8 of 8), features 1 to 3 at S = 0.5 (7 of 8), features 1 to 4 at S = 0.7 (7 of 8),
// and all five at S = 0.9 (7 of 8)."
const pat = (r, S, p) => n(r, S, (g) => g.pat === p);
check('t3: falloff 0.8: features 1 and 2 at S = 0 (8 of 8)', pat(0.8, 0, '##...') === 8, pat(0.8, 0, '##...'));
check('t3: features 1 to 3 at S = 0.5 (7 of 8)', pat(0.8, 0.5, '###..') === 7, pat(0.8, 0.5, '###..'));
check('t3: features 1 to 4 at S = 0.7 (7 of 8)', pat(0.8, 0.7, '####.') === 7, pat(0.8, 0.7, '####.'));
check('t3: all five at S = 0.9 (7 of 8)', pat(0.8, 0.9, '#####') === 7, pat(0.8, 0.9, '#####'));
check("t3: the tool's first run (group 0) shows this order: 2, 3, 4, 5 features at S = 0, 0.5, 0.7, 0.9", [0, 0.5, 0.7, 0.9].map((S) => sw(0.8, S)[0].pat).join(' ') === '##... ###.. ####. #####', [0, 0.5, 0.7, 0.9].map((S) => sw(0.8, S)[0].pat).join(' '));
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
