// Claims on page 3.2, checked numerically on the vectors the page ships. Run: node tools/test-page-3-2.mjs
import fs from 'node:fs';
import path from 'node:path';
import { decodeVectors, nearest, row, analogy, pca2, mapNeighbors } from '../docs/js/lib/s3j-embed.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const doc = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/3-2/vectors.json')));
const E = decodeVectors(doc);
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const words = (r) => r.map((x) => x.word);
const nnOf = (w, k = 10) => nearest(E, row(E, E.index.get(w)), k, [E.index.get(w)]);

check('4,039 words, 300 numbers each', E.n === 4039 && E.d === 300, `${E.n} x ${E.d}`);
{
  let maxErr = 0;
  for (let i = 0; i < E.n; i += 97) { const r = row(E, i); maxErr = Math.max(maxErr, Math.abs(Math.hypot(...r) - 1)); }
  check('decoded vectors have unit length', maxErr < 1e-5, maxErr.toExponential(1));
}

// Task 1: neighbors of bank are all about money; river's neighbors do not include bank.
{
  const b = words(nnOf('bank'));
  const expect = ['banks', 'banking', 'Bank', 'branch', 'lending', 'deposit', 'mortgage', 'loans', 'accounts', 'loan'];
  check('bank: ten nearest neighbors as listed', JSON.stringify(b) === JSON.stringify(expect), b.join(', '));
  const r = words(nnOf('river'));
  check('river: first six neighbors as listed, no bank in ten', JSON.stringify(r.slice(0, 6)) === JSON.stringify(['River', 'water', 'sea', 'bridge', 'boat', 'Lake']) && !r.includes('bank'), r.join(', '));
}

// Task 2: king - man + woman.
{
  const off = analogy(E, 'king', 'man', 'woman', { exclude: false, k: 3 });
  check('exclusion off: king first (0.799), queen second (0.712)', off[0].word === 'king' && off[1].word === 'queen' && off[0].sim.toFixed(3) === '0.799' && off[1].sim.toFixed(3) === '0.712', off.map((x) => `${x.word} ${x.sim.toFixed(3)}`).join(', '));
  const on = analogy(E, 'king', 'man', 'woman', { exclude: true, k: 3 });
  check('exclusion on: queen first', on[0].word === 'queen', words(on).join(', '));
}

// Task 3: two of three capitals with the exclusion on; one with it off.
{
  const qs = [['Tokyo', 'Japan', 'France', 'Paris'], ['Paris', 'France', 'Italy', 'Rome'], ['Oslo', 'Norway', 'Austria', 'Vienna']];
  const on = qs.map(([a, b, c]) => analogy(E, a, b, c, { exclude: true, k: 2 }));
  const off = qs.map(([a, b, c]) => analogy(E, a, b, c, { exclude: false, k: 2 }));
  const okOn = on.filter((r, k) => r[0].word === qs[k][3]).length, okOff = off.filter((r, k) => r[0].word === qs[k][3]).length;
  check('exclusion on: two of three right', okOn === 2, on.map((r) => words(r).join('/')).join('; '));
  check('Paris - France + Italy: Milan 0.721 then Rome 0.704', on[1][0].word === 'Milan' && on[1][1].word === 'Rome' && on[1][0].sim.toFixed(3) === '0.721' && on[1][1].sim.toFixed(3) === '0.704', on[1].map((x) => x.sim.toFixed(3)).join(' '));
  check('exclusion off: only Tokyo - Japan + France right; Oslo analogy returns Austria', okOff === 1 && off[0][0].word === 'Paris' && off[2][0].word === 'Austria', off.map((r) => words(r).join('/')).join('; '));
}

// Task 4: neighbors kept on the maps.
{
  const i = E.index.get('king'), nn = nnOf('king').map((r) => r.i);
  const a = mapNeighbors(E.pca, i, nn), b = mapNeighbors(E.umap, i, nn);
  check('king: PCA keeps 0 of 10, UMAP keeps 4', a === 0 && b === 4, `PCA ${a}, UMAP ${b}`);
  let sp = 0, su = 0, N = 0;
  for (let j = 0; j < E.n; j += 7) { const n2 = nearest(E, row(E, j), 10, [j]).map((r) => r.i); sp += mapNeighbors(E.pca, j, n2); su += mapNeighbors(E.umap, j, n2); N++; }
  check('sample of 577 words: UMAP 3.3, PCA 0.3 on average', N === 577 && (su / N).toFixed(1) === '3.3' && (sp / N).toFixed(1) === '0.3', `UMAP ${(su / N).toFixed(2)}, PCA ${(sp / N).toFixed(2)} over ${N}`);
}

// Figure: country-to-capital arrows nearly parallel in the PCA of the twelve vectors.
{
  const pairs = [['France', 'Paris'], ['Germany', 'Berlin'], ['Poland', 'Warsaw'], ['Russia', 'Moscow'], ['Egypt', 'Cairo'], ['Japan', 'Tokyo']];
  const ids = pairs.flat().map((w) => E.index.get(w));
  const [c1, c2] = pca2(E, ids);
  const ang = pairs.map((_, p) => Math.atan2(c2[2 * p + 1] - c2[2 * p], c1[2 * p + 1] - c1[2 * p]) * 180 / Math.PI);
  const spread = Math.max(...ang) - Math.min(...ang);
  check('figure: arrows within 13 degrees of one another', spread < 13.5 && spread > 12.5, `angles ${ang.map((a) => a.toFixed(0)).join(', ')}; spread ${spread.toFixed(1)}`);
  // PCA by power iteration agrees with the variance it should capture (sanity: components orthogonal).
  const dot = c1.reduce((s, x, j) => s + x * c2[j], 0);
  check('figure: the two components are uncorrelated', Math.abs(dot) < 1e-6, dot.toExponential(1));
}

// Sidenote: 4,000 frequent words plus 39 added.
check('4,000 + 39 words', doc.words.length - 4000 === 39, `${doc.words.length - 4000} added`);

console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
