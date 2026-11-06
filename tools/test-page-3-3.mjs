// Claims on page 3.3, checked against the stored GPT-2 distributions. Run: node tools/test-page-3-3.mjs
import fs from 'node:fs';
import path from 'node:path';
import { decodeTree, distribution, storedMass, draw } from '../docs/js/lib/s3j-sample.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const doc = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/3-3/tree.json')));
const T = doc.temps;
const trees = doc.prompts.map((p) => decodeTree(p.tree, doc.k, T.length));
const word = (id) => doc.vocab[String(id)];
const P = (name) => doc.prompts.findIndex((p) => p.text === name);
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const top = (pi, temp, k = 3, node = 0) => distribution(trees[pi], node, T, temp, 1).rows.slice(0, k).map((r) => `${JSON.stringify(word(r.id))} ${r.pT.toFixed(3)}`).join(', ');

// The temperature formula: probabilities at T from the stored offsets agree with softmax of the stored
// log probabilities when the 20 tokens hold almost all the probability ("Thank you very": 0.992 on " much").
{
  const tr = trees[P('Thank you very')];
  const lp = Array.from({ length: doc.k }, (_, j) => -tr.nlp[j] / 1000);
  for (const temp of [0.5, 2]) {
    const d = distribution(tr, 0, T, temp, 1).rows;
    // ratio of two stored tokens' probabilities must be exp((lp_i - lp_j) / T) exactly
    const ratio = d[0].pT / d[1].pT, want = Math.exp((lp[0] - lp[1]) / temp);
    check(`ratio of top two at T = ${temp} is exp(gap / T)`, Math.abs(ratio / want - 1) < 1e-9, `${ratio.toFixed(4)} vs ${want.toFixed(4)}`);
  }
  const s1 = distribution(tr, 0, T, 1, 1).rows.reduce((s, r) => s + r.pT, 0);
  check('probabilities at T = 1 from offsets agree with the stored log probabilities', Math.abs(distribution(tr, 0, T, 1, 1).rows[0].pT - Math.exp(lp[0])) < 2e-3, s1.toFixed(4));
}

// Task 1: at T = 0 the draw is always the most likely token; repeated runs give the same text.
for (const [pi, p] of doc.prompts.entries()) {
  const tr = trees[pi];
  const runs = [0, 1, 2].map((seed) => {
    let node = 0, text = '', u = seed * 0.37 + 0.11;
    for (;;) {
      const d = distribution(tr, node, T, 0, 1);
      const r = draw(d, u); u = (u * 7.13) % 1;
      if (!r) break;
      text += word(r.id); node = tr.children[node].get(r.id);
    }
    return text;
  });
  check(`${p.text}: T = 0 gives the same text every time (${runs[0].length} chars)`, runs.every((t) => t === runs[0]), JSON.stringify(runs[0]));
  check(`${p.text}: greedy path in the tree matches generate() greedy`, p.greedy.startsWith(runs[0]), JSON.stringify(p.greedy.slice(0, 60)));
}

// Task 3: first-token certainty.
for (const name of ['The capital of France is', 'Thank you very', 'My favorite food is', 'Once upon a time', '1, 2, 3, 4,']) console.log('     ', name, '->', top(P(name), 1, 5));

// Task 4: top-p 0.5 after "Once upon a time".
{
  const pi = P('Once upon a time');
  for (const temp of [1, 2]) {
    const d = distribution(trees[pi], 0, T, temp, 0.5);
    console.log(`      Once upon a time, T = ${temp}, top-p 0.5: kept ${d.kept}${d.nucleusInside ? '' : ' (and beyond the top 20)'}`);
  }
}

// Figure: top tokens after "The capital of France is" at T = 0.5, 1, 2.
for (const temp of [0.5, 1, 2]) console.log(`      France, T = ${temp}:`, top(P('The capital of France is'), temp, 5));

// Stored mass at the root and along the T = 1 path for each prompt, and the tree sizes.
for (const [pi, p] of doc.prompts.entries()) {
  const tr = trees[pi];
  const d = distribution(tr, 0, T, 1, 1);
  const depths = [];
  for (let s = 0; s < 400; s++) {
    let node = 0, u = (s * 0.6180339887) % 1, k = 0;
    for (;;) { const r = draw(distribution(tr, node, T, 1, 1), u); u = (u * 9.7 + 0.31) % 1; if (!r) break; node = tr.children[node].get(r.id); k++; }
    depths.push(k);
  }
  depths.sort((a, b) => a - b);
  console.log(`      ${p.text}: ${tr.n} nodes; stored mass at root (T=1) ${storedMass(d).toFixed(3)}; generated length at T=1: median ${depths[200]}, min ${depths[0]}, max ${depths[399]}`);
}

// Exact numbers stated on the page.
{
  const d = (name, temp, p = 1) => distribution(trees[P(name)], 0, T, temp, p);
  const f = d('The capital of France is', 1).rows;
  check('France: the 0.085, Paris 0.032, France 0.032', f[0].pT.toFixed(3) === '0.085' && word(f[0].id) === ' the' && f.some((r) => word(r.id) === ' Paris' && r.pT.toFixed(3) === '0.032') && f.some((r) => word(r.id) === ' France' && r.pT.toFixed(3) === '0.032'), f.slice(0, 5).map((r) => r.pT.toFixed(3)).join(' '));
  const t = d('Thank you very', 1).rows[0];
  check('Thank you very: much 0.992', word(t.id) === ' much' && t.pT.toFixed(3) === '0.992', t.pT.toFixed(4));
  const held = (name, temp) => d(name, temp).rows.reduce((s, r) => s + r.pT, 0);
  check('favorite food: top 20 hold 29%', Math.round(held('My favorite food is', 1) * 100) === 29, held('My favorite food is', 1).toFixed(3));
  check('Once upon a time: top 20 hold 79% at T=1, 22% at T=1.5', Math.round(held('Once upon a time', 1) * 100) === 79 && Math.round(held('Once upon a time', 1.5) * 100) === 22, `${held('Once upon a time', 1).toFixed(3)} ${held('Once upon a time', 1.5).toFixed(3)}`);
  const o = d('Once upon a time', 1, 0.5);
  check('Once upon a time, top-p 0.5: 3 kept (, 0.427, the 0.065, I 0.041), first two 0.491', o.kept === 3 && o.rows[0].pT.toFixed(3) === '0.427' && o.rows[1].pT.toFixed(3) === '0.065' && o.rows[2].pT.toFixed(3) === '0.041' && (o.rows[0].pT + o.rows[1].pT).toFixed(3) === '0.491', `${o.kept}`);
  const o2 = d('Once upon a time', 2, 0.5);
  check('at T = 2, top-p 0.5 keeps all 20 and more', o2.kept === 20 && !o2.nucleusInside, `${o2.kept} ${o2.nucleusInside}`);
  const fr = [0.5, 1, 2].map((temp) => d('The capital of France is', temp).rows[0].pT.toFixed(3));
  check('figure: the at 0.397, 0.085, 0.005', fr.join() === '0.397,0.085,0.005', fr.join());
  const ord = [0.5, 1, 2].map((temp) => d('The capital of France is', temp).rows.slice(0, 10).map((r) => r.id).join());
  check('figure: order identical at the three temperatures', ord[0] === ord[1] && ord[1] === ord[2], 'same order');
  const g = doc.prompts[P('My favorite food is')].greedy;
  check('greedy loop text', g.startsWith(' the chicken. I love the chicken. I love the chicken.'), JSON.stringify(g.slice(0, 50)));
  const s15 = doc.prompts[P('Once upon a time')].samples['1.5|1.0'];
  check('T = 1.5 example quoted', s15.some((x) => x.startsWith(' Marty stood consulting here where Tam today had conductedphotograph orb')), JSON.stringify(s15[0].slice(0, 60)));
  // Generated length at T = 1 (median over 400 runs): 3 to 9 by prompt; T = 0 stops after 24.
}

console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
