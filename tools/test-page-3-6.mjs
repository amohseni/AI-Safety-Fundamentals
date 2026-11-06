// Claims on page 3.6, checked against the precomputed logit-lens data (tools/3-6/lens.py).
// Run: node tools/test-page-3-6.mjs. Reading s is the embedding (s = 0) or the stream after layer s - 1.
import fs from 'node:fs';
const D = JSON.parse(fs.readFileSync(new URL('../docs/data/3-6/lens.json', import.meta.url)));
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const P = (start) => D.prompts.find((p) => p.prompt.startsWith(start));
const last = (p, s) => p.cells[s][p.tokens.length - 1];
const firstTop = (p, tok) => { for (let s = 0; s < 13; s++) if (last(p, s).t[0] === tok) return s; return -1; };
const r2 = (v) => v.toFixed(2);

const C = P('The Colosseum');
const sR = firstTop(C, ' Rome');
check('t1: Rome first top after layer 8', sR === 9, `after layer ${sR - 1}, p=${last(C, sR).p[0]}`);
check('t1: probability 0.72 there', r2(last(C, sR).p[0]) === '0.72', last(C, sR).p[0]);
check('t1: Rome stays on top to the output', [9, 10, 11, 12].every((s) => last(C, s).t[0] === ' Rome'), [9, 10, 11, 12].map((s) => last(C, s).t[0]).join('|'));
check('t1: rows above show "the" then La, La, Ing', [1, 2, 3, 4, 5].every((s) => last(C, s).t[0] === ' the') && last(C, 6).t[0] === ' La' && last(C, 7).t[0] === ' La' && last(C, 8).t[0] === ' Ing',
  [...Array(9)].map((_, s) => last(C, s).t[0]).join('|'));
check('t1: embedding row predicts "destro"', last(C, 0).t[0] === ' destro', last(C, 0).t[0]);
const [a, m, b] = last(C, sR).chg;
check('t2: Rome 0.04 -> 0.17 (attention) -> 0.72 (MLP); MLP larger', r2(a) === '0.04' && r2(m) === '0.17' && r2(b) === '0.72' && (b - m) > (m - a), `${a} ${m} ${b}`);
const J = P('The capital of Japan');
const sT = firstTop(J, ' Tokyo');
const [ja, jm, jb] = last(J, sT).chg;
check('t3: Tokyo first top after layer 9; 0.07 -> 0.76 -> 0.86; attention larger', sT === 10 && r2(ja) === '0.07' && r2(jm) === '0.76' && r2(jb) === '0.86' && (jm - ja) > (jb - jm), `s=${sT} ${ja} ${jm} ${jb}`);
check('t4: Rome 0.86 after layer 9, 0.45 after layer 10, 0.02 at the output', r2(last(C, 10).p[0]) === '0.86' && r2(last(C, 11).p[0]) === '0.45' && r2(last(C, 12).p[0]) === '0.02',
  [10, 11, 12].map((s) => last(C, s).p[0]).join(' '));
for (const p of D.prompts) {
  const t11 = last(p, 11), t12 = last(p, 12);
  check(`t4: last layer lowers the top token, which stays on top: ${p.prompt}`, t11.t[0] === t12.t[0] && t12.p[0] < t11.p[0], `${t11.t[0]} ${t11.p[0]} -> ${t12.p[0]}`);
}
check('six prompts', D.prompts.length === 6, D.prompts.length);
console.log(fails ? `${fails} FAILED` : 'all ok');
process.exit(fails ? 1 : 0);
