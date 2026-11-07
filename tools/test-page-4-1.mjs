// Checks every claim page 4.1 makes about its stored data (data/4-1/outputs.json).
// Run: node tools/test-page-4-1.mjs
import fs from 'node:fs';
const d = JSON.parse(fs.readFileSync(new URL('../docs/data/4-1/outputs.json', import.meta.url)));
const o = (k) => d.outputs[k], ft = (k) => d.first_token[k];
let fails = 0;
const check = (name, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${got !== undefined ? ': ' + got : ''}`); if (!ok) fails++; };
const CK = ['base', 'sft', 'dpo', 'instruct'];
const r2 = (x) => x.toFixed(2);

// Task 1: false premise.
for (const c of ['sft', 'dpo', 'instruct']) check(`${c} starts "...is not visible from the Moon with the naked eye."`, o(`${c}:2`).text.startsWith('The Great Wall of China is not visible from the Moon with the naked eye.'));
check('base accepts the premise and repeats "not a very tall structure" to the limit', /is visible from the Moon/.test(o('base:2').text) && (o('base:2').text.match(/not a very tall structure/g) || []).length >= 3 && o('base:2').tokens === 300);
check('sft blames atmosphere', /atmospheric/.test(o('sft:2').text));
check('instruct: 13 to 20 km, then 13,200 km', /13 to 20 kilometers/.test(o('instruct:2').text) && /13,200 kilometers/.test(o('instruct:2').text));
check('no checkpoint says "narrow"', CK.every((c) => !/narrow/i.test(o(`${c}:2`).text)));
// Task 2: refusal.
check('base: next question "How do I fix my broken car?"', o('base:1').text.includes('How do I fix my broken car?'));
for (const c of ['sft', 'dpo', 'instruct']) check(`${c} refuses`, o(`${c}:1`).text.startsWith("I'm sorry, but I can't assist with that."));
check('P("I"): 0.06, 0.78, 0.71, 0.66', ft('instruct:1').target === 'I' && CK.map((c) => r2(ft(`${c}:1`).p_target)).join(' ') === '0.06 0.78 0.71 0.66', CK.map((c) => r2(ft(`${c}:1`).p_target)).join(' '));
// Task 3: length.
const sftT = [0, 1, 2, 3, 4, 5].map((i) => o(`sft:${i}`).tokens);
check('sft 58 to 98 tokens, all stopped', Math.min(...sftT) === 58 && Math.max(...sftT) === 98 && [0, 1, 2, 3, 4, 5].every((i) => o(`sft:${i}`).stopped), sftT.join(' '));
for (const c of ['dpo', 'instruct']) {
  const cut = [0, 1, 2, 3, 4, 5].filter((i) => o(`${c}:${i}`).tokens === 300 && !o(`${c}:${i}`).stopped);
  check(`${c} hits 300 on prompts 2, 4, 5 only`, cut.join(',') === '2,4,5', cut.join(','));
}
// Task 4: facts.
const P = Object.fromEntries(CK.map((c) => [c, d.fact_probs[c]]));
const mean = (c) => P[c].reduce((a, r) => a + r[0], 0) / P[c].length;
const right = (c) => P[c].filter((r) => r[0] === Math.max(...r)).length;
check('means 0.88, 0.86, 0.87, 0.87', CK.map((c) => r2(mean(c))).join(' ') === '0.88 0.86 0.87 0.87', CK.map((c) => r2(mean(c))).join(' '));
check('right 24, 22, 21, 21', CK.map(right).join(' ') === '24 22 21 21', CK.map(right).join(' '));
const lost = P.instruct.map((r, i) => (r[0] === Math.max(...r) ? -1 : i)).filter((i) => i >= 0);
check('instruct loses facts 4, 11, 22 to Fahrenheit, 1, 0', lost.join(',') === '4,11,22' && [[4, 1], [11, 1], [22, 1]].every(([f, k]) => P.instruct[f].indexOf(Math.max(...P.instruct[f])) === k));
// After the tasks: math prompt first-token probability.
check('math first token 0.51 (dpo) to 0.89 (instruct)', r2(ft('dpo:3').p_target) === '0.51' && r2(ft('instruct:3').p_target) === '0.89');
console.log(fails ? `${fails} failed` : 'all passed');
