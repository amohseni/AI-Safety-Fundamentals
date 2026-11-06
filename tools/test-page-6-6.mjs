// Claims on page 6.6, checked against the precomputed GPT-2 small generations (tools/6-6/steer.py).
// Run: node tools/test-page-6-6.mjs
import fs from 'node:fs';
const D = JSON.parse(fs.readFileSync(new URL('../docs/data/6-6/steer.json', import.meta.url)));
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const C = D.coefs, at = (k) => C.indexOf(k);
const cell = (p, v, L) => D.cells[`${p}|${v}|${L}`];
const r2 = (x) => x.toFixed(2), r1 = (x) => x.toFixed(1);
check('grid complete: 2 prompts x 2 vectors x 12 layers, 21 coefficients', Object.keys(D.cells).length === 48 && C.length === 21 && C[0] === -10 && C[20] === 10, Object.keys(D.cells).length);

// Task 1: park, Love-Hate, layer 0.
let c = cell(0, 'love', 0);
check('t1: unsteered fluency -1.8, sentiment 0.75', r1(c.flu[at(0)]) === '-1.8' && r2(c.sent[at(0)]) === '0.75', `${c.flu[at(0)]} ${c.sent[at(0)]}`);
check('t1: +2 sentiment 1.00, fluency -2.3', r2(c.sent[at(2)]) === '1.00' && r1(c.flu[at(2)]) === '-2.3', `${c.sent[at(2)]} ${c.flu[at(2)]}`);
check('t1: +3 sentiment above unsteered', c.sent[at(3)] > 0.8, c.sent[at(3)]);
check('t1: +5 fluency -3.2, +10 -4.1, sentiment +10 0.19', r1(c.flu[at(5)]) === '-3.2' && r1(c.flu[at(10)]) === '-4.1' && r2(c.sent[at(10)]) === '0.19', `${c.flu[at(5)]} ${c.flu[at(10)]} ${c.sent[at(10)]}`);
check('t1: +5 sample 1 repeats fragments', c.texts[at(5)][0].startsWith(' to the the and the and the it'), JSON.stringify(c.texts[at(5)][0].slice(0, 40)));
check('t1: fluency falls monotonically from +5 on only roughly; every value from +5 to +10 is at or below -3.1', C.slice(at(5)).every((k) => c.flu[at(k)] <= -3.15), C.slice(at(5)).map((k) => c.flu[at(k)]).join(' '));
const c6 = cell(0, 'love', 6);
check('t1: at layer 6 fluency stays near -2 (between -2.25 and -1.75) up to +10', C.slice(at(0)).every((k) => c6.flu[at(k)] >= -2.25 && c6.flu[at(k)] <= -1.75), C.slice(at(0)).map((k) => c6.flu[at(k)]).join(' '));

// Task 2: I hate you because, +4.
const s4 = [...Array(12)].map((_, L) => cell(1, 'love', L).sent[at(4)]);
check('t2: unsteered 0.05', r2(cell(1, 'love', 0).sent[at(0)]) === '0.05', cell(1, 'love', 0).sent[at(0)]);
check('t2: +4 at layers 0, 6, 11: 0.87, 1.00, 0.94', r2(s4[0]) === '0.87' && r2(s4[6]) === '1.00' && r2(s4[11]) === '0.94', s4.map(r2).join(' '));
check('t2: layers 7 to 10 only 0.23 to 0.49', r2(Math.min(...s4.slice(7, 11))) === '0.23' && r2(Math.max(...s4.slice(7, 11))) === '0.49', s4.slice(7, 11).map(r2).join(' '));
const c5 = cell(1, 'love', 5);
check('t2: layer 5: +2 and +3 give 0.01, +4 0.61', r2(c5.sent[at(2)]) === '0.01' && r2(c5.sent[at(3)]) === '0.01' && r2(c5.sent[at(4)]) === '0.61', `${c5.sent[at(2)]} ${c5.sent[at(3)]} ${c5.sent[at(4)]}`);

// Task 3: I hate you because, layer 6, negative coefficients.
c = cell(1, 'love', 6);
const neg = C.filter((k) => k <= -2).map((k) => c.sent[at(k)]);
check('t3: every coefficient from -2 to -10 raises the score, to 0.16..0.75', Math.min(...neg) > c.sent[at(0)] && r2(Math.min(...neg)) === '0.16' && r2(Math.max(...neg)) === '0.75', neg.map(r2).join(' '));
check('t3: -3 gives 0.51, -8 gives 0.24, -1 gives 0.02', r2(c.sent[at(-3)]) === '0.51' && r2(c.sent[at(-8)]) === '0.24' && r2(c.sent[at(-1)]) === '0.02', `${c.sent[at(-3)]} ${c.sent[at(-8)]} ${c.sent[at(-1)]}`);

// Task 4: weddings, park.
const full = (L) => C.filter((k) => k >= 4 && cell(0, 'wed', L).wed[at(k)] === 1);
check('t4: all 8 mention weddings at some coefficient in +4..+10 at layers 1, 2, 11', full(1).length && full(2).length && full(11).length, `${full(1)} | ${full(2)} | ${full(11)}`);
check('t4: at layer 2 that happens at +4 and +5 only', full(2).join() === '4,5', full(2).join());
const w2 = cell(0, 'wed', 2);
check('t4: layer 2 falls to 12% at +9 and +10, fluency -4.8 at +9', w2.wed[at(9)] === 0.125 && w2.wed[at(10)] === 0.125 && r1(w2.flu[at(9)]) === '-4.8', `${w2.wed[at(9)]} ${w2.wed[at(10)]} ${w2.flu[at(9)]}`);
const mid = [3, 4, 5, 6, 7, 8, 9, 10].map((L) => Math.max(...cell(0, 'wed', L).wed));
check('t4: layers 3 to 10: at most 3 of 8 at any coefficient', Math.max(...mid) <= 0.375 + 1e-9, mid.join(' '));
check('t4: no layer reaches 100% for weddings except 1, 2, 11', [...Array(12)].every((_, L) => [1, 2, 11].includes(L) || Math.max(...cell(0, 'wed', L).wed) < 1), '');
console.log(fails ? `${fails} FAILED` : 'all ok');
process.exit(fails ? 1 : 0);
