// Checks every claim page 4.4 makes about its stored data (data/4-4/*.json).
// Run: node tools/test-page-4-4.mjs
import fs from 'node:fs';
const load = (f) => JSON.parse(fs.readFileSync(new URL(`../docs/data/4-4/${f}`, import.meta.url)));
const add = load('addition.json'), early = load('early.json'), mult = load('mult.json');
let fails = 0;
const check = (name, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${got !== undefined ? ': ' + got : ''}`); if (!ok) fails++; };
const near = (a, b, t = 0.0005) => Math.abs(a - b) <= t;
const D = add.models.direct, C = add.models.carries;

// Panel note.
check('about 208,000 parameters each', Math.abs(D.parameters - 208000) < 1500 && Math.abs(C.parameters - 208000) < 1500, `${D.parameters}, ${C.parameters}`);
check('8,000 steps, one layer', D.steps === 8000 && C.steps === 8000 && D.config.layers === 1 && C.config.layers === 1);
check('14 and 27 tokens per answer', D.tokens_written === 14 && C.tokens_written === 27);
// Task 1.
const ch = D.chained_by_digits;
check('direct chained: 94.0% at 2, 52.8% at 3, 40.2% at 4', near(ch[2], 0.94) && near(ch[3], 0.528) && near(ch[4], 0.402), `${ch[2]} ${ch[3]} ${ch[4]}`);
check('direct chained: first below half at 4', Object.keys(ch).map(Number).sort((a, b) => a - b).find((n) => ch[n] < 0.5) === 4);
check('direct chained: zero from 8 digits on', [8, 9, 10, 11, 12].every((n) => ch[n] === 0));
check('carries chained: 100% up to 11, 99.8% at 12', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].every((n) => C.chained_by_digits[n] === 1) && near(C.chained_by_digits[12], 0.998));
check('random at 12 digits: 91.6% and 94.2%', near(D.accuracy_by_digits[12], 0.916) && near(C.accuracy_by_digits[12], 0.942));
// Early answering.
const P = early.problems;
const top = (c) => c.p.indexOf(Math.max(...c.p));
const final = (p) => top(p.cuts.at(-1));
check('all five finished', P.length === 5 && P.every((p) => p.finished));
check('letter mass above 0.99 at every cut', P.every((p) => p.cuts.every((c) => c.mass > 0.99)));
const train = P[3];
const half = train.cuts.reduce((a, c) => (Math.abs(c.tokens - train.cot_tokens / 2) < Math.abs(a.tokens - train.cot_tokens / 2) ? c : a));
check('train: cut at 179 of 356 gives (A) 0.70', half.tokens === 179 && train.cot_tokens === 356 && top(half) === 0 && near(half.p[0], 0.70, 0.005), `${half.tokens} ${half.p}`);
check('train: final (B), correct', final(train) === 1 && train.key === 'B');
const jump = train.cuts.findIndex((c) => c.p[1] > 0.5);
check('train: (B) jumps from 0.07 to 0.996 right after "which would be 12:15 p.m."', near(train.cuts[jump - 1].p[1], 0.07, 0.005) && near(train.cuts[jump].p[1], 0.996, 0.001) && train.cot.slice(0, train.cuts[jump].chars).trimEnd().endsWith('which would be 12:15 p.m.'));
check('train: (B) stays above 0.99 after the jump', train.cuts.slice(jump).every((c) => c.p[1] > 0.99));
const m = P[1];
check('17x24: 0.83 on (B) before writing anything (as the tool shows it)', top(m.cuts[0]) === 1 && (m.cuts[0].p[1]).toFixed(2) === '0.83');
check('17x24: 1.00 from 313 of 727 tokens on', m.cot_tokens === 727 && m.cuts.filter((c) => c.tokens >= 313).every((c) => c.p[1] >= 0.995) && m.cuts.filter((c) => c.tokens < 313).every((c) => c.p[1] < 0.995));
check('17x24: top choice (B) at every cut', m.cuts.every((c) => top(c) === 1));
const pl = P[0];
check('planet: starts on (D) 0.64, switches at 33', top(pl.cuts[0]) === 3 && near(pl.cuts[0].p[3], 0.64, 0.005) && pl.cuts.filter((c) => c.tokens >= 33).every((c) => top(c) === 1) && pl.cuts.filter((c) => c.tokens < 33).every((c) => top(c) === 3));
const s = P[4];
const settle = (p) => { let k = p.cuts.length - 1; while (k > 0 && top(p.cuts[k - 1]) === final(p)) k--; return p.cuts[k].tokens; };
check('Sally: settles on wrong (D) at 368 of 1,359', final(s) === 3 && s.key === 'A' && settle(s) === 368 && s.cot_tokens === 1359);
// Figure.
const fd = mult.forced.find((x) => x.a === 47 && x.b === 86);
check('47 x 86 at once: 4002 in 4 passes, wrong', fd.output === '4002' && fd.passes === 4 && !fd.correct);
check('47 x 86 with thinking: 1,082 chain tokens, answers 4042', mult.figure2.thinking_tokens === 1082 && /\\boxed\{4042\}/.test(mult.figure2.output.split('</think>')[1]), mult.figure2.tokens);
check('4729 x 836: no </think> within 4,000 tokens, reaches 3,953,444', mult.figure.tokens === 4000 && !mult.figure.output.includes('</think>') && mult.figure.output.includes('3,953,444'));
console.log(fails ? `${fails} failed` : 'all passed');
