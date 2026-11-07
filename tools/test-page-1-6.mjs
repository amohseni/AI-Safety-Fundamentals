// Checks every claim pages 1.6 makes about the stored outputs (data/1-6/*.json).
// Run: node tools/test-page-1-6.mjs
import fs from 'node:fs';
const d = JSON.parse(fs.readFileSync(new URL('../docs/data/1-6/outputs.json', import.meta.url)));
const w = JSON.parse(fs.readFileSync(new URL('../docs/data/1-6/weight_diff.json', import.meta.url)));
const o = (k) => d.outputs[k];
let fails = 0;
const check = (name, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${got !== undefined ? ': ' + got : ''}`); if (!ok) fails++; };

// Task 1: the base model continues the document only on the Einstein prompt (6): its text opens with a new question.
const opensWithQuestion = [0, 1, 2, 3, 4, 5, 6, 7].filter((i) => /^\s*[^.!\n]*\?/.test(o(`base:${i}`).text.split('\n')[0]));
check('base opens with a further question only on prompt 6', JSON.stringify(opensWithQuestion) === '[6]', JSON.stringify(opensWithQuestion));
check('base text on prompt 6 starts "Also, how did he manage"', o('base:6').text.trim().startsWith('Also, how did he manage'));
const baseCut = [0, 1, 2, 3, 4, 5, 6, 7].filter((i) => !o(`base:${i}`).stopped);
check('base reached its 200-token limit on three prompts', baseCut.length === 3 && baseCut.every((i) => o(`base:${i}`).tokens === 200), JSON.stringify(baseCut));
check('chat model stopped by itself on all eight', [0, 1, 2, 3, 4, 5, 6, 7].every((i) => o(`chat:${i}`).stopped));

// Task 2: letter count.
check('chat says once', /\*\*once\*\*/.test(o('chat:1').text) && o('chat:1').tokens === 28, o('chat:1').tokens);
check('reasoning says 3, with 319 thinking tokens', /\*\*3 times\*\*/.test(o('reasoning:1').text) && o('reasoning:1').thinking_tokens === 319);
check('reasoning asks "Wait, is that correct?" and lists positions 3, 8, 9', o('reasoning:1').text.includes('Wait, is that correct?') && o('reasoning:1').text.includes('positions 3, 8, and 9'));
check('base says twice', /twice/.test(o('base:1').text));

// Task 3: bat and ball.
const bb = o('reasoning:5');
const frac = bb.text.indexOf('0.05') / bb.text.length;
check('chat finds 0.05', /0\.05/.test(o('chat:5').text));
check('reasoning hits the 3,000-token limit with no </think>', !bb.stopped && bb.tokens === 3000 && !bb.text.includes('</think>'));
check('0.05 first appears about 350 tokens in', Math.abs(frac * 3000 - 350) < 60, Math.round(frac * 3000));
check('reasoning later gets 10 cents by taking the bat as $1.00', bb.text.includes('the ball would be 10 cents'));

// Task 4: false premises.
for (const m of ['base', 'chat', 'reasoning']) check(`${m} does not mention the photoelectric effect`, !/photoelectric/i.test(o(`${m}:6`).text));
check('base Mars: David Scott, 1976', /David Scott/.test(o('base:7').text) && /1976/.test(o('base:7').text));
check('chat Mars: Neil Armstrong, then no record', /Neil Armstrong/.test(o('chat:7').text) && /no record of a person walking on Mars/.test(o('chat:7').text));
check('reasoning Mars: Mariner 2', /Mariner 2/.test(o('reasoning:7').text.split('</think>')[1]));

// After the tasks: parameter change.
check('overall relative change 6.3%', Math.abs(w.overall_relative_change - 0.063) < 0.0006, w.overall_relative_change);
check('median tensor 10.2%, max 18.6%, 310 tensors', Math.abs(w.median_tensor_relative_change - 0.102) < 0.0006 && Math.abs(w.max_tensor_relative_change - 0.186) < 0.0006 && w.tensors === 310);
console.log(fails ? `${fails} failed` : 'all passed');
