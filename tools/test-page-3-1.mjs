// Numbers and splits quoted on page 3.1, recomputed with the vendored tokenizer and the page's tiny BPE.
// Run: node tools/test-page-3-1.mjs
import { Tiktoken } from '../docs/vendor/js-tiktoken-1.0.21/lite.js';
import gpt2 from '../docs/vendor/js-tiktoken-1.0.21/gpt2.js';
import o200k from '../docs/vendor/js-tiktoken-1.0.21/o200k_base.js';
import { TinyBPE } from '../docs/js/lib/3-models.js';
import { CORPUS, EXAMPLES } from '../docs/js/page/3-1-data.js';

const enc = { gpt2: new Tiktoken(gpt2), o200k: new Tiktoken(o200k) };
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const split = (e, s) => enc[e].encode(s).map((i) => enc[e].decode([i]));
const show = (e, s) => split(e, s).map((t) => JSON.stringify(t)).join(' | ');

check('GPT-2 vocabulary size', enc.gpt2.rankMap.size + Object.keys(gpt2.special_tokens).length === 50257, `${enc.gpt2.rankMap.size} ordinary + ${Object.keys(gpt2.special_tokens).length} special`);
check('GPT-2 has 256 single-byte tokens', [...enc.gpt2.textMap.values()].filter((b) => b.length === 1).length === 256, '256');
check('o200k_base ordinary tokens', enc.o200k.rankMap.size === 199998, enc.o200k.rankMap.size);

const fig = 'Unbelievably, the model tokenizes strawberry.';
check('figure sentence, GPT-2', enc.gpt2.encode(fig).length === 11, show('gpt2', fig) + '  ids ' + enc.gpt2.encode(fig).join(' '));
check('" strawberry" is one GPT-2 token', enc.gpt2.encode(' strawberry').length === 1, enc.gpt2.encode(' strawberry'));
check('"strawberry" is three GPT-2 tokens', show('gpt2', 'strawberry') === '"st" | "raw" | "berry"', show('gpt2', 'strawberry'));
check('"strawberry" is three GPT-4o tokens', split('o200k', 'strawberry').length === 3, show('o200k', 'strawberry'));
check('12345678, GPT-2', show('gpt2', '12345678') === '"123" | "45" | "678"', show('gpt2', '12345678'));
check('12345678, GPT-4o', show('o200k', '12345678') === '"123" | "456" | "78"', show('o200k', '12345678'));
check('1234 + 5678 = 6912, GPT-2', show('gpt2', '1234 + 5678 = 6912') === '"12" | "34" | " +" | " 5" | "678" | " =" | " 69" | "12"', show('gpt2', '1234 + 5678 = 6912'));
const en = EXAMPLES.find((x) => x.key === 'english').text, fa = EXAMPLES.find((x) => x.key === 'persian').text;
const cnt = (e, s) => enc[e].encode(s).length;
check('English example, GPT-2', cnt('gpt2', en) === 5, `${cnt('gpt2', en)} tokens for ${[...en].length} characters`);
check('Persian example, GPT-2', cnt('gpt2', fa) === 29, `${cnt('gpt2', fa)} tokens for ${[...fa].length} characters (ratio ${(cnt('gpt2', fa) / cnt('gpt2', en)).toFixed(1)})`);
check('English example, GPT-4o', cnt('o200k', en) === 5, cnt('o200k', en));
check('Persian example, GPT-4o', cnt('o200k', fa) === 9, `${cnt('o200k', fa)} (ratio ${(cnt('o200k', fa) / cnt('o200k', en)).toFixed(1)})`);
for (const x of EXAMPLES) console.log(`     example ${x.key}: GPT-2 ${cnt('gpt2', x.text)}, GPT-4o ${cnt('o200k', x.text)}: ${show('gpt2', x.text)}`);

// The tiny corpus: which merges come first, and when " the" becomes one token.
const bpe = new TinyBPE(CORPUS);
const top = bpe.pairs().slice(0, 4).map((p) => `${JSON.stringify(p.a + p.b)}:${p.n}`).join(' ');
check('first merge is t + h', bpe.pairs()[0].a === 't' && bpe.pairs()[0].b === 'h' && bpe.pairs()[0].n > bpe.pairs()[1].n, top);
const start = bpe.tokens().length;
let k = 0, at = null;
const log = [];
while (bpe.merge() && k < 40) { k++; const m = bpe.merges[k - 1]; log.push(`${k}:${JSON.stringify(m.a)}+${JSON.stringify(m.b)}(${m.n})`); if (at === null && bpe.words.get(' the').length === 1) at = k; }
check('" the" is one token after 3 merges', at === 3, `after ${at} merges; ${log.slice(0, 8).join(' ')}`);
const b2 = new TinyBPE(CORPUS); for (let i = 0; i < 3; i++) b2.merge();
console.log(`     corpus: ${CORPUS.length} characters, ${start} tokens at the start, ${b2.tokens().length} after 3 merges, vocabulary ${new TinyBPE(CORPUS).vocabSize()} then ${b2.vocabSize()}`);
process.exit(fails ? 1 : 0);
